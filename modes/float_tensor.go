package modes

import (
	"encoding/binary"
	"math"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// FloatTensorSchema is the canonical FLOAT_TENSOR descriptor schema.
const FloatTensorSchema = "weft.ai.float_tensor.v1"

// FloatTensorLevel is the frame level for FLOAT_TENSOR. No zstd frame is
// used, so the field is 0.
const FloatTensorLevel = 0

// FloatDtype describes one byte-aligned float element layout.
type FloatDtype struct {
	Name         string
	SignBits     uint64
	ExponentBits uint64
	MantissaBits uint64
}

// ParseFloatDtype parses a canonical float-tensor dtype name.
func ParseFloatDtype(name string) (FloatDtype, error) {
	switch name {
	case "bf16":
		return FloatDtype{Name: name, SignBits: 1, ExponentBits: 8, MantissaBits: 7}, nil
	case "fp16":
		return FloatDtype{Name: name, SignBits: 1, ExponentBits: 5, MantissaBits: 10}, nil
	case "fp32":
		return FloatDtype{Name: name, SignBits: 1, ExponentBits: 8, MantissaBits: 23}, nil
	case "fp8-e4m3":
		return FloatDtype{Name: name, SignBits: 1, ExponentBits: 4, MantissaBits: 3}, nil
	case "fp8-e5m2":
		return FloatDtype{Name: name, SignBits: 1, ExponentBits: 5, MantissaBits: 2}, nil
	default:
		return FloatDtype{}, werrors.Malformedf(
			"unknown float tensor dtype %q (v1 allows bf16, fp16, fp32, fp8-e4m3, fp8-e5m2)",
			name)
	}
}

// ElementBits is the total element width in bits.
func (d FloatDtype) ElementBits() uint64 {
	return d.SignBits + d.ExponentBits + d.MantissaBits
}

// ElementSize is the element width in bytes (byte-aligned for every v1 dtype).
func (d FloatDtype) ElementSize() uint64 {
	return d.ElementBits() / 8
}

// FieldBits returns the (sign, exponent, mantissa) widths.
func (d FloatDtype) FieldBits() (uint64, uint64, uint64) {
	return d.SignBits, d.ExponentBits, d.MantissaBits
}

// FloatTensorMeta is a parsed FLOAT_TENSOR descriptor.
type FloatTensorMeta struct {
	Dtype            FloatDtype
	Shape            []uint64
	ElementCount     uint64
	SignBits         uint64
	ExponentBits     uint64
	MantissaBits     uint64
	ExponentPlaneLen uint64
	MantissaPlaneLen uint64
}

func metaByteField(m map[string]wcb.Value, key string) (uint64, error) {
	value, err := metaU64(m, key)
	if err != nil {
		return 0, err
	}
	if value > 255 {
		return 0, werrors.Malformedf(
			"float tensor %q must fit an unsigned byte, got %d", key, value)
	}
	return value, nil
}

func floatTensorProduct(shape []uint64, lim limits.Limits) (uint64, error) {
	if len(shape) == 0 {
		return 0, werrors.NewMalformed("float tensor shape rank must be at least 1")
	}
	if uint64(len(shape)) > lim.MaxTensorRank {
		return 0, werrors.LimitExceededf(
			"float tensor rank %d exceeds limit %d", len(shape), lim.MaxTensorRank)
	}
	product := uint64(1)
	for _, dim := range shape {
		if dim == 0 {
			return 0, werrors.NewMalformed(
				"float tensor shape dimensions must be greater than zero")
		}
		if product > math.MaxUint64/dim {
			return 0, werrors.NewLimitExceeded("float tensor shape product overflows u64")
		}
		product *= dim
	}
	if product > lim.MaxTensorElements {
		return 0, werrors.LimitExceededf(
			"float tensor element count %d exceeds limit %d", product, lim.MaxTensorElements)
	}
	return product, nil
}

// pushBits appends count low bits of value, most significant first.
func pushBits(out []byte, bitLen *uint64, value, count uint64) []byte {
	for shift := int(count) - 1; shift >= 0; shift-- {
		bit := (value >> uint(shift)) & 1
		byteIndex := *bitLen / 8
		if byteIndex == uint64(len(out)) {
			out = append(out, 0)
		}
		out[byteIndex] |= byte(bit) << uint(7-(*bitLen%8))
		*bitLen = *bitLen + 1
	}
	return out
}

// SplitFloatTensorPayload splits a little-endian float payload into the
// huffman-canonical-v1 exponent plane and the raw sign+mantissa plane.
func SplitFloatTensorPayload(payload []byte, dtype FloatDtype, elementCount uint64) ([]byte, []byte, error) {
	expectedBytes, err := limits.MulChecked(elementCount, dtype.ElementSize(),
		"float tensor decoded byte count")
	if err != nil {
		return nil, nil, err
	}
	if uint64(len(payload)) != expectedBytes {
		return nil, nil, werrors.Malformedf(
			"float tensor payload is %d bytes but the descriptor requires %d",
			len(payload), expectedBytes)
	}
	elementBytes, err := limits.ToInt(dtype.ElementSize(), "float tensor element size")
	if err != nil {
		return nil, nil, err
	}
	count, err := limits.ToInt(elementCount, "float tensor element count")
	if err != nil {
		return nil, nil, err
	}
	capHint := count
	if capHint > 1<<20 {
		capHint = 1 << 20
	}
	exponentPlane := make([]byte, 0, capHint)
	mantissaPlane := make([]byte, 0, capHint)
	var bitLen uint64
	mantissaMask := uint64(1)<<dtype.MantissaBits - 1
	exponentMask := uint64(1)<<dtype.ExponentBits - 1
	for offset := 0; offset+elementBytes <= len(payload); offset += elementBytes {
		var word uint64
		for index := 0; index < elementBytes; index++ {
			word |= uint64(payload[offset+index]) << uint(8*index)
		}
		exponent := (word >> uint(dtype.MantissaBits)) & exponentMask
		mantissa := word & mantissaMask
		sign := (word >> uint(dtype.ExponentBits+dtype.MantissaBits)) & 1
		exponentPlane = append(exponentPlane, byte(exponent))
		mantissaPlane = pushBits(mantissaPlane, &bitLen, sign, dtype.SignBits)
		mantissaPlane = pushBits(mantissaPlane, &bitLen, mantissa, dtype.MantissaBits)
	}
	compressed := HuffmanEncode(exponentPlane)
	return compressed, mantissaPlane, nil
}

// JoinFloatTensorPayload reconstructs the exact little-endian element bytes
// from the two planes.
func JoinFloatTensorPayload(
	exponentPlane, mantissaPlane []byte,
	dtype FloatDtype,
	elementCount uint64,
	lim limits.Limits,
) ([]byte, error) {
	if uint64(len(exponentPlane)) != elementCount {
		return nil, werrors.Corruptf(
			"float tensor exponent plane has %d bytes but element_count is %d",
			len(exponentPlane), elementCount)
	}
	decodedBytes, err := limits.MulChecked(elementCount, dtype.ElementSize(),
		"float tensor decoded byte count")
	if err != nil {
		return nil, err
	}
	if err := limits.CheckLen(decodedBytes, lim.MaxDecodedBytes, "float tensor decoded bytes"); err != nil {
		return nil, err
	}
	mantissaBits, err := limits.MulChecked(elementCount,
		dtype.SignBits+dtype.MantissaBits, "float tensor mantissa bit count")
	if err != nil {
		return nil, err
	}
	expectedMantissaBytes, err := limits.AddChecked(mantissaBits, 7,
		"float tensor mantissa length")
	if err != nil {
		return nil, err
	}
	expectedMantissaBytes /= 8
	if uint64(len(mantissaPlane)) != expectedMantissaBytes {
		return nil, werrors.Corruptf(
			"float tensor mantissa plane has %d bytes but descriptor arithmetic requires %d",
			len(mantissaPlane), expectedMantissaBytes)
	}
	capacity, err := limits.ToInt(decodedBytes, "float tensor decoded bytes")
	if err != nil {
		return nil, err
	}
	// The output is exactly the declared decoded byte count, which was bounded
	// by MaxDecodedBytes above, so the allocation is bounded as well.
	out := make([]byte, capacity)
	elementBytes := int(dtype.ElementSize())
	// BF16 is the one width where the sign+mantissa plane is exactly one byte
	// per element, so the join is a direct two-byte pack. Go has no SIMD
	// intrinsics, so this is a word-level kernel: it avoids the 64-bit
	// accumulator and the per-element switch below.
	if dtype.Name == "bf16" {
		for index := range exponentPlane {
			exponent := exponentPlane[index]
			plane := mantissaPlane[index]
			out[index*2] = (plane & 0x7f) | ((exponent & 1) << 7)
			out[index*2+1] = ((exponent >> 1) & 0x7f) | (plane & 0x80)
		}
		return out, nil
	}
	mantissaMask := (uint64(1) << uint(dtype.MantissaBits)) - 1
	width := dtype.SignBits + dtype.MantissaBits

	// MSB-first 64-bit accumulator holding the next accumulatedBits bits in its
	// high positions and zeros below: refill eight bits at a time while a whole
	// byte still fits, then read the whole sign+mantissa width in one shift and
	// drop it by shifting left. accumulatedBits >= width always holds because
	// the mantissa plane is exactly ceil(elementCount * width / 8) bytes and
	// width <= 24.
	var accumulator uint64
	var accumulatedBits uint
	planePos := 0
	for index, exponentByte := range exponentPlane {
		if uint64(exponentByte)>>uint(dtype.ExponentBits) != 0 {
			return nil, werrors.NewCorrupt(
				"float tensor exponent byte exceeds the dtype exponent width")
		}
		for accumulatedBits <= 56 && planePos < len(mantissaPlane) {
			accumulator |= uint64(mantissaPlane[planePos]) << (56 - accumulatedBits)
			planePos++
			accumulatedBits += 8
		}
		value := accumulator >> (64 - uint(width))
		accumulator <<= uint(width)
		accumulatedBits -= uint(width)

		sign := value >> uint(dtype.MantissaBits)
		mantissa := value & mantissaMask
		word := sign<<uint(dtype.ExponentBits+dtype.MantissaBits) |
			uint64(exponentByte)<<uint(dtype.MantissaBits) | mantissa
		start := index * elementBytes
		switch elementBytes {
		case 1:
			out[start] = byte(word)
		case 2:
			binary.LittleEndian.PutUint16(out[start:], uint16(word))
		case 4:
			binary.LittleEndian.PutUint32(out[start:], uint32(word))
		default:
			var buf [8]byte
			binary.LittleEndian.PutUint64(buf[:], word)
			copy(out[start:start+elementBytes], buf[:elementBytes])
		}
	}
	usedBits := uint64(len(mantissaPlane)) * 8
	padding := usedBits - mantissaBits
	if padding > 0 {
		last := mantissaPlane[len(mantissaPlane)-1]
		mask := byte(1<<uint(padding)) - 1
		if last&mask != 0 {
			return nil, werrors.NewCorrupt(
				"float tensor mantissa padding bits must be zero")
		}
	}
	return out, nil
}

// EncodeFloatTensor encodes a FLOAT_TENSOR frame (WEFT-AI-CODEC §5).
func EncodeFloatTensor(payload []byte, dtype FloatDtype, shape []uint64, lim limits.Limits) ([]byte, error) {
	if err := limits.CheckLen(uint64(len(payload)), lim.MaxDecodedBytes, "float tensor payload"); err != nil {
		return nil, err
	}
	if dtype.ElementBits()%8 != 0 {
		return nil, werrors.Malformedf(
			"float tensor dtype %q is not byte-aligned", dtype.Name)
	}
	elementCount, err := floatTensorProduct(shape, lim)
	if err != nil {
		return nil, err
	}
	decodedBytes, err := limits.MulChecked(elementCount, dtype.ElementSize(),
		"float tensor decoded byte count")
	if err != nil {
		return nil, err
	}
	if err := limits.CheckLen(decodedBytes, lim.MaxDecodedBytes, "float tensor decoded bytes"); err != nil {
		return nil, err
	}
	if decodedBytes != uint64(len(payload)) {
		return nil, werrors.Malformedf(
			"float tensor payload is %d bytes but shape %v x %s is %d bytes",
			len(payload), shape, dtype.Name, decodedBytes)
	}

	exponentPlane, mantissaPlane, err := SplitFloatTensorPayload(payload, dtype, elementCount)
	if err != nil {
		return nil, err
	}
	stored := make([]byte, 0, len(exponentPlane)+len(mantissaPlane))
	stored = append(stored, exponentPlane...)
	stored = append(stored, mantissaPlane...)

	shapeItems := make([]wcb.Value, 0, len(shape))
	for _, dim := range shape {
		shapeItems = append(shapeItems, wcb.U64Value(dim))
	}
	meta := wcb.MapValue(map[string]wcb.Value{
		"schema":             wcb.TextValue(FloatTensorSchema),
		"dtype":              wcb.TextValue(dtype.Name),
		"shape":              wcb.ArrayValue(shapeItems),
		"element_count":      wcb.U64Value(elementCount),
		"sign_bits":          wcb.U64Value(dtype.SignBits),
		"exponent_bits":      wcb.U64Value(dtype.ExponentBits),
		"mantissa_bits":      wcb.U64Value(dtype.MantissaBits),
		"exponent_coder":     wcb.TextValue(HuffmanCoderID),
		"exponent_plane_len": wcb.U64Value(uint64(len(exponentPlane))),
		"mantissa_plane_len": wcb.U64Value(uint64(len(mantissaPlane))),
		"payload_hash":       wcb.BytesValue(hash.Domain(hash.DomainFloatTensorPayload, stored).Bytes()),
	})
	body, err := buildAIBody(meta, stored, lim)
	if err != nil {
		return nil, err
	}
	parts := frame.PayloadParts(
		frame.ModeFloatTensor, 0, uint16(FloatTensorLevel), decodedBytes,
		hash.Domain(hash.DomainChunk, payload), body)
	return frame.Encode(parts, lim)
}

func parseFloatTensorBody(f *frame.Frame, lim limits.Limits) (FloatTensorMeta, []byte, error) {
	descriptorBytes, stored, err := splitAIBodyPrefix(f.Body)
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	value, err := wcb.Decode(descriptorBytes, lim)
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	if value.Kind != wcb.Map {
		return FloatTensorMeta{}, nil, werrors.NewMalformed(
			"float tensor metadata must be a WCB map")
	}
	m := value.Map
	if err := checkExactKeys(m, []string{
		"schema", "dtype", "shape", "element_count",
		"sign_bits", "exponent_bits", "mantissa_bits",
		"exponent_coder", "exponent_plane_len", "mantissa_plane_len", "payload_hash",
	}, "float tensor"); err != nil {
		return FloatTensorMeta{}, nil, err
	}
	schema, err := metaText(m, "schema")
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	if schema != FloatTensorSchema {
		return FloatTensorMeta{}, nil, werrors.Malformedf(
			"unsupported float tensor schema %q", schema)
	}
	dtypeName, err := metaText(m, "dtype")
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	dtype, err := ParseFloatDtype(dtypeName)
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	exponentCoder, err := metaText(m, "exponent_coder")
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	if exponentCoder != HuffmanCoderID {
		return FloatTensorMeta{}, nil, werrors.Malformedf(
			"unsupported float tensor exponent coder %q (v1 allows %q)",
			exponentCoder, HuffmanCoderID)
	}
	signBits, err := metaByteField(m, "sign_bits")
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	exponentBits, err := metaByteField(m, "exponent_bits")
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	mantissaBits, err := metaByteField(m, "mantissa_bits")
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	wantSign, wantExponent, wantMantissa := dtype.FieldBits()
	if signBits != wantSign || exponentBits != wantExponent || mantissaBits != wantMantissa {
		return FloatTensorMeta{}, nil, werrors.Malformedf(
			"float tensor sub-field widths %d, %d, %d disagree with dtype %q (%d, %d, %d)",
			signBits, exponentBits, mantissaBits, dtype.Name,
			wantSign, wantExponent, wantMantissa)
	}
	elementBits := signBits + exponentBits + mantissaBits
	if elementBits%8 != 0 {
		return FloatTensorMeta{}, nil, werrors.NewMalformed(
			"float tensor element width is not byte-aligned")
	}
	elementSize := elementBits / 8
	shape, err := metaShape(m)
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	declaredCount, err := metaU64(m, "element_count")
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	expectedCount, err := floatTensorProduct(shape, lim)
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	if declaredCount != expectedCount {
		return FloatTensorMeta{}, nil, werrors.Malformedf(
			"float tensor element_count %d does not match shape product %d",
			declaredCount, expectedCount)
	}
	decodedBytes, err := limits.MulChecked(declaredCount, elementSize,
		"float tensor decoded byte count")
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	if err := limits.CheckLen(decodedBytes, lim.MaxDecodedBytes, "float tensor decoded bytes"); err != nil {
		return FloatTensorMeta{}, nil, err
	}
	if decodedBytes != f.Header.DecodedLen {
		return FloatTensorMeta{}, nil, werrors.Corruptf(
			"float tensor decoded_len is %d but the descriptor requires %d",
			f.Header.DecodedLen, decodedBytes)
	}

	mantissaBitTotal, err := limits.MulChecked(declaredCount, signBits+mantissaBits,
		"float tensor mantissa bit count")
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	expectedMantissaBytes, err := limits.AddChecked(mantissaBitTotal, 7,
		"float tensor mantissa length")
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	expectedMantissaBytes /= 8
	declaredMantissaLen, err := metaU64(m, "mantissa_plane_len")
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	if declaredMantissaLen != expectedMantissaBytes {
		return FloatTensorMeta{}, nil, werrors.Malformedf(
			"float tensor mantissa_plane_len %d does not match element_count x (sign_bits + mantissa_bits) / 8 %d",
			declaredMantissaLen, expectedMantissaBytes)
	}
	declaredExponentLen, err := metaU64(m, "exponent_plane_len")
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	storedLen, err := limits.AddChecked(declaredExponentLen, declaredMantissaLen,
		"float tensor plane lengths")
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	if storedLen != uint64(len(stored)) {
		return FloatTensorMeta{}, nil, werrors.Malformedf(
			"float tensor stored payload is %d bytes but the descriptor declares %d",
			len(stored), storedLen)
	}

	exponentLen, err := limits.ToInt(declaredExponentLen, "float tensor exponent plane length")
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	exponentCompressed := stored[:exponentLen]
	mantissaPlane := stored[exponentLen:]
	exponentBytes, err := limits.ToInt(declaredCount, "float tensor element count")
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	exponentPlane, err := HuffmanDecode(exponentCompressed, exponentBytes)
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	bytesOut, err := JoinFloatTensorPayload(
		exponentPlane, mantissaPlane, dtype, declaredCount, lim)
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}

	declaredHash, err := metaBytes(m, "payload_hash")
	if err != nil {
		return FloatTensorMeta{}, nil, err
	}
	if len(declaredHash) != hash.HashLen {
		return FloatTensorMeta{}, nil, werrors.Malformedf(
			"float tensor payload_hash must be %d bytes, got %d",
			hash.HashLen, len(declaredHash))
	}
	actual := hash.Domain(hash.DomainFloatTensorPayload, stored)
	if !bytesEqual(declaredHash, actual.Bytes()) {
		return FloatTensorMeta{}, nil, werrors.NewCorrupt(
			"float tensor payload_hash mismatch")
	}
	return FloatTensorMeta{
		Dtype:            dtype,
		Shape:            shape,
		ElementCount:     declaredCount,
		SignBits:         signBits,
		ExponentBits:     exponentBits,
		MantissaBits:     mantissaBits,
		ExponentPlaneLen: declaredExponentLen,
		MantissaPlaneLen: declaredMantissaLen,
	}, bytesOut, nil
}

// FloatTensorMetadata validates and returns a FLOAT_TENSOR descriptor.
func FloatTensorMetadata(f *frame.Frame, lim limits.Limits) (FloatTensorMeta, error) {
	meta, _, err := parseFloatTensorBody(f, lim)
	return meta, err
}

// DecodeFloatTensorFrame validates a FLOAT_TENSOR frame and returns the exact
// reconstructed little-endian element bytes.
func DecodeFloatTensorFrame(f *frame.Frame, lim limits.Limits) ([]byte, error) {
	_, bytesOut, err := parseFloatTensorBody(f, lim)
	return bytesOut, err
}
