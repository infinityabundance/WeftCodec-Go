// Package modes implements mode 11 WEIGHT_DELTA (WEFT-AI-CODEC §12).
package modes

import (
	"context"
	"encoding/binary"
	"slices"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// WeightDeltaSchema is the canonical descriptor schema.
const WeightDeltaSchema = "weft.ai.weight_delta.v1"

// WeightDeltaOp is the only delta operation implemented in v1.
const WeightDeltaOp = "mod-sub"

// WeightDeltaPlaneCodec is the only plane coder implemented in v1.
const WeightDeltaPlaneCodec = HuffmanCoderID

// WeightDtype describes one element word layout.
type WeightDtype struct {
	Name         string
	WordBits     uint64
	StorageBytes uint64
	Planes       uint64
}

// ParseWeightDtype parses a canonical WEIGHT_DELTA dtype name.
func ParseWeightDtype(name string) (WeightDtype, error) {
	switch name {
	case "bf16":
		return WeightDtype{Name: name, WordBits: 16, StorageBytes: 2, Planes: 2}, nil
	case "fp16":
		return WeightDtype{Name: name, WordBits: 16, StorageBytes: 2, Planes: 2}, nil
	case "fp32":
		return WeightDtype{Name: name, WordBits: 32, StorageBytes: 4, Planes: 4}, nil
	case "q8_0":
		return WeightDtype{Name: name, WordBits: 8, StorageBytes: 1, Planes: 1}, nil
	case "q4_0":
		return WeightDtype{Name: name, WordBits: 4, StorageBytes: 1, Planes: 1}, nil
	default:
		return WeightDtype{}, werrors.Malformedf(
			"unknown WEIGHT_DELTA dtype %q (v1 allows bf16, fp16, fp32, q8_0, q4_0)", name)
	}
}

// IsQuantized reports whether the dtype's words are quantized integers.
func (d WeightDtype) IsQuantized() bool {
	return d.Name == "q8_0" || d.Name == "q4_0"
}

// WeightDeltaMeta is a parsed mode-11 descriptor.
type WeightDeltaMeta struct {
	Dtype        WeightDtype
	Shape        []uint64
	ElementCount uint64
	WordBits     uint64
	Base         hash.Hash32
	BaseDtype    WeightDtype
	BaseShape    []uint64
	Op           string
	Planes       uint64
	PlaneCodec   string
	DeltaRoot    hash.Hash32
	PayloadLen   uint64
}

func weightDeltaMetaText(m map[string]wcb.Value, key string) (string, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.Text {
		return "", werrors.Malformedf("WEIGHT_DELTA metadata %q must be text", key)
	}
	return v.Str, nil
}

func weightDeltaMetaU64(m map[string]wcb.Value, key string) (uint64, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.U64 {
		return 0, werrors.Malformedf("WEIGHT_DELTA metadata %q must be u64", key)
	}
	return v.U, nil
}

func weightDeltaMetaBytes(m map[string]wcb.Value, key string) ([]byte, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.Bytes {
		return nil, werrors.Malformedf("WEIGHT_DELTA metadata %q must be bytes", key)
	}
	return v.Bytes, nil
}

func weightDeltaMetaHash(m map[string]wcb.Value, key string) (hash.Hash32, error) {
	bytes, err := weightDeltaMetaBytes(m, key)
	if err != nil {
		return hash.Hash32{}, err
	}
	if len(bytes) != hash.HashLen {
		return hash.Hash32{}, werrors.Malformedf(
			"WEIGHT_DELTA metadata %q must be %d bytes, got %d", key, hash.HashLen, len(bytes))
	}
	return hash.FromSlice(bytes)
}

func weightDeltaMetaShape(m map[string]wcb.Value, key string) ([]uint64, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.Array {
		return nil, werrors.Malformedf("WEIGHT_DELTA metadata %q must be an array", key)
	}
	shape := make([]uint64, 0, len(v.Array))
	for _, item := range v.Array {
		if item.Kind != wcb.U64 {
			return nil, werrors.Malformedf(
				"WEIGHT_DELTA metadata %q elements must be u64", key)
		}
		shape = append(shape, item.U)
	}
	return shape, nil
}

func weightDeltaShape(shape []uint64, lim limits.Limits) (uint64, error) {
	if len(shape) == 0 {
		return 0, werrors.NewMalformed("WEIGHT_DELTA shape rank must be at least 1")
	}
	if uint64(len(shape)) > lim.MaxTensorRank {
		return 0, werrors.LimitExceededf(
			"WEIGHT_DELTA rank %d exceeds limit %d", len(shape), lim.MaxTensorRank)
	}
	product := uint64(1)
	for _, dim := range shape {
		if dim == 0 {
			return 0, werrors.NewMalformed(
				"WEIGHT_DELTA shape dimensions must be greater than zero")
		}
		next, err := limits.MulChecked(product, dim, "WEIGHT_DELTA shape product")
		if err != nil {
			return 0, err
		}
		product = next
	}
	if product > lim.MaxTensorElements {
		return 0, werrors.LimitExceededf(
			"WEIGHT_DELTA element count %d exceeds limit %d", product, lim.MaxTensorElements)
	}
	return product, nil
}

func weightDeltaPayloadLen(elementCount uint64, dtype WeightDtype) (uint64, error) {
	return limits.MulChecked(elementCount, dtype.StorageBytes, "WEIGHT_DELTA element bytes")
}

// ValidateWeightDeltaMeta validates every descriptor invariant and returns the
// element count.
func ValidateWeightDeltaMeta(meta WeightDeltaMeta, lim limits.Limits) (uint64, error) {
	if meta.Op != WeightDeltaOp {
		return 0, werrors.Malformedf(
			"WEIGHT_DELTA op %q is not implemented (only %s)", meta.Op, WeightDeltaOp)
	}
	if meta.PlaneCodec != WeightDeltaPlaneCodec {
		return 0, werrors.Malformedf(
			"WEIGHT_DELTA plane_codec %q is not implemented (only %s)",
			meta.PlaneCodec, WeightDeltaPlaneCodec)
	}
	if meta.Dtype.WordBits != meta.WordBits {
		return 0, werrors.Malformedf(
			"WEIGHT_DELTA word_bits %d disagrees with dtype %s (%d bits)",
			meta.WordBits, meta.Dtype.Name, meta.Dtype.WordBits)
	}
	if meta.Dtype.Planes != meta.Planes {
		return 0, werrors.Malformedf(
			"WEIGHT_DELTA planes %d disagrees with dtype %s (%d planes)",
			meta.Planes, meta.Dtype.Name, meta.Dtype.Planes)
	}
	if meta.BaseDtype.Name != meta.Dtype.Name {
		return 0, werrors.Malformedf(
			"WEIGHT_DELTA base_dtype %q must equal dtype %q",
			meta.BaseDtype.Name, meta.Dtype.Name)
	}
	if !slices.Equal(meta.BaseShape, meta.Shape) {
		return 0, werrors.NewMalformed("WEIGHT_DELTA base_shape must equal shape")
	}
	count, err := weightDeltaShape(meta.Shape, lim)
	if err != nil {
		return 0, err
	}
	if meta.ElementCount != count {
		return 0, werrors.Malformedf(
			"WEIGHT_DELTA element_count %d does not match shape product %d",
			meta.ElementCount, count)
	}
	if _, err := weightDeltaPayloadLen(count, meta.Dtype); err != nil {
		return 0, err
	}
	return count, nil
}

func weightDeltaDescriptorValue(meta WeightDeltaMeta) wcb.Value {
	shape := make([]wcb.Value, 0, len(meta.Shape))
	for _, dim := range meta.Shape {
		shape = append(shape, wcb.U64Value(dim))
	}
	baseShape := make([]wcb.Value, 0, len(meta.BaseShape))
	for _, dim := range meta.BaseShape {
		baseShape = append(baseShape, wcb.U64Value(dim))
	}
	return wcb.MapValue(map[string]wcb.Value{
		"schema":        wcb.TextValue(WeightDeltaSchema),
		"dtype":         wcb.TextValue(meta.Dtype.Name),
		"shape":         wcb.ArrayValue(shape),
		"element_count": wcb.U64Value(meta.ElementCount),
		"word_bits":     wcb.U64Value(meta.WordBits),
		"base":          wcb.BytesValue(meta.Base.Bytes()),
		"base_dtype":    wcb.TextValue(meta.BaseDtype.Name),
		"base_shape":    wcb.ArrayValue(baseShape),
		"op":            wcb.TextValue(meta.Op),
		"planes":        wcb.U64Value(meta.Planes),
		"plane_codec":   wcb.TextValue(meta.PlaneCodec),
		"delta_root":    wcb.BytesValue(meta.DeltaRoot.Bytes()),
		"payload_len":   wcb.U64Value(meta.PayloadLen),
	})
}

func parseWeightDeltaDescriptor(bytes []byte, lim limits.Limits) (WeightDeltaMeta, error) {
	value, err := wcb.Decode(bytes, lim)
	if err != nil {
		return WeightDeltaMeta{}, err
	}
	if value.Kind != wcb.Map {
		return WeightDeltaMeta{}, werrors.NewMalformed(
			"WEIGHT_DELTA metadata must be a WCB map")
	}
	m := value.Map
	keys := []string{
		"schema", "dtype", "shape", "element_count", "word_bits",
		"base", "base_dtype", "base_shape", "op", "planes",
		"plane_codec", "delta_root", "payload_len",
	}
	for key := range m {
		if !containsString(keys, key) {
			return WeightDeltaMeta{}, werrors.Malformedf(
				"WEIGHT_DELTA metadata has unknown key %q", key)
		}
	}
	for _, key := range keys {
		if _, ok := m[key]; !ok {
			return WeightDeltaMeta{}, werrors.Malformedf(
				"WEIGHT_DELTA metadata is missing key %q", key)
		}
	}
	schema, err := weightDeltaMetaText(m, "schema")
	if err != nil {
		return WeightDeltaMeta{}, err
	}
	if schema != WeightDeltaSchema {
		return WeightDeltaMeta{}, werrors.Malformedf(
			"unsupported WEIGHT_DELTA schema %q", schema)
	}
	dtypeText, err := weightDeltaMetaText(m, "dtype")
	if err != nil {
		return WeightDeltaMeta{}, err
	}
	dtype, err := ParseWeightDtype(dtypeText)
	if err != nil {
		return WeightDeltaMeta{}, err
	}
	shape, err := weightDeltaMetaShape(m, "shape")
	if err != nil {
		return WeightDeltaMeta{}, err
	}
	elementCount, err := weightDeltaMetaU64(m, "element_count")
	if err != nil {
		return WeightDeltaMeta{}, err
	}
	wordBits, err := weightDeltaMetaU64(m, "word_bits")
	if err != nil {
		return WeightDeltaMeta{}, err
	}
	base, err := weightDeltaMetaHash(m, "base")
	if err != nil {
		return WeightDeltaMeta{}, err
	}
	baseDtypeText, err := weightDeltaMetaText(m, "base_dtype")
	if err != nil {
		return WeightDeltaMeta{}, err
	}
	baseDtype, err := ParseWeightDtype(baseDtypeText)
	if err != nil {
		return WeightDeltaMeta{}, err
	}
	baseShape, err := weightDeltaMetaShape(m, "base_shape")
	if err != nil {
		return WeightDeltaMeta{}, err
	}
	op, err := weightDeltaMetaText(m, "op")
	if err != nil {
		return WeightDeltaMeta{}, err
	}
	planes, err := weightDeltaMetaU64(m, "planes")
	if err != nil {
		return WeightDeltaMeta{}, err
	}
	planeCodec, err := weightDeltaMetaText(m, "plane_codec")
	if err != nil {
		return WeightDeltaMeta{}, err
	}
	deltaRoot, err := weightDeltaMetaHash(m, "delta_root")
	if err != nil {
		return WeightDeltaMeta{}, err
	}
	payloadLen, err := weightDeltaMetaU64(m, "payload_len")
	if err != nil {
		return WeightDeltaMeta{}, err
	}
	meta := WeightDeltaMeta{
		Dtype:        dtype,
		Shape:        shape,
		ElementCount: elementCount,
		WordBits:     wordBits,
		Base:         base,
		BaseDtype:    baseDtype,
		BaseShape:    baseShape,
		Op:           op,
		Planes:       planes,
		PlaneCodec:   planeCodec,
		DeltaRoot:    deltaRoot,
		PayloadLen:   payloadLen,
	}
	if _, err := ValidateWeightDeltaMeta(meta, lim); err != nil {
		return WeightDeltaMeta{}, err
	}
	return meta, nil
}

// WeightDeltaMetadata parses and validates a mode-11 descriptor without
// resolving the base.
func WeightDeltaMetadata(f *frame.Frame, lim limits.Limits) (WeightDeltaMeta, error) {
	if f.Header.Mode != frame.ModeWeightDelta {
		return WeightDeltaMeta{}, werrors.Malformedf(
			"expected a WEIGHT_DELTA frame, got mode %s", f.Header.Mode)
	}
	descriptor, _, err := splitAIBodyPrefix(f.Body)
	if err != nil {
		return WeightDeltaMeta{}, err
	}
	return parseWeightDeltaDescriptor(descriptor, lim)
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Delta words and planes
// ---------------------------------------------------------------------------

func weightWordAt(dtype WeightDtype, bytes []byte, index int) uint32 {
	switch dtype.Name {
	case "bf16", "fp16":
		offset := index * 2
		return uint32(binary.LittleEndian.Uint16(bytes[offset : offset+2]))
	case "fp32":
		offset := index * 4
		return binary.LittleEndian.Uint32(bytes[offset : offset+4])
	default:
		return uint32(bytes[index])
	}
}

func pushWeightWord(dtype WeightDtype, word uint32, out []byte) []byte {
	switch dtype.Name {
	case "bf16", "fp16":
		var buffer [2]byte
		binary.LittleEndian.PutUint16(buffer[:], uint16(word))
		return append(out, buffer[:]...)
	case "fp32":
		var buffer [4]byte
		binary.LittleEndian.PutUint32(buffer[:], word)
		return append(out, buffer[:]...)
	default:
		return append(out, byte(word))
	}
}

func weightMask(dtype WeightDtype) uint32 {
	switch dtype.WordBits {
	case 32:
		return ^uint32(0)
	case 16:
		return 0xFFFF
	case 8:
		return 0xFF
	default:
		return 0x0F
	}
}

func buildWeightPlanes(dtype WeightDtype, target, base []byte, elementCount int) ([]byte, [][]byte) {
	planes := make([][]byte, dtype.Planes)
	for index := range planes {
		planes[index] = make([]byte, 0, elementCount)
	}
	deltaBytes := make([]byte, 0, elementCount*int(dtype.StorageBytes))
	mask := weightMask(dtype)
	for index := 0; index < elementCount; index++ {
		delta := (weightWordAt(dtype, target, index) - weightWordAt(dtype, base, index)) & mask
		deltaBytes = pushWeightWord(dtype, delta, deltaBytes)
		switch dtype.Name {
		case "bf16", "fp16":
			planes[0] = append(planes[0], byte(delta))
			planes[1] = append(planes[1], byte(delta>>8))
		case "fp32":
			planes[0] = append(planes[0], byte(delta))
			planes[1] = append(planes[1], byte(delta>>8))
			planes[2] = append(planes[2], byte(delta>>16))
			planes[3] = append(planes[3], byte(delta>>24))
		default:
			planes[0] = append(planes[0], byte(delta))
		}
	}
	return deltaBytes, planes
}

// ---------------------------------------------------------------------------
// Base frame descriptor
// ---------------------------------------------------------------------------

func tensorStorageToWeight(dtype string) (WeightDtype, error) {
	switch dtype {
	case "bf16":
		return ParseWeightDtype("bf16")
	case "f16":
		return ParseWeightDtype("fp16")
	case "f32":
		return ParseWeightDtype("fp32")
	case "i8":
		return ParseWeightDtype("q8_0")
	case "u8":
		return ParseWeightDtype("q4_0")
	default:
		return WeightDtype{}, werrors.Malformedf(
			"base frame dtype %q cannot back a WEIGHT_DELTA word plane", dtype)
	}
}

func floatStorageToWeight(dtype string) (WeightDtype, error) {
	switch dtype {
	case "bf16", "fp16", "fp32":
		return ParseWeightDtype(dtype)
	default:
		return WeightDtype{}, werrors.Malformedf(
			"base frame dtype %q cannot back a WEIGHT_DELTA word plane", dtype)
	}
}

// BaseFrameDescriptor returns the logical weight dtype and shape carried by a
// base frame. q8_0 is backed by a mode-6 i8 tensor frame and q4_0 by a mode-6
// u8 frame (one unpacked nibble per byte).
func BaseFrameDescriptor(f *frame.Frame, lim limits.Limits) (WeightDtype, []uint64, error) {
	switch f.Header.Mode {
	case frame.ModeTensor:
		meta, err := TensorMetadata(f, lim)
		if err != nil {
			return WeightDtype{}, nil, err
		}
		dtype, err := tensorStorageToWeight(meta.Dtype)
		if err != nil {
			return WeightDtype{}, nil, err
		}
		return dtype, meta.Shape, nil
	case frame.ModeFloatTensor:
		meta, err := FloatTensorMetadata(f, lim)
		if err != nil {
			return WeightDtype{}, nil, err
		}
		dtype, err := floatStorageToWeight(meta.Dtype.Name)
		if err != nil {
			return WeightDtype{}, nil, err
		}
		return dtype, meta.Shape, nil
	case frame.ModeWeightDelta:
		meta, err := WeightDeltaMetadata(f, lim)
		if err != nil {
			return WeightDtype{}, nil, err
		}
		return meta.Dtype, meta.Shape, nil
	case frame.ModeSemanticDelta:
		meta, err := SemanticDeltaMetadata(f, lim)
		if err != nil {
			return WeightDtype{}, nil, err
		}
		return meta.Dtype, meta.Shape, nil
	case frame.ModeSparseDelta:
		meta, err := SparseDeltaMetadata(f, lim)
		if err != nil {
			return WeightDtype{}, nil, err
		}
		return meta.Dtype, meta.Shape, nil
	default:
		return WeightDtype{}, nil, werrors.Malformedf(
			"base frame mode %s does not carry tensor dtype/shape metadata", f.Header.Mode)
	}
}

func checkQuantizedWords(dtype WeightDtype, bytes []byte, what string) error {
	if dtype.Name != "q4_0" {
		return nil
	}
	for index, value := range bytes {
		if value > 15 {
			return werrors.Corruptf(
				"WEIGHT_DELTA %s q4_0 word %d is %d, outside 0..=15", what, index, value)
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

// EncodeWeightDeltaDescriptorFrame builds a frame from an explicit descriptor
// and stored payload. It is the low-level constructor used by the encoder and
// by adversarial vector generation.
func EncodeWeightDeltaDescriptorFrame(
	meta WeightDeltaMeta,
	payload []byte,
	decodedLen uint64,
	contentHash hash.Hash32,
	baseMode uint32,
	lim limits.Limits,
) ([]byte, error) {
	body, err := buildAIBody(weightDeltaDescriptorValue(meta), payload, lim)
	if err != nil {
		return nil, err
	}
	parts := frame.Parts{
		FormatMajor: frame.FormatMajor,
		FormatMinor: frame.FormatMinor,
		Domain:      frame.DomainPayload,
		Mode:        frame.ModeWeightDelta,
		Flags:       0,
		Level:       0,
		DecodedLen:  decodedLen,
		ContentHash: contentHash,
		Dependencies: []frame.Dependency{
			frame.NewDependency(meta.Base, decodedLen, frame.DepDomainPayload, baseMode),
		},
		Chunks: nil,
		Body:   body,
	}
	return frame.Encode(parts, lim)
}

// EncodeWeightDelta encodes target as a mode-11 delta against baseFrame.
func EncodeWeightDelta(
	ctx context.Context,
	target []byte,
	dtype WeightDtype,
	shape []uint64,
	baseFrame []byte,
	lim limits.Limits,
) ([]byte, error) {
	return EncodeWeightDeltaWithResolver(
		ctx, target, dtype, shape, baseFrame, dependency.NewMapResolver(), lim)
}

// EncodeWeightDeltaWithResolver encodes a target against a base frame whose own
// dependencies are resolved through resolver.
func EncodeWeightDeltaWithResolver(
	ctx context.Context,
	target []byte,
	dtype WeightDtype,
	shape []uint64,
	baseFrame []byte,
	resolver dependency.Resolver,
	lim limits.Limits,
) ([]byte, error) {
	baseParsed, err := frame.Parse(baseFrame, lim)
	if err != nil {
		return nil, err
	}
	baseDtype, baseShape, err := BaseFrameDescriptor(baseParsed, lim)
	if err != nil {
		return nil, err
	}
	if baseDtype.Name != dtype.Name {
		return nil, werrors.Malformedf(
			"base frame dtype %q does not match WEIGHT_DELTA dtype %q",
			baseDtype.Name, dtype.Name)
	}
	if !slices.Equal(baseShape, shape) {
		return nil, werrors.NewMalformed(
			"base frame shape does not match the WEIGHT_DELTA shape")
	}
	elementCount, err := weightDeltaShape(shape, lim)
	if err != nil {
		return nil, err
	}
	expected, err := weightDeltaPayloadLen(elementCount, dtype)
	if err != nil {
		return nil, err
	}
	if err := limits.CheckLen(expected, lim.MaxDecodedBytes, "WEIGHT_DELTA element bytes"); err != nil {
		return nil, err
	}
	if uint64(len(target)) != expected {
		return nil, werrors.Malformedf(
			"WEIGHT_DELTA target is %d bytes but shape %s x %d requires %d",
			len(target), dtype.Name, elementCount, expected)
	}
	baseBytes, err := DecodeBytes(ctx, baseFrame, resolver, lim)
	if err != nil {
		return nil, err
	}
	if uint64(len(baseBytes)) != expected {
		return nil, werrors.DependencyMismatchf(
			"base frame decodes to %d bytes but the descriptor requires %d", len(baseBytes), expected)
	}
	if dtype.IsQuantized() {
		if err := checkQuantizedWords(dtype, target, "target"); err != nil {
			return nil, err
		}
		if err := checkQuantizedWords(dtype, baseBytes, "base"); err != nil {
			return nil, err
		}
	}
	count, err := limits.ToInt(elementCount, "WEIGHT_DELTA element count")
	if err != nil {
		return nil, err
	}
	deltaBytes, planes := buildWeightPlanes(dtype, target, baseBytes, count)
	deltaRoot := hash.Domain(tensor.DomainTensorValues, deltaBytes)
	var payload []byte
	for _, plane := range planes {
		payload = append(payload, HuffmanEncode(plane)...)
	}
	meta := WeightDeltaMeta{
		Dtype:        dtype,
		Shape:        append([]uint64(nil), shape...),
		ElementCount: elementCount,
		WordBits:     dtype.WordBits,
		Base:         baseParsed.Header.ContentHash,
		BaseDtype:    dtype,
		BaseShape:    append([]uint64(nil), shape...),
		Op:           WeightDeltaOp,
		Planes:       dtype.Planes,
		PlaneCodec:   WeightDeltaPlaneCodec,
		DeltaRoot:    deltaRoot,
		PayloadLen:   uint64(len(payload)),
	}
	return EncodeWeightDeltaDescriptorFrame(
		meta, payload, uint64(len(target)), hash.Domain(hash.DomainChunk, target),
		uint32(baseParsed.Header.Mode), lim)
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

type weightFrameProvider interface {
	ResolveFrame(ctx context.Context, contentHash hash.Hash32) ([]byte, bool, error)
}

func decodeWeightPlanes(dtype WeightDtype, payload []byte, elementCount int) ([][]byte, error) {
	planes := make([][]byte, 0, dtype.Planes)
	offset := 0
	for planeIndex := uint64(0); planeIndex < dtype.Planes; planeIndex++ {
		if offset >= len(payload) {
			return nil, werrors.NewCorrupt(
				"WEIGHT_DELTA payload ends before all declared planes")
		}
		plane, consumed, err := HuffmanDecodePrefix(payload[offset:], elementCount)
		if err != nil {
			return nil, err
		}
		offset += consumed
		planes = append(planes, plane)
	}
	if offset != len(payload) {
		return nil, werrors.NewCorrupt(
			"WEIGHT_DELTA payload has trailing bytes after the declared planes")
	}
	return planes, nil
}

func weightDeltaFromPlanes(dtype WeightDtype, planes [][]byte, index int) (uint32, error) {
	switch dtype.Name {
	case "bf16", "fp16":
		return uint32(planes[0][index]) | (uint32(planes[1][index]) << 8), nil
	case "fp32":
		return uint32(planes[0][index]) |
			(uint32(planes[1][index]) << 8) |
			(uint32(planes[2][index]) << 16) |
			(uint32(planes[3][index]) << 24), nil
	case "q8_0":
		return uint32(planes[0][index]), nil
	default:
		nibble := planes[0][index]
		if nibble > 15 {
			return 0, werrors.Corruptf(
				"WEIGHT_DELTA q4_0 delta word %d is %d, outside 0..=15", index, nibble)
		}
		return uint32(nibble), nil
	}
}

// DecodeWeightDeltaFrame decodes a mode-11 frame, resolving the base through
// resolver. A resolver that can also return the base frame bytes (the DAG
// resolver does) additionally gets its dtype/shape descriptor verified.
func DecodeWeightDeltaFrame(
	ctx context.Context,
	f *frame.Frame,
	resolver dependency.Resolver,
	lim limits.Limits,
) ([]byte, error) {
	return decodeWeightDelta(ctx, f, nil, resolver, lim)
}

// DecodeWeightDeltaFrameWithBase decodes a mode-11 frame with the base frame
// supplied out of band.
func DecodeWeightDeltaFrameWithBase(
	ctx context.Context,
	f *frame.Frame,
	baseFrame []byte,
	resolver dependency.Resolver,
	lim limits.Limits,
) ([]byte, error) {
	return decodeWeightDelta(ctx, f, baseFrame, resolver, lim)
}

func decodeWeightDelta(
	ctx context.Context,
	f *frame.Frame,
	baseFrame []byte,
	resolver dependency.Resolver,
	lim limits.Limits,
) ([]byte, error) {
	if f.Header.Mode != frame.ModeWeightDelta {
		return nil, werrors.Malformedf(
			"expected a WEIGHT_DELTA frame, got mode %s", f.Header.Mode)
	}
	descriptor, payload, err := splitAIBodyPrefix(f.Body)
	if err != nil {
		return nil, err
	}
	meta, err := parseWeightDeltaDescriptor(descriptor, lim)
	if err != nil {
		return nil, err
	}
	elementCount, err := ValidateWeightDeltaMeta(meta, lim)
	if err != nil {
		return nil, err
	}
	expected, err := weightDeltaPayloadLen(elementCount, meta.Dtype)
	if err != nil {
		return nil, err
	}
	if f.Header.DecodedLen != expected {
		return nil, werrors.Corruptf(
			"WEIGHT_DELTA decoded_len is %d but the descriptor requires %d",
			f.Header.DecodedLen, expected)
	}
	if err := limits.CheckLen(expected, lim.MaxDecodedBytes, "WEIGHT_DELTA element bytes"); err != nil {
		return nil, err
	}
	if meta.PayloadLen != uint64(len(payload)) {
		return nil, werrors.Malformedf(
			"WEIGHT_DELTA payload is %d bytes but payload_len declares %d",
			len(payload), meta.PayloadLen)
	}
	if f.Header.DepCount != 1 {
		return nil, werrors.NewMalformed(
			"WEIGHT_DELTA requires exactly one dependency")
	}
	dep := f.Dependencies[0]
	if dep.Hash != meta.Base {
		return nil, werrors.NewMalformed(
			"WEIGHT_DELTA dependency hash must equal the descriptor base")
	}
	if dep.DecodedLen != f.Header.DecodedLen {
		return nil, werrors.NewMalformed(
			"WEIGHT_DELTA dependency decoded_len must equal the frame decoded_len")
	}
	count, err := limits.ToInt(elementCount, "WEIGHT_DELTA element count")
	if err != nil {
		return nil, err
	}
	planes, err := decodeWeightPlanes(meta.Dtype, payload, count)
	if err != nil {
		return nil, err
	}
	deltaBytes := make([]byte, 0, count*int(meta.Dtype.StorageBytes))
	for index := 0; index < count; index++ {
		delta, err := weightDeltaFromPlanes(meta.Dtype, planes, index)
		if err != nil {
			return nil, err
		}
		deltaBytes = pushWeightWord(meta.Dtype, delta, deltaBytes)
	}
	if hash.Domain(tensor.DomainTensorValues, deltaBytes) != meta.DeltaRoot {
		return nil, werrors.NewCorrupt("WEIGHT_DELTA delta_root mismatch")
	}

	var baseBytes []byte
	var suppliedFrame *frame.Frame
	if baseFrame != nil {
		baseBytes, err = DecodeBytes(ctx, baseFrame, resolver, lim)
		if err != nil {
			return nil, err
		}
		suppliedFrame, err = frame.Parse(baseFrame, lim)
		if err != nil {
			return nil, err
		}
	} else {
		var fetched []byte
		var ok bool
		if provider, providerOK := resolver.(weightFrameProvider); providerOK {
			fetched, ok, err = provider.ResolveFrame(ctx, meta.Base)
			if err != nil {
				return nil, err
			}
			if !ok {
				fetched = nil
			}
		}
		baseBytes, err = resolver.Resolve(ctx, meta.Base, f.Header.DecodedLen, frame.DepDomainPayload)
		if err != nil {
			return nil, err
		}
		if fetched != nil {
			suppliedFrame, err = frame.Parse(fetched, lim)
			if err != nil {
				return nil, err
			}
		}
	}
	if suppliedFrame != nil {
		if suppliedFrame.Header.ContentHash != meta.Base {
			return nil, werrors.NewDependencyMismatch(
				"supplied base frame content hash does not equal the descriptor base")
		}
		baseDtype, baseShape, err := BaseFrameDescriptor(suppliedFrame, lim)
		if err != nil {
			return nil, werrors.DependencyMismatchf("%s", err.Error())
		}
		if baseDtype.Name != meta.Dtype.Name {
			return nil, werrors.DependencyMismatchf(
				"base frame dtype %q does not match WEIGHT_DELTA dtype %q",
				baseDtype.Name, meta.Dtype.Name)
		}
		if !slices.Equal(baseShape, meta.Shape) {
			return nil, werrors.NewDependencyMismatch(
				"base frame shape does not match the WEIGHT_DELTA shape")
		}
	}
	if uint64(len(baseBytes)) != expected {
		return nil, werrors.DependencyMismatchf(
			"base frame decodes to %d bytes but the descriptor requires %d", len(baseBytes), expected)
	}
	if meta.Dtype.IsQuantized() {
		if err := checkQuantizedWords(meta.Dtype, baseBytes, "base"); err != nil {
			return nil, err
		}
	}
	mask := weightMask(meta.Dtype)
	target := make([]byte, 0, count*int(meta.Dtype.StorageBytes))
	for index := 0; index < count; index++ {
		delta, err := weightDeltaFromPlanes(meta.Dtype, planes, index)
		if err != nil {
			return nil, err
		}
		word := (delta + weightWordAt(meta.Dtype, baseBytes, index)) & mask
		target = pushWeightWord(meta.Dtype, word, target)
	}
	if uint64(len(target)) != expected {
		return nil, werrors.NewCorrupt(
			"WEIGHT_DELTA reconstructed element length is inconsistent")
	}
	if hash.Domain(hash.DomainChunk, target) != f.Header.ContentHash {
		return nil, werrors.NewCorrupt(
			"WEIGHT_DELTA content_hash mismatch against the reconstructed tensor")
	}
	return target, nil
}
