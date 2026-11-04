package modes

// Mode 10 TENSOR_SHARDED (WEFT-AI-CODEC §11).
//
// A frame carries one shard of a quantized tensor. The descriptor names the
// whole logical shape, the shard position, the global quantization blocks, and
// the WeftTensor LogicalTensorRoot of the unsharded tensor. The payload is one
// huffman-canonical-v1 stream over:
//
//	q8_0: [shard i8 words][global block scale bits]
//	q4_0: [shard packed i4 nibbles][global block scale bits]
//
// Decoding reconstructs those planes exactly; no dequantization happens here.

import (
	"encoding/binary"
	"math"

	"lukechampine.com/blake3"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// TensorShardedSchema is the canonical descriptor schema.
const TensorShardedSchema = "weft.ai.tensor_sharded.v1"

// ShardedBlockSizes are the only block sizes implemented in v1.
var ShardedBlockSizes = []uint64{16, 32, 64, 128, 256}

// ShardedDtype is a mode-10 quantized dtype.
type ShardedDtype string

const (
	// ShardedQ8_0 is one signed 8-bit value per element.
	ShardedQ8_0 ShardedDtype = "q8_0"
	// ShardedQ4_0 is two signed 4-bit values per byte, low nibble first.
	ShardedQ4_0 ShardedDtype = "q4_0"
)

// ParseShardedDtype parses the canonical dtype name.
func ParseShardedDtype(name string) (ShardedDtype, error) {
	switch name {
	case "q8_0":
		return ShardedQ8_0, nil
	case "q4_0":
		return ShardedQ4_0, nil
	default:
		return "", werrors.Malformedf(
			"unknown TENSOR_SHARDED dtype %q (v1 allows q8_0, q4_0)", name)
	}
}

// Name returns the canonical dtype name.
func (d ShardedDtype) Name() string { return string(d) }

// QuantPlaneLen returns the number of quantized bytes for shardElements.
func (d ShardedDtype) QuantPlaneLen(shardElements uint64) (uint64, error) {
	switch d {
	case ShardedQ8_0:
		return shardElements, nil
	case ShardedQ4_0:
		if shardElements%2 != 0 {
			return 0, werrors.NewMalformed(
				"q4_0 shard element count must be even to pack nibble pairs")
		}
		return shardElements / 2, nil
	default:
		return 0, werrors.Malformedf("unknown TENSOR_SHARDED dtype %q", d)
	}
}

// ShardedTensorMeta is the descriptor of one shard.
type ShardedTensorMeta struct {
	Dtype        ShardedDtype
	Shape        []uint64
	ElementCount uint64
	ShardAxis    uint32
	ShardCount   uint32
	ShardIndex   uint32
	BlockSize    uint64
	Scales       []uint32
	ZeroPoints   []int64
	LayoutRoot   hash.Hash32
	PayloadLen   uint64
}

// ShardElements returns the number of elements in each shard.
func (m ShardedTensorMeta) ShardElements() (uint64, error) {
	if m.ShardCount == 0 {
		return 0, werrors.NewMalformed("TENSOR_SHARDED shard_count must be at least 1")
	}
	count := uint64(m.ShardCount)
	if m.ElementCount%count != 0 {
		return 0, werrors.Malformedf(
			"TENSOR_SHARDED element_count %d is not divisible by shard_count %d",
			m.ElementCount, m.ShardCount)
	}
	return m.ElementCount / count, nil
}

// BlockCount returns the number of global quantization blocks (derived, never
// on the wire).
func (m ShardedTensorMeta) BlockCount() (uint64, error) {
	if m.BlockSize == 0 {
		return 0, werrors.NewMalformed(
			"TENSOR_SHARDED block_size must be greater than zero")
	}
	if m.ElementCount%m.BlockSize != 0 {
		return 0, werrors.Malformedf(
			"TENSOR_SHARDED element_count %d is not divisible by block_size %d",
			m.ElementCount, m.BlockSize)
	}
	return m.ElementCount / m.BlockSize, nil
}

// PlanesLen returns the reconstructed plane length: quantized bytes plus one
// f32 scale per block.
func (m ShardedTensorMeta) PlanesLen() (uint64, error) {
	shardElements, err := m.ShardElements()
	if err != nil {
		return 0, err
	}
	quant, err := m.Dtype.QuantPlaneLen(shardElements)
	if err != nil {
		return 0, err
	}
	blocks, err := m.BlockCount()
	if err != nil {
		return 0, err
	}
	scales, err := limits.MulChecked(blocks, 4, "TENSOR_SHARDED scale plane length")
	if err != nil {
		return 0, err
	}
	planes, err := limits.AddChecked(quant, scales, "TENSOR_SHARDED plane length")
	if err != nil {
		return 0, err
	}
	return planes, nil
}

// ShardedTensorInput is a whole quantized tensor to be split into shards.
type ShardedTensorInput struct {
	Dtype      ShardedDtype
	Shape      []uint64
	ShardAxis  uint32
	ShardCount uint32
	BlockSize  uint64
	Scales     []uint32
	ZeroPoints []int64
	Words      []int8
}

// DecodedTensorShard is one decoded shard.
type DecodedTensorShard struct {
	Meta   ShardedTensorMeta
	Words  []int8
	Planes []byte
}

func validateShardedShape(shape []uint64, lim limits.Limits) (uint64, error) {
	if len(shape) == 0 {
		return 0, werrors.NewMalformed(
			"TENSOR_SHARDED shape rank must be at least 1")
	}
	if uint64(len(shape)) > lim.MaxTensorRank {
		return 0, werrors.LimitExceededf(
			"TENSOR_SHARDED rank %d exceeds limit %d",
			len(shape), lim.MaxTensorRank)
	}
	product := uint64(1)
	for _, dim := range shape {
		if dim == 0 {
			return 0, werrors.NewMalformed(
				"TENSOR_SHARDED shape dimensions must be greater than zero")
		}
		if product > math.MaxUint64/dim {
			return 0, werrors.NewLimitExceeded(
				"TENSOR_SHARDED shape product overflows u64")
		}
		product *= dim
	}
	if product > lim.MaxTensorElements {
		return 0, werrors.LimitExceededf(
			"TENSOR_SHARDED element count %d exceeds limit %d",
			product, lim.MaxTensorElements)
	}
	return product, nil
}

func validateShardedScaleBits(bits uint32) error {
	scale := math.Float32frombits(bits)
	if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return werrors.NewMalformed(
			"TENSOR_SHARDED scale bits must be a finite f32")
	}
	if scale <= 0 {
		return werrors.NewMalformed(
			"TENSOR_SHARDED scale must be greater than zero")
	}
	return nil
}

func shardedBlockSizeValid(blockSize uint64) bool {
	for _, allowed := range ShardedBlockSizes {
		if blockSize == allowed {
			return true
		}
	}
	return false
}

func validateShardedDescriptor(m ShardedTensorMeta, lim limits.Limits) (uint64, uint64, uint64, error) {
	elementCount, err := validateShardedShape(m.Shape, lim)
	if err != nil {
		return 0, 0, 0, err
	}
	if m.ElementCount != elementCount {
		return 0, 0, 0, werrors.Malformedf(
			"TENSOR_SHARDED element_count %d does not match shape product %d",
			m.ElementCount, elementCount)
	}
	if m.ShardCount == 0 {
		return 0, 0, 0, werrors.NewMalformed(
			"TENSOR_SHARDED shard_count must be at least 1")
	}
	if uint64(m.ShardAxis) >= uint64(len(m.Shape)) {
		return 0, 0, 0, werrors.Malformedf(
			"TENSOR_SHARDED shard_axis %d is not below rank %d",
			m.ShardAxis, len(m.Shape))
	}
	if m.ShardIndex >= m.ShardCount {
		return 0, 0, 0, werrors.Malformedf(
			"TENSOR_SHARDED shard_index %d is not below shard_count %d",
			m.ShardIndex, m.ShardCount)
	}
	axisDim := m.Shape[m.ShardAxis]
	if axisDim%uint64(m.ShardCount) != 0 {
		return 0, 0, 0, werrors.Malformedf(
			"TENSOR_SHARDED shape[%d] = %d is not divisible by shard_count %d",
			m.ShardAxis, axisDim, m.ShardCount)
	}
	shardElements, err := m.ShardElements()
	if err != nil {
		return 0, 0, 0, err
	}
	if !shardedBlockSizeValid(m.BlockSize) {
		return 0, 0, 0, werrors.Malformedf(
			"TENSOR_SHARDED block_size %d is not one of 16, 32, 64, 128, 256",
			m.BlockSize)
	}
	blockCount, err := m.BlockCount()
	if err != nil {
		return 0, 0, 0, err
	}
	if blockCount > lim.MaxStructuredItems {
		return 0, 0, 0, werrors.LimitExceededf(
			"TENSOR_SHARDED block count %d exceeds WCB item limit %d",
			blockCount, lim.MaxStructuredItems)
	}
	if uint64(len(m.Scales)) != blockCount || uint64(len(m.ZeroPoints)) != blockCount {
		return 0, 0, 0, werrors.Malformedf(
			"TENSOR_SHARDED descriptor has %d scales and %d zero points but requires %d",
			len(m.Scales), len(m.ZeroPoints), blockCount)
	}
	for _, bits := range m.Scales {
		if err := validateShardedScaleBits(bits); err != nil {
			return 0, 0, 0, err
		}
	}
	planes, err := m.PlanesLen()
	if err != nil {
		return 0, 0, 0, err
	}
	if err := limits.CheckLen(planes, lim.MaxDecodedBytes,
		"TENSOR_SHARDED reconstructed planes"); err != nil {
		return 0, 0, 0, err
	}
	return elementCount, shardElements, blockCount, nil
}

func validateShardedWords(dtype ShardedDtype, words []int8) error {
	if dtype == ShardedQ4_0 {
		for _, word := range words {
			if word < -8 || word > 7 {
				return werrors.Malformedf(
					"q4_0 quantized value %d is outside the signed 4-bit range -8..=7",
					word)
			}
		}
	}
	return nil
}

// LogicalRootSharded computes the WeftTensor LogicalTensorRoot over a whole
// quantized tensor.
func LogicalRootSharded(
	dtype ShardedDtype,
	shape []uint64,
	words []int8,
	scales []uint32,
	zeroPoints []int64,
	lim limits.Limits,
) (hash.Hash32, error) {
	elementCount, err := validateShardedShape(shape, lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	if uint64(len(words)) != elementCount {
		return hash.Hash32{}, werrors.Malformedf(
			"TENSOR_SHARDED tensor has %d words but shape requires %d",
			len(words), elementCount)
	}
	if len(scales) != len(zeroPoints) {
		return hash.Hash32{}, werrors.NewMalformed(
			"TENSOR_SHARDED scales and zero_points must have the same length")
	}
	if err := validateShardedWords(dtype, words); err != nil {
		return hash.Hash32{}, err
	}
	for _, bits := range scales {
		if err := validateShardedScaleBits(bits); err != nil {
			return hash.Hash32{}, err
		}
	}

	domain := tensor.DomainTensorValues
	hasher := blake3.New(hash.HashLen, nil)
	var prefix [4]byte
	binary.LittleEndian.PutUint32(prefix[:], uint32(len(domain)))
	_, _ = hasher.Write(prefix[:])
	_, _ = hasher.Write([]byte(domain))
	var byteBuf [1]byte
	for _, word := range words {
		byteBuf[0] = byte(word)
		_, _ = hasher.Write(byteBuf[:])
	}
	var four [4]byte
	var eight [8]byte
	for index, scale := range scales {
		binary.LittleEndian.PutUint32(four[:], scale)
		_, _ = hasher.Write(four[:])
		binary.LittleEndian.PutUint64(eight[:], uint64(zeroPoints[index]))
		_, _ = hasher.Write(eight[:])
	}
	var valuesHash hash.Hash32
	sum := hasher.Sum(valuesHash[:0])
	copy(valuesHash[:], sum)

	shapeItems := make([]wcb.Value, 0, len(shape))
	for _, dim := range shape {
		shapeItems = append(shapeItems, wcb.U64Value(dim))
	}
	record := wcb.MapValue(map[string]wcb.Value{
		"schema":        wcb.TextValue(tensor.LogicalSchema),
		"dtype":         wcb.TextValue(dtype.Name()),
		"shape":         wcb.ArrayValue(shapeItems),
		"element_count": wcb.U64Value(elementCount),
		"values_hash":   wcb.BytesValue(valuesHash.Bytes()),
	})
	canonical, err := wcb.Encode(record, lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(tensor.DomainTensorLogical, canonical), nil
}

func packQ4(words []int8) ([]byte, error) {
	if len(words)%2 != 0 {
		return nil, werrors.NewMalformed(
			"q4_0 shard element count must be even to pack nibble pairs")
	}
	out := make([]byte, len(words)/2)
	for index := range out {
		low := byte(words[2*index]) & 0x0f
		high := byte(words[2*index+1]) & 0x0f
		out[index] = low | (high << 4)
	}
	return out, nil
}

func nibbleToInt8(nibble byte) int8 {
	if nibble < 8 {
		return int8(nibble)
	}
	return int8(nibble) - 16
}

func unpackQ4(packed []byte, count int) []int8 {
	out := make([]int8, 0, count)
	for _, value := range packed {
		out = append(out, nibbleToInt8(value&0x0f))
		out = append(out, nibbleToInt8(value>>4))
	}
	if len(out) > count {
		out = out[:count]
	}
	return out
}

func quantizeShard(dtype ShardedDtype, shardBytes []byte) ([]byte, error) {
	switch dtype {
	case ShardedQ8_0:
		out := make([]byte, len(shardBytes))
		copy(out, shardBytes)
		return out, nil
	case ShardedQ4_0:
		words := make([]int8, len(shardBytes))
		for index, value := range shardBytes {
			words[index] = int8(value)
		}
		return packQ4(words)
	default:
		return nil, werrors.Malformedf("unknown TENSOR_SHARDED dtype %q", dtype)
	}
}

func shardedScalePlane(scales []uint32) []byte {
	out := make([]byte, 0, len(scales)*4)
	var buf [4]byte
	for _, scale := range scales {
		binary.LittleEndian.PutUint32(buf[:], scale)
		out = append(out, buf[:]...)
	}
	return out
}

func shardedDescriptorValue(meta ShardedTensorMeta, payloadHash hash.Hash32) wcb.Value {
	shapeItems := make([]wcb.Value, 0, len(meta.Shape))
	for _, dim := range meta.Shape {
		shapeItems = append(shapeItems, wcb.U64Value(dim))
	}
	scaleItems := make([]wcb.Value, 0, len(meta.Scales))
	for _, bits := range meta.Scales {
		scaleItems = append(scaleItems, wcb.U64Value(uint64(bits)))
	}
	zeroItems := make([]wcb.Value, 0, len(meta.ZeroPoints))
	for _, zero := range meta.ZeroPoints {
		zeroItems = append(zeroItems, wcb.I64Value(zero))
	}
	return wcb.MapValue(map[string]wcb.Value{
		"schema":        wcb.TextValue(TensorShardedSchema),
		"dtype":         wcb.TextValue(meta.Dtype.Name()),
		"shape":         wcb.ArrayValue(shapeItems),
		"element_count": wcb.U64Value(meta.ElementCount),
		"shard_axis":    wcb.U64Value(uint64(meta.ShardAxis)),
		"shard_count":   wcb.U64Value(uint64(meta.ShardCount)),
		"shard_index":   wcb.U64Value(uint64(meta.ShardIndex)),
		"block_size":    wcb.U64Value(meta.BlockSize),
		"scales":        wcb.ArrayValue(scaleItems),
		"zero_points":   wcb.ArrayValue(zeroItems),
		"layout_root":   wcb.BytesValue(meta.LayoutRoot.Bytes()),
		"payload_len":   wcb.U64Value(meta.PayloadLen),
		"payload_hash":  wcb.BytesValue(payloadHash.Bytes()),
	})
}

// EncodeShardFrameWithStored builds a frame from explicit planes and an
// already-encoded Huffman stream. It validates only the stored-length binding.
func EncodeShardFrameWithStored(
	meta ShardedTensorMeta,
	planes []byte,
	stored []byte,
	lim limits.Limits,
) ([]byte, error) {
	if meta.PayloadLen != uint64(len(stored)) {
		return nil, werrors.Malformedf(
			"TENSOR_SHARDED payload_len %d does not match the stored stream length %d",
			meta.PayloadLen, len(stored))
	}
	payloadHash := hash.Domain(hash.DomainTensorShardedPayload, stored)
	metaValue := shardedDescriptorValue(meta, payloadHash)
	body, err := buildAIBody(metaValue, stored, lim)
	if err != nil {
		return nil, err
	}
	parts := frame.PayloadParts(
		frame.ModeTensorSharded, 0, 0, uint64(len(planes)),
		hash.Domain(hash.DomainChunk, planes), body)
	return frame.Encode(parts, lim)
}

// EncodeShardFrame encodes planes with huffman-canonical-v1 and builds the
// shard frame.
func EncodeShardFrame(meta ShardedTensorMeta, planes []byte, lim limits.Limits) ([]byte, error) {
	expected, err := meta.PlanesLen()
	if err != nil {
		return nil, err
	}
	if uint64(len(planes)) != expected {
		return nil, werrors.Malformedf(
			"TENSOR_SHARDED planes are %d bytes but the descriptor requires %d",
			len(planes), expected)
	}
	stored := HuffmanEncode(planes)
	meta.PayloadLen = uint64(len(stored))
	return EncodeShardFrameWithStored(meta, planes, stored, lim)
}

// EncodeShardedTensor encodes a whole quantized tensor into shardCount mode-10
// frames, returned in shard_index order.
func EncodeShardedTensor(input ShardedTensorInput, lim limits.Limits) ([][]byte, error) {
	elementCount, err := validateShardedShape(input.Shape, lim)
	if err != nil {
		return nil, err
	}
	meta := ShardedTensorMeta{
		Dtype:        input.Dtype,
		Shape:        input.Shape,
		ElementCount: elementCount,
		ShardAxis:    input.ShardAxis,
		ShardCount:   input.ShardCount,
		BlockSize:    input.BlockSize,
		Scales:       input.Scales,
		ZeroPoints:   input.ZeroPoints,
	}
	_, _, _, err = validateShardedDescriptor(meta, lim)
	if err != nil {
		return nil, err
	}
	if uint64(len(input.Words)) != elementCount {
		return nil, werrors.Malformedf(
			"TENSOR_SHARDED tensor has %d words but shape requires %d",
			len(input.Words), elementCount)
	}
	if err := validateShardedWords(input.Dtype, input.Words); err != nil {
		return nil, err
	}
	root, err := LogicalRootSharded(
		input.Dtype, input.Shape, input.Words, input.Scales, input.ZeroPoints, lim)
	if err != nil {
		return nil, err
	}
	wordBytes := make([]byte, len(input.Words))
	for index, word := range input.Words {
		wordBytes[index] = byte(word)
	}
	shards, err := tensor.SplitShards(
		"i8", input.Shape, input.ShardAxis, input.ShardCount, wordBytes, lim)
	if err != nil {
		return nil, err
	}
	scalePlane := shardedScalePlane(input.Scales)
	frames := make([][]byte, 0, len(shards))
	for index, shard := range shards {
		quant, err := quantizeShard(input.Dtype, shard)
		if err != nil {
			return nil, err
		}
		planes := make([]byte, 0, len(quant)+len(scalePlane))
		planes = append(planes, quant...)
		planes = append(planes, scalePlane...)
		shardMeta := meta
		shardMeta.ShardIndex = uint32(index)
		shardMeta.LayoutRoot = root
		encoded, err := EncodeShardFrame(shardMeta, planes, lim)
		if err != nil {
			return nil, err
		}
		frames = append(frames, encoded)
	}
	return frames, nil
}

func metaU32(m map[string]wcb.Value, key string) (uint32, error) {
	value, err := metaU64(m, key)
	if err != nil {
		return 0, err
	}
	if value > math.MaxUint32 {
		return 0, werrors.Malformedf(
			"TENSOR_SHARDED metadata %q = %d does not fit u32", key, value)
	}
	return uint32(value), nil
}

func metaU32Array(m map[string]wcb.Value, key string) ([]uint32, error) {
	value, ok := m[key]
	if !ok || value.Kind != wcb.Array {
		return nil, werrors.Malformedf(
			"TENSOR_SHARDED metadata %q must be an array", key)
	}
	out := make([]uint32, 0, len(value.Array))
	for _, item := range value.Array {
		if item.Kind != wcb.U64 {
			return nil, werrors.Malformedf(
				"TENSOR_SHARDED %q elements must be u64", key)
		}
		if item.U > math.MaxUint32 {
			return nil, werrors.NewMalformed(
				"TENSOR_SHARDED scale bits exceed the 32-bit f32 pattern")
		}
		out = append(out, uint32(item.U))
	}
	return out, nil
}

func metaI64Array(m map[string]wcb.Value, key string) ([]int64, error) {
	value, ok := m[key]
	if !ok || value.Kind != wcb.Array {
		return nil, werrors.Malformedf(
			"TENSOR_SHARDED metadata %q must be an array", key)
	}
	out := make([]int64, 0, len(value.Array))
	for _, item := range value.Array {
		if item.Kind != wcb.I64 {
			return nil, werrors.Malformedf(
				"TENSOR_SHARDED %q elements must be i64", key)
		}
		out = append(out, item.I)
	}
	return out, nil
}

func parseShardedDescriptor(descriptorBytes []byte, lim limits.Limits) (ShardedTensorMeta, []byte, error) {
	value, err := wcb.Decode(descriptorBytes, lim)
	if err != nil {
		return ShardedTensorMeta{}, nil, err
	}
	if value.Kind != wcb.Map {
		return ShardedTensorMeta{}, nil, werrors.NewMalformed(
			"TENSOR_SHARDED metadata must be a WCB map")
	}
	m := value.Map
	if err := checkExactKeys(m, []string{
		"schema", "dtype", "shape", "element_count",
		"shard_axis", "shard_count", "shard_index", "block_size",
		"scales", "zero_points", "layout_root", "payload_len", "payload_hash",
	}, "TENSOR_SHARDED"); err != nil {
		return ShardedTensorMeta{}, nil, err
	}
	schema, err := metaText(m, "schema")
	if err != nil {
		return ShardedTensorMeta{}, nil, err
	}
	if schema != TensorShardedSchema {
		return ShardedTensorMeta{}, nil, werrors.Malformedf(
			"unsupported TENSOR_SHARDED schema %q", schema)
	}
	dtypeName, err := metaText(m, "dtype")
	if err != nil {
		return ShardedTensorMeta{}, nil, err
	}
	dtype, err := ParseShardedDtype(dtypeName)
	if err != nil {
		return ShardedTensorMeta{}, nil, err
	}
	shape, err := metaShape(m)
	if err != nil {
		return ShardedTensorMeta{}, nil, err
	}
	elementCount, err := metaU64(m, "element_count")
	if err != nil {
		return ShardedTensorMeta{}, nil, err
	}
	shardAxis, err := metaU32(m, "shard_axis")
	if err != nil {
		return ShardedTensorMeta{}, nil, err
	}
	shardCount, err := metaU32(m, "shard_count")
	if err != nil {
		return ShardedTensorMeta{}, nil, err
	}
	shardIndex, err := metaU32(m, "shard_index")
	if err != nil {
		return ShardedTensorMeta{}, nil, err
	}
	blockSize, err := metaU64(m, "block_size")
	if err != nil {
		return ShardedTensorMeta{}, nil, err
	}
	scales, err := metaU32Array(m, "scales")
	if err != nil {
		return ShardedTensorMeta{}, nil, err
	}
	zeroPoints, err := metaI64Array(m, "zero_points")
	if err != nil {
		return ShardedTensorMeta{}, nil, err
	}
	rootBytes, err := metaBytes(m, "layout_root")
	if err != nil {
		return ShardedTensorMeta{}, nil, err
	}
	if len(rootBytes) != hash.HashLen {
		return ShardedTensorMeta{}, nil, werrors.Malformedf(
			"TENSOR_SHARDED layout_root must be %d bytes, got %d",
			hash.HashLen, len(rootBytes))
	}
	layoutRoot, err := hash.FromSlice(rootBytes)
	if err != nil {
		return ShardedTensorMeta{}, nil, err
	}
	payloadLen, err := metaU64(m, "payload_len")
	if err != nil {
		return ShardedTensorMeta{}, nil, err
	}
	payloadHash, err := metaBytes(m, "payload_hash")
	if err != nil {
		return ShardedTensorMeta{}, nil, err
	}
	if len(payloadHash) != hash.HashLen {
		return ShardedTensorMeta{}, nil, werrors.Malformedf(
			"TENSOR_SHARDED payload_hash must be %d bytes, got %d",
			hash.HashLen, len(payloadHash))
	}
	return ShardedTensorMeta{
		Dtype:        dtype,
		Shape:        shape,
		ElementCount: elementCount,
		ShardAxis:    shardAxis,
		ShardCount:   shardCount,
		ShardIndex:   shardIndex,
		BlockSize:    blockSize,
		Scales:       scales,
		ZeroPoints:   zeroPoints,
		LayoutRoot:   layoutRoot,
		PayloadLen:   payloadLen,
	}, payloadHash, nil
}

// DecodeShardFrame validates and decodes one TENSOR_SHARDED frame.
func DecodeShardFrame(f *frame.Frame, lim limits.Limits) (DecodedTensorShard, error) {
	if f.Header.Mode != frame.ModeTensorSharded {
		return DecodedTensorShard{}, werrors.Malformedf(
			"expected a TENSOR_SHARDED frame, got mode %s", f.Header.Mode)
	}
	descriptorBytes, stored, err := splitAIBodyPrefix(f.Body)
	if err != nil {
		return DecodedTensorShard{}, err
	}
	meta, declaredPayloadHash, err := parseShardedDescriptor(descriptorBytes, lim)
	if err != nil {
		return DecodedTensorShard{}, err
	}
	_, shardElements, blockCount, err := validateShardedDescriptor(meta, lim)
	if err != nil {
		return DecodedTensorShard{}, err
	}
	planesLen, err := meta.PlanesLen()
	if err != nil {
		return DecodedTensorShard{}, err
	}
	if meta.PayloadLen != uint64(len(stored)) {
		return DecodedTensorShard{}, werrors.Malformedf(
			"TENSOR_SHARDED payload is %d bytes but payload_len declares %d",
			len(stored), meta.PayloadLen)
	}
	if f.Header.DecodedLen != planesLen {
		return DecodedTensorShard{}, werrors.Corruptf(
			"TENSOR_SHARDED decoded_len is %d but the descriptor requires %d",
			f.Header.DecodedLen, planesLen)
	}
	actualPayloadHash := hash.Domain(hash.DomainTensorShardedPayload, stored)
	if !bytesEqual(declaredPayloadHash, actualPayloadHash.Bytes()) {
		return DecodedTensorShard{}, werrors.NewCorrupt(
			"TENSOR_SHARDED payload_hash mismatch")
	}
	if err := limits.CheckLen(planesLen, lim.MaxDecodedBytes,
		"TENSOR_SHARDED reconstructed planes"); err != nil {
		return DecodedTensorShard{}, err
	}
	planesSize, err := limits.ToInt(planesLen, "TENSOR_SHARDED plane length")
	if err != nil {
		return DecodedTensorShard{}, err
	}
	planes, err := HuffmanDecode(stored, planesSize)
	if err != nil {
		return DecodedTensorShard{}, err
	}
	quantLen, err := meta.Dtype.QuantPlaneLen(shardElements)
	if err != nil {
		return DecodedTensorShard{}, err
	}
	quantSize, err := limits.ToInt(quantLen, "TENSOR_SHARDED quant plane length")
	if err != nil {
		return DecodedTensorShard{}, err
	}
	if len(planes) != planesSize || quantSize > len(planes) {
		return DecodedTensorShard{}, werrors.NewCorrupt(
			"TENSOR_SHARDED reconstructed plane length is inconsistent")
	}
	quantPlane := planes[:quantSize]
	scaleBytes := planes[quantSize:]
	scaleCount, err := limits.ToInt(blockCount, "TENSOR_SHARDED block count")
	if err != nil {
		return DecodedTensorShard{}, err
	}
	if len(scaleBytes) != scaleCount*4 {
		return DecodedTensorShard{}, werrors.NewCorrupt(
			"TENSOR_SHARDED scale plane length is inconsistent")
	}
	for index := 0; index < scaleCount; index++ {
		actual := binary.LittleEndian.Uint32(scaleBytes[index*4 : index*4+4])
		if actual != meta.Scales[index] {
			return DecodedTensorShard{}, werrors.Corruptf(
				"TENSOR_SHARDED scale plane block %d does not match the descriptor", index)
		}
	}
	if f.Header.ContentHash != hash.Domain(hash.DomainChunk, planes) {
		return DecodedTensorShard{}, werrors.NewCorrupt(
			"TENSOR_SHARDED content_hash mismatch against the reconstructed planes")
	}
	var words []int8
	switch meta.Dtype {
	case ShardedQ8_0:
		words = make([]int8, len(quantPlane))
		for index, value := range quantPlane {
			words[index] = int8(value)
		}
	case ShardedQ4_0:
		words = unpackQ4(quantPlane, int(shardElements))
	default:
		return DecodedTensorShard{}, werrors.Malformedf(
			"unknown TENSOR_SHARDED dtype %q", meta.Dtype)
	}
	if uint64(len(words)) != shardElements {
		return DecodedTensorShard{}, werrors.NewCorrupt(
			"TENSOR_SHARDED reconstructed word count is inconsistent")
	}
	return DecodedTensorShard{Meta: meta, Words: words, Planes: planes}, nil
}

// DecodeShardedPlanes validates a TENSOR_SHARDED frame and returns its
// reconstructed planes.
func DecodeShardedPlanes(f *frame.Frame, lim limits.Limits) ([]byte, error) {
	shard, err := DecodeShardFrame(f, lim)
	if err != nil {
		return nil, err
	}
	return shard.Planes, nil
}

func sameShardedTensor(left, right ShardedTensorMeta) bool {
	if left.Dtype != right.Dtype ||
		left.ElementCount != right.ElementCount ||
		left.ShardAxis != right.ShardAxis ||
		left.ShardCount != right.ShardCount ||
		left.BlockSize != right.BlockSize ||
		left.LayoutRoot != right.LayoutRoot ||
		len(left.Shape) != len(right.Shape) ||
		len(left.Scales) != len(right.Scales) ||
		len(left.ZeroPoints) != len(right.ZeroPoints) {
		return false
	}
	for index := range left.Shape {
		if left.Shape[index] != right.Shape[index] {
			return false
		}
	}
	for index := range left.Scales {
		if left.Scales[index] != right.Scales[index] {
			return false
		}
	}
	for index := range left.ZeroPoints {
		if left.ZeroPoints[index] != right.ZeroPoints[index] {
			return false
		}
	}
	return true
}

func orderedShards(shards []DecodedTensorShard, lim limits.Limits) ([]DecodedTensorShard, ShardedTensorMeta, error) {
	if len(shards) == 0 {
		return nil, ShardedTensorMeta{}, werrors.NewMalformed(
			"TENSOR_SHARDED shard set is empty")
	}
	first := shards[0].Meta
	if _, _, _, err := validateShardedDescriptor(first, lim); err != nil {
		return nil, ShardedTensorMeta{}, err
	}
	count := int(first.ShardCount)
	if len(shards) != count {
		return nil, ShardedTensorMeta{}, werrors.Malformedf(
			"TENSOR_SHARDED shard set has %d entries but the descriptor declares %d",
			len(shards), first.ShardCount)
	}
	ordered := make([]*DecodedTensorShard, count)
	for index := range shards {
		shard := &shards[index]
		if !sameShardedTensor(first, shard.Meta) {
			return nil, ShardedTensorMeta{}, werrors.NewNonCanonical(
				"TENSOR_SHARDED shard descriptors disagree on tensor identity")
		}
		slot := int(shard.Meta.ShardIndex)
		if slot >= count {
			return nil, ShardedTensorMeta{}, werrors.NonCanonicalf(
				"TENSOR_SHARDED shard index %d is not below shard count %d",
				shard.Meta.ShardIndex, first.ShardCount)
		}
		if ordered[slot] != nil {
			return nil, ShardedTensorMeta{}, werrors.NonCanonicalf(
				"duplicate TENSOR_SHARDED shard index %d", shard.Meta.ShardIndex)
		}
		ordered[slot] = shard
	}
	result := make([]DecodedTensorShard, 0, count)
	for index, shard := range ordered {
		if shard == nil {
			return nil, ShardedTensorMeta{}, werrors.NonCanonicalf(
				"missing TENSOR_SHARDED shard index %d", index)
		}
		result = append(result, *shard)
	}
	return result, first, nil
}

func assembleOrdered(ordered []DecodedTensorShard, meta ShardedTensorMeta, lim limits.Limits) ([]int8, error) {
	payloads := make([][]byte, 0, len(ordered))
	for _, shard := range ordered {
		payload := make([]byte, len(shard.Words))
		for index, word := range shard.Words {
			payload[index] = byte(word)
		}
		payloads = append(payloads, payload)
	}
	physical, err := tensor.NewSharded("i8", meta.Shape, meta.ShardAxis, payloads, lim)
	if err != nil {
		return nil, err
	}
	bytesOut, err := physical.AssembleWithLimits(lim)
	if err != nil {
		return nil, err
	}
	words := make([]int8, len(bytesOut))
	for index, value := range bytesOut {
		words[index] = int8(value)
	}
	return words, nil
}

// AssembleShardedWords reassembles the whole-tensor quantized words from a
// complete shard set.
func AssembleShardedWords(shards []DecodedTensorShard, lim limits.Limits) ([]int8, error) {
	ordered, meta, err := orderedShards(shards, lim)
	if err != nil {
		return nil, err
	}
	return assembleOrdered(ordered, meta, lim)
}

// ShardedSetRoot computes the logical root from a complete shard set in any
// order. Duplicate, missing, and inconsistent shards fail closed.
func ShardedSetRoot(shards []DecodedTensorShard, lim limits.Limits) (hash.Hash32, error) {
	ordered, meta, err := orderedShards(shards, lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	words, err := assembleOrdered(ordered, meta, lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	root, err := LogicalRootSharded(
		meta.Dtype, meta.Shape, words, meta.Scales, meta.ZeroPoints, lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	if root != meta.LayoutRoot {
		return hash.Hash32{}, werrors.Corruptf(
			"TENSOR_SHARDED shard set recomputes logical root %s, but the descriptor declares %s",
			root, meta.LayoutRoot)
	}
	return root, nil
}
