// Package modes implements mode 13 ACTIVATION_STREAM (WEFT-AI-CODEC §14).
package modes

import (
	"math"
	"slices"
	"sort"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// ActivationStreamSchema is the canonical descriptor schema.
const ActivationStreamSchema = "weft.ai.activation_stream.v1"

// ActivationStreamPlaneCodec is the only plane coder implemented in v1.
const ActivationStreamPlaneCodec = HuffmanCoderID

// DomainAIStream is the stream-root hash domain.
const DomainAIStream = "weft/ai/stream/v1"

// StreamKind is activation or gradient.
type StreamKind struct {
	Name string
}

// ActivationStreamKind returns the activation stream kind.
var ActivationStreamKind = StreamKind{Name: "activation"}

// GradientStreamKind returns the gradient stream kind.
var GradientStreamKind = StreamKind{Name: "gradient"}

// ParseStreamKind parses a canonical stream name.
func ParseStreamKind(name string) (StreamKind, error) {
	switch name {
	case "activation":
		return ActivationStreamKind, nil
	case "gradient":
		return GradientStreamKind, nil
	default:
		return StreamKind{}, werrors.Malformedf(
			"unknown ACTIVATION_STREAM stream %q (v1 allows activation, gradient)", name)
	}
}

// StreamDtype describes one activation element dtype.
type StreamDtype struct {
	Name      string
	WordBytes uint64
	Planes    uint64
}

// ParseStreamDtype parses a canonical ACTIVATION_STREAM dtype name.
func ParseStreamDtype(name string) (StreamDtype, error) {
	switch name {
	case "bf16":
		return StreamDtype{Name: name, WordBytes: 2, Planes: 2}, nil
	case "fp16":
		return StreamDtype{Name: name, WordBytes: 2, Planes: 2}, nil
	case "fp32":
		return StreamDtype{Name: name, WordBytes: 4, Planes: 4}, nil
	default:
		return StreamDtype{}, werrors.Malformedf(
			"unknown ACTIVATION_STREAM dtype %q (v1 allows bf16, fp16, fp32)", name)
	}
}

// ActivationStreamMeta is a parsed mode-13 descriptor.
type ActivationStreamMeta struct {
	Stream       StreamKind
	Step         uint64
	Rank         uint32
	World        uint32
	ChunkIndex   uint64
	ChunkCount   uint64
	Dtype        StreamDtype
	Shape        []uint64
	ElementCount uint64
	DigestRoot   hash.Hash32
	PlaneCodec   string
	PayloadLen   uint64
}

// ActivationStreamHead is the stream-wide head shared by all chunks.
type ActivationStreamHead struct {
	Stream     StreamKind
	Step       uint64
	Rank       uint32
	World      uint32
	ChunkCount uint64
	Dtype      StreamDtype
	Shape      []uint64
}

// DecodedActivationChunk is one decoded chunk.
type DecodedActivationChunk struct {
	Meta  ActivationStreamMeta
	Bytes []byte
}

// ActivationStreamAssembly is the assembled stream.
type ActivationStreamAssembly struct {
	Stream        StreamKind
	Step          uint64
	Rank          uint32
	World         uint32
	ChunkCount    uint64
	Dtype         StreamDtype
	Bytes         []byte
	StreamRoot    hash.Hash32
	ChunkDigests  []hash.Hash32
	ChunksDecoded uint64
}

func streamMetaText(m map[string]wcb.Value, key string) (string, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.Text {
		return "", werrors.Malformedf("ACTIVATION_STREAM metadata %q must be text", key)
	}
	return v.Str, nil
}

func streamMetaU64(m map[string]wcb.Value, key string) (uint64, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.U64 {
		return 0, werrors.Malformedf("ACTIVATION_STREAM metadata %q must be u64", key)
	}
	return v.U, nil
}

func streamMetaU32(m map[string]wcb.Value, key string) (uint32, error) {
	value, err := streamMetaU64(m, key)
	if err != nil {
		return 0, err
	}
	if value > math.MaxUint32 {
		return 0, werrors.Malformedf(
			"ACTIVATION_STREAM metadata %q = %d exceeds u32", key, value)
	}
	return uint32(value), nil
}

func streamMetaShape(m map[string]wcb.Value) ([]uint64, error) {
	v, ok := m["shape"]
	if !ok || v.Kind != wcb.Array {
		return nil, werrors.NewMalformed(
			"ACTIVATION_STREAM metadata \"shape\" must be an array")
	}
	shape := make([]uint64, 0, len(v.Array))
	for _, item := range v.Array {
		if item.Kind != wcb.U64 {
			return nil, werrors.NewMalformed(
				"ACTIVATION_STREAM metadata \"shape\" elements must be u64")
		}
		shape = append(shape, item.U)
	}
	return shape, nil
}

func streamMetaHash(m map[string]wcb.Value, key string) (hash.Hash32, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.Bytes {
		return hash.Hash32{}, werrors.Malformedf(
			"ACTIVATION_STREAM metadata %q must be bytes", key)
	}
	if len(v.Bytes) != hash.HashLen {
		return hash.Hash32{}, werrors.Malformedf(
			"ACTIVATION_STREAM metadata %q must be %d bytes, got %d",
			key, hash.HashLen, len(v.Bytes))
	}
	return hash.FromSlice(v.Bytes)
}

func streamExactKeys(m map[string]wcb.Value, expected []string) error {
	for key := range m {
		if !slices.Contains(expected, key) {
			return werrors.Malformedf(
				"ACTIVATION_STREAM metadata has unknown key %q", key)
		}
	}
	for _, key := range expected {
		if _, ok := m[key]; !ok {
			return werrors.Malformedf(
				"ACTIVATION_STREAM metadata is missing key %q", key)
		}
	}
	return nil
}

func streamShapeProduct(shape []uint64, lim limits.Limits) (uint64, error) {
	if len(shape) == 0 {
		return 0, werrors.NewMalformed(
			"ACTIVATION_STREAM shape rank must be at least 1")
	}
	if uint64(len(shape)) > lim.MaxTensorRank {
		return 0, werrors.LimitExceededf(
			"ACTIVATION_STREAM rank %d exceeds limit %d", len(shape), lim.MaxTensorRank)
	}
	product := uint64(1)
	for _, dim := range shape {
		if dim == 0 {
			return 0, werrors.NewMalformed(
				"ACTIVATION_STREAM shape dimensions must be greater than zero")
		}
		var ok bool
		product, ok = mulChecked(product, dim)
		if !ok {
			return 0, werrors.NewLimitExceeded(
				"ACTIVATION_STREAM shape product overflows u64")
		}
	}
	if product > lim.MaxTensorElements {
		return 0, werrors.LimitExceededf(
			"ACTIVATION_STREAM element count %d exceeds limit %d",
			product, lim.MaxTensorElements)
	}
	return product, nil
}

func validateStreamHead(head ActivationStreamHead, lim limits.Limits) (uint64, error) {
	if head.World == 0 {
		return 0, werrors.NewMalformed("ACTIVATION_STREAM world must be at least 1")
	}
	if head.Rank >= head.World {
		return 0, werrors.Malformedf(
			"ACTIVATION_STREAM rank %d must be below world %d", head.Rank, head.World)
	}
	if head.ChunkCount == 0 {
		return 0, werrors.NewMalformed(
			"ACTIVATION_STREAM chunk_count must be at least 1")
	}
	elementCount, err := streamShapeProduct(head.Shape, lim)
	if err != nil {
		return 0, err
	}
	bytes, ok := mulChecked(elementCount, head.Dtype.WordBytes)
	if !ok {
		return 0, werrors.NewLimitExceeded(
			"ACTIVATION_STREAM chunk byte count overflows u64")
	}
	if err := limits.CheckLen(bytes, lim.MaxDecodedBytes, "ACTIVATION_STREAM chunk"); err != nil {
		return 0, err
	}
	return bytes, nil
}

// ValidateActivationStreamMeta validates the descriptor and returns the chunk
// byte length.
func ValidateActivationStreamMeta(meta ActivationStreamMeta, lim limits.Limits) (uint64, error) {
	if meta.PlaneCodec != ActivationStreamPlaneCodec {
		return 0, werrors.Malformedf(
			"ACTIVATION_STREAM plane_codec %q is not implemented (only %s)",
			meta.PlaneCodec, ActivationStreamPlaneCodec)
	}
	head := ActivationStreamHead{
		Stream: meta.Stream, Step: meta.Step, Rank: meta.Rank, World: meta.World,
		ChunkCount: meta.ChunkCount, Dtype: meta.Dtype, Shape: meta.Shape,
	}
	bytes, err := validateStreamHead(head, lim)
	if err != nil {
		return 0, err
	}
	if meta.ChunkIndex >= meta.ChunkCount {
		return 0, werrors.Malformedf(
			"ACTIVATION_STREAM chunk_index %d is not below chunk_count %d",
			meta.ChunkIndex, meta.ChunkCount)
	}
	product, err := streamShapeProduct(meta.Shape, lim)
	if err != nil {
		return 0, err
	}
	if meta.ElementCount != product {
		return 0, werrors.Malformedf(
			"ACTIVATION_STREAM element_count %d does not match shape product %d",
			meta.ElementCount, product)
	}
	return bytes, nil
}

// activationStreamDescriptorValue builds the canonical WCB descriptor value.
func activationStreamDescriptorValue(meta ActivationStreamMeta) wcb.Value {
	shape := make([]wcb.Value, 0, len(meta.Shape))
	for _, dim := range meta.Shape {
		shape = append(shape, wcb.U64Value(dim))
	}
	return wcb.MapValue(map[string]wcb.Value{
		"schema":        wcb.TextValue(ActivationStreamSchema),
		"stream":        wcb.TextValue(meta.Stream.Name),
		"step":          wcb.U64Value(meta.Step),
		"rank":          wcb.U64Value(uint64(meta.Rank)),
		"world":         wcb.U64Value(uint64(meta.World)),
		"chunk_index":   wcb.U64Value(meta.ChunkIndex),
		"chunk_count":   wcb.U64Value(meta.ChunkCount),
		"dtype":         wcb.TextValue(meta.Dtype.Name),
		"shape":         wcb.ArrayValue(shape),
		"element_count": wcb.U64Value(meta.ElementCount),
		"digest_root":   wcb.BytesValue(meta.DigestRoot[:]),
		"plane_codec":   wcb.TextValue(meta.PlaneCodec),
		"payload_len":   wcb.U64Value(meta.PayloadLen),
	})
}

func parseActivationStreamDescriptor(bytes []byte, lim limits.Limits) (ActivationStreamMeta, error) {
	value, err := wcb.Decode(bytes, lim)
	if err != nil {
		return ActivationStreamMeta{}, err
	}
	if value.Kind != wcb.Map {
		return ActivationStreamMeta{}, werrors.NewMalformed(
			"ACTIVATION_STREAM metadata must be a WCB map")
	}
	mapValue := value.Map
	keys := []string{
		"schema", "stream", "step", "rank", "world", "chunk_index",
		"chunk_count", "dtype", "shape", "element_count", "digest_root",
		"plane_codec", "payload_len",
	}
	if err := streamExactKeys(mapValue, keys); err != nil {
		return ActivationStreamMeta{}, err
	}
	schema, err := streamMetaText(mapValue, "schema")
	if err != nil {
		return ActivationStreamMeta{}, err
	}
	if schema != ActivationStreamSchema {
		return ActivationStreamMeta{}, werrors.Malformedf(
			"unsupported ACTIVATION_STREAM schema %q", schema)
	}
	streamName, err := streamMetaText(mapValue, "stream")
	if err != nil {
		return ActivationStreamMeta{}, err
	}
	stream, err := ParseStreamKind(streamName)
	if err != nil {
		return ActivationStreamMeta{}, err
	}
	step, err := streamMetaU64(mapValue, "step")
	if err != nil {
		return ActivationStreamMeta{}, err
	}
	rank, err := streamMetaU32(mapValue, "rank")
	if err != nil {
		return ActivationStreamMeta{}, err
	}
	world, err := streamMetaU32(mapValue, "world")
	if err != nil {
		return ActivationStreamMeta{}, err
	}
	chunkIndex, err := streamMetaU64(mapValue, "chunk_index")
	if err != nil {
		return ActivationStreamMeta{}, err
	}
	chunkCount, err := streamMetaU64(mapValue, "chunk_count")
	if err != nil {
		return ActivationStreamMeta{}, err
	}
	dtypeName, err := streamMetaText(mapValue, "dtype")
	if err != nil {
		return ActivationStreamMeta{}, err
	}
	dtype, err := ParseStreamDtype(dtypeName)
	if err != nil {
		return ActivationStreamMeta{}, err
	}
	shape, err := streamMetaShape(mapValue)
	if err != nil {
		return ActivationStreamMeta{}, err
	}
	elementCount, err := streamMetaU64(mapValue, "element_count")
	if err != nil {
		return ActivationStreamMeta{}, err
	}
	digestRoot, err := streamMetaHash(mapValue, "digest_root")
	if err != nil {
		return ActivationStreamMeta{}, err
	}
	planeCodec, err := streamMetaText(mapValue, "plane_codec")
	if err != nil {
		return ActivationStreamMeta{}, err
	}
	payloadLen, err := streamMetaU64(mapValue, "payload_len")
	if err != nil {
		return ActivationStreamMeta{}, err
	}
	meta := ActivationStreamMeta{
		Stream: stream, Step: step, Rank: rank, World: world,
		ChunkIndex: chunkIndex, ChunkCount: chunkCount, Dtype: dtype,
		Shape: shape, ElementCount: elementCount, DigestRoot: digestRoot,
		PlaneCodec: planeCodec, PayloadLen: payloadLen,
	}
	if _, err := ValidateActivationStreamMeta(meta, lim); err != nil {
		return ActivationStreamMeta{}, err
	}
	return meta, nil
}

// ActivationChunkMetadata parses and validates an ACTIVATION_STREAM descriptor.
func ActivationChunkMetadata(f *frame.Frame, lim limits.Limits) (ActivationStreamMeta, error) {
	if f.Header.Mode != frame.ModeActivationStream {
		return ActivationStreamMeta{}, werrors.Malformedf(
			"expected an ACTIVATION_STREAM frame, got mode %s", f.Header.Mode)
	}
	descriptor, _, err := splitAIBodyPrefix(f.Body)
	if err != nil {
		return ActivationStreamMeta{}, err
	}
	return parseActivationStreamDescriptor(descriptor, lim)
}

func buildStreamPlanes(dtype StreamDtype, bytes []byte, elementCount int) [][]byte {
	planes := make([][]byte, 0, dtype.Planes)
	for planeIndex := 0; planeIndex < int(dtype.Planes); planeIndex++ {
		plane := make([]byte, 0, elementCount)
		for index := 0; index < elementCount; index++ {
			var word uint32
			if dtype.WordBytes == 2 {
				offset := index * 2
				word = uint32(bytes[offset]) | (uint32(bytes[offset+1]) << 8)
			} else {
				offset := index * 4
				word = uint32(bytes[offset]) |
					(uint32(bytes[offset+1]) << 8) |
					(uint32(bytes[offset+2]) << 16) |
					(uint32(bytes[offset+3]) << 24)
			}
			plane = append(plane, byte(word>>(8*uint(planeIndex))))
		}
		planes = append(planes, plane)
	}
	return planes
}

func decodeStreamPlanes(dtype StreamDtype, payload []byte, byteLen uint64) ([]byte, error) {
	if byteLen%dtype.WordBytes != 0 {
		return nil, werrors.NewMalformed(
			"ACTIVATION_STREAM byte length is not a whole number of elements")
	}
	elementCount, err := limits.ToInt(byteLen/dtype.WordBytes, "ACTIVATION_STREAM element count")
	if err != nil {
		return nil, err
	}
	planes := make([][]byte, 0, dtype.Planes)
	offset := 0
	for planeIndex := uint64(0); planeIndex < dtype.Planes; planeIndex++ {
		if offset >= len(payload) {
			return nil, werrors.NewCorrupt(
				"ACTIVATION_STREAM payload ends before all declared planes")
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
			"ACTIVATION_STREAM payload has trailing bytes after the declared planes")
	}
	bytes := make([]byte, 0, int(byteLen))
	for index := 0; index < elementCount; index++ {
		var word uint32
		for planeIndex, plane := range planes {
			word |= uint32(plane[index]) << (8 * uint(planeIndex))
		}
		if dtype.WordBytes == 2 {
			bytes = append(bytes, byte(word), byte(word>>8))
		} else {
			bytes = append(bytes, byte(word), byte(word>>8), byte(word>>16), byte(word>>24))
		}
	}
	return bytes, nil
}

// EncodeActivationStreamDescriptorFrame builds a frame from an explicit
// descriptor and stored payload.
func EncodeActivationStreamDescriptorFrame(
	meta ActivationStreamMeta,
	payload []byte,
	decodedLen uint64,
	contentHash hash.Hash32,
	lim limits.Limits,
) ([]byte, error) {
	body, err := buildAIBody(activationStreamDescriptorValue(meta), payload, lim)
	if err != nil {
		return nil, err
	}
	return frame.Encode(frame.Parts{
		FormatMajor:  frame.FormatMajor,
		FormatMinor:  frame.FormatMinor,
		Domain:       frame.DomainPayload,
		Mode:         frame.ModeActivationStream,
		Flags:        0,
		Level:        0,
		DecodedLen:   decodedLen,
		ContentHash:  contentHash,
		Dependencies: nil,
		Chunks:       nil,
		Body:         body,
	}, lim)
}

// EncodeActivationChunk encodes one activation/gradient chunk.
func EncodeActivationChunk(
	head ActivationStreamHead,
	chunkIndex uint64,
	elements []byte,
	lim limits.Limits,
) ([]byte, error) {
	expected, err := validateStreamHead(head, lim)
	if err != nil {
		return nil, err
	}
	if chunkIndex >= head.ChunkCount {
		return nil, werrors.Malformedf(
			"ACTIVATION_STREAM chunk_index %d is not below chunk_count %d",
			chunkIndex, head.ChunkCount)
	}
	if uint64(len(elements)) != expected {
		return nil, werrors.Malformedf(
			"ACTIVATION_STREAM chunk is %d bytes but the descriptor requires %d",
			len(elements), expected)
	}
	elementCount, err := streamShapeProduct(head.Shape, lim)
	if err != nil {
		return nil, err
	}
	count, err := limits.ToInt(elementCount, "ACTIVATION_STREAM element count")
	if err != nil {
		return nil, err
	}
	planes := buildStreamPlanes(head.Dtype, elements, count)
	var payload []byte
	for _, plane := range planes {
		payload = append(payload, HuffmanEncode(plane)...)
	}
	meta := ActivationStreamMeta{
		Stream: head.Stream, Step: head.Step, Rank: head.Rank, World: head.World,
		ChunkIndex: chunkIndex, ChunkCount: head.ChunkCount, Dtype: head.Dtype,
		Shape: append([]uint64(nil), head.Shape...), ElementCount: elementCount,
		DigestRoot: hash.Domain(tensor.DomainTensorValues, elements),
		PlaneCodec: ActivationStreamPlaneCodec,
		PayloadLen: uint64(len(payload)),
	}
	if _, err := ValidateActivationStreamMeta(meta, lim); err != nil {
		return nil, err
	}
	return EncodeActivationStreamDescriptorFrame(
		meta, payload, uint64(len(elements)), hash.Domain(hash.DomainChunk, elements), lim)
}

// DecodeActivationChunk decodes one chunk, verifying its descriptor, digest,
// and content hash. No other chunk is read or decoded.
func DecodeActivationChunk(f *frame.Frame, lim limits.Limits) (DecodedActivationChunk, error) {
	if f.Header.Mode != frame.ModeActivationStream {
		return DecodedActivationChunk{}, werrors.Malformedf(
			"expected an ACTIVATION_STREAM frame, got mode %s", f.Header.Mode)
	}
	descriptor, payload, err := splitAIBodyPrefix(f.Body)
	if err != nil {
		return DecodedActivationChunk{}, err
	}
	meta, err := parseActivationStreamDescriptor(descriptor, lim)
	if err != nil {
		return DecodedActivationChunk{}, err
	}
	byteLen, err := ValidateActivationStreamMeta(meta, lim)
	if err != nil {
		return DecodedActivationChunk{}, err
	}
	if f.Header.DecodedLen != byteLen {
		return DecodedActivationChunk{}, werrors.Corruptf(
			"ACTIVATION_STREAM decoded_len is %d but the descriptor requires %d",
			f.Header.DecodedLen, byteLen)
	}
	if meta.PayloadLen != uint64(len(payload)) {
		return DecodedActivationChunk{}, werrors.Malformedf(
			"ACTIVATION_STREAM payload is %d bytes but payload_len declares %d",
			len(payload), meta.PayloadLen)
	}
	bytes, err := decodeStreamPlanes(meta.Dtype, payload, byteLen)
	if err != nil {
		return DecodedActivationChunk{}, err
	}
	if uint64(len(bytes)) != byteLen {
		return DecodedActivationChunk{}, werrors.NewCorrupt(
			"ACTIVATION_STREAM reconstructed chunk length is inconsistent")
	}
	if hash.Domain(tensor.DomainTensorValues, bytes) != meta.DigestRoot {
		return DecodedActivationChunk{}, werrors.NewCorrupt(
			"ACTIVATION_STREAM digest_root mismatch")
	}
	if hash.Domain(hash.DomainChunk, bytes) != f.Header.ContentHash {
		return DecodedActivationChunk{}, werrors.NewCorrupt(
			"ACTIVATION_STREAM content_hash mismatch against the reconstructed chunk")
	}
	return DecodedActivationChunk{Meta: meta, Bytes: bytes}, nil
}

// DecodeActivationChunkRange decodes a byte sub-range inside one chunk.
func DecodeActivationChunkRange(
	f *frame.Frame,
	start, end uint64,
	lim limits.Limits,
) ([]byte, error) {
	chunk, err := DecodeActivationChunk(f, lim)
	if err != nil {
		return nil, err
	}
	length := uint64(len(chunk.Bytes))
	if start > end || end > length {
		return nil, werrors.Malformedf(
			"ACTIVATION_STREAM range %d..%d is outside the %d-byte chunk",
			start, end, length)
	}
	return append([]byte(nil), chunk.Bytes[start:end]...), nil
}

// StreamRoot computes the normative stream root from per-chunk digests.
func StreamRoot(
	stream StreamKind,
	step uint64,
	rank, world uint32,
	chunkCount uint64,
	digestRoots []hash.Hash32,
	lim limits.Limits,
) (hash.Hash32, error) {
	if uint64(len(digestRoots)) != chunkCount {
		return hash.Hash32{}, werrors.Malformedf(
			"ACTIVATION_STREAM stream root needs %d digests, got %d",
			chunkCount, len(digestRoots))
	}
	digests := make([]wcb.Value, 0, len(digestRoots))
	for _, digest := range digestRoots {
		digests = append(digests, wcb.BytesValue(digest[:]))
	}
	value := wcb.MapValue(map[string]wcb.Value{
		"stream":       wcb.TextValue(stream.Name),
		"step":         wcb.U64Value(step),
		"rank":         wcb.U64Value(uint64(rank)),
		"world":        wcb.U64Value(uint64(world)),
		"chunk_count":  wcb.U64Value(chunkCount),
		"digest_roots": wcb.ArrayValue(digests),
	})
	bytes, err := wcb.Encode(value, lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(DomainAIStream, bytes), nil
}

// AssembleStream assembles one stream from its chunk frames. Frames may arrive
// in any order; duplicate or missing chunk indices are rejected.
func AssembleStream(frames [][]byte, lim limits.Limits) (ActivationStreamAssembly, error) {
	if len(frames) == 0 {
		return ActivationStreamAssembly{}, werrors.NewMalformed(
			"ACTIVATION_STREAM assembly needs at least one chunk")
	}
	decoded := make([]DecodedActivationChunk, 0, len(frames))
	for _, bytes := range frames {
		f, err := frame.Parse(bytes, lim)
		if err != nil {
			return ActivationStreamAssembly{}, err
		}
		chunk, err := DecodeActivationChunk(f, lim)
		if err != nil {
			return ActivationStreamAssembly{}, err
		}
		decoded = append(decoded, chunk)
	}
	first := decoded[0].Meta
	if uint64(len(frames)) != first.ChunkCount {
		return ActivationStreamAssembly{}, werrors.Malformedf(
			"ACTIVATION_STREAM has %d chunk frames but chunk_count is %d",
			len(frames), first.ChunkCount)
	}
	for _, chunk := range decoded {
		meta := chunk.Meta
		if meta.Stream.Name != first.Stream.Name ||
			meta.Step != first.Step || meta.Rank != first.Rank ||
			meta.World != first.World || meta.ChunkCount != first.ChunkCount ||
			meta.Dtype.Name != first.Dtype.Name {
			return ActivationStreamAssembly{}, werrors.NewMalformed(
				"ACTIVATION_STREAM chunk heads disagree on stream/step/rank/world/chunk_count/dtype")
		}
		if !slices.Equal(meta.Shape, first.Shape) || meta.ElementCount != first.ElementCount {
			return ActivationStreamAssembly{}, werrors.NewMalformed(
				"ACTIVATION_STREAM chunk heads disagree on shape/element_count")
		}
	}
	sort.Slice(decoded, func(i, j int) bool {
		return decoded[i].Meta.ChunkIndex < decoded[j].Meta.ChunkIndex
	})
	seen := make(map[uint64]struct{}, len(decoded))
	for _, chunk := range decoded {
		if _, ok := seen[chunk.Meta.ChunkIndex]; ok {
			return ActivationStreamAssembly{}, werrors.NonCanonicalf(
				"ACTIVATION_STREAM has duplicate chunk_index %d", chunk.Meta.ChunkIndex)
		}
		seen[chunk.Meta.ChunkIndex] = struct{}{}
	}
	for expected, chunk := range decoded {
		if chunk.Meta.ChunkIndex != uint64(expected) {
			return ActivationStreamAssembly{}, werrors.Malformedf(
				"ACTIVATION_STREAM is missing chunk_index %d", expected)
		}
	}
	var bytes []byte
	digests := make([]hash.Hash32, 0, len(decoded))
	for _, chunk := range decoded {
		bytes = append(bytes, chunk.Bytes...)
		digests = append(digests, chunk.Meta.DigestRoot)
	}
	root, err := StreamRoot(
		first.Stream, first.Step, first.Rank, first.World, first.ChunkCount, digests, lim)
	if err != nil {
		return ActivationStreamAssembly{}, err
	}
	return ActivationStreamAssembly{
		Stream: first.Stream, Step: first.Step, Rank: first.Rank,
		World: first.World, ChunkCount: first.ChunkCount, Dtype: first.Dtype,
		Bytes: bytes, StreamRoot: root, ChunkDigests: digests,
		ChunksDecoded: uint64(len(decoded)),
	}, nil
}

// ActivationStreamWriter appends chunks in order without decoding a previous
// chunk. DecodedChunks is incremented only by Assemble.
type ActivationStreamWriter struct {
	head          ActivationStreamHead
	nextIndex     uint64
	encodedChunks uint64
	decodedChunks uint64
}

// NewActivationStreamWriter creates a writer for one stream head.
func NewActivationStreamWriter(head ActivationStreamHead, lim limits.Limits) (*ActivationStreamWriter, error) {
	if _, err := validateStreamHead(head, lim); err != nil {
		return nil, err
	}
	return &ActivationStreamWriter{head: head}, nil
}

// AppendChunk encodes and appends the next chunk. This path never decodes a
// chunk.
func (w *ActivationStreamWriter) AppendChunk(elements []byte, lim limits.Limits) ([]byte, error) {
	frameBytes, err := EncodeActivationChunk(w.head, w.nextIndex, elements, lim)
	if err != nil {
		return nil, err
	}
	if w.nextIndex == math.MaxUint64 {
		return nil, werrors.NewLimitExceeded("ACTIVATION_STREAM chunk index overflows u64")
	}
	w.nextIndex++
	w.encodedChunks++
	return frameBytes, nil
}

// EncodedChunks reports chunks encoded through this writer.
func (w *ActivationStreamWriter) EncodedChunks() uint64 { return w.encodedChunks }

// DecodedChunks reports chunks decoded through this writer.
func (w *ActivationStreamWriter) DecodedChunks() uint64 { return w.decodedChunks }

// NextIndex reports the next implicit chunk index.
func (w *ActivationStreamWriter) NextIndex() uint64 { return w.nextIndex }

// Assemble assembles the supplied frames and counts the chunks it decoded.
func (w *ActivationStreamWriter) Assemble(frames [][]byte, lim limits.Limits) (ActivationStreamAssembly, error) {
	assembly, err := AssembleStream(frames, lim)
	if err != nil {
		return ActivationStreamAssembly{}, err
	}
	w.decodedChunks += assembly.ChunksDecoded
	return assembly, nil
}

// ActivationChunkSet is an ordered set of chunk frames that decodes one chunk
// at a time.
type ActivationChunkSet struct {
	frames        [][]byte
	decodedChunks uint64
}

// NewActivationChunkSet builds a set from chunk frames in chunk-index order.
func NewActivationChunkSet(frames [][]byte) *ActivationChunkSet {
	return &ActivationChunkSet{frames: frames}
}

// Len reports the number of chunk frames.
func (s *ActivationChunkSet) Len() int { return len(s.frames) }

// DecodedChunks reports chunk-level decodes performed through this set.
func (s *ActivationChunkSet) DecodedChunks() uint64 { return s.decodedChunks }

// DecodeChunk decodes exactly one chunk by index.
func (s *ActivationChunkSet) DecodeChunk(index uint64, lim limits.Limits) (DecodedActivationChunk, error) {
	if index >= uint64(len(s.frames)) {
		return DecodedActivationChunk{}, werrors.Malformedf(
			"ACTIVATION_STREAM chunk index %d is outside the %d-chunk set",
			index, len(s.frames))
	}
	f, err := frame.Parse(s.frames[index], lim)
	if err != nil {
		return DecodedActivationChunk{}, err
	}
	chunk, err := DecodeActivationChunk(f, lim)
	if err != nil {
		return DecodedActivationChunk{}, err
	}
	if chunk.Meta.ChunkIndex != index {
		return DecodedActivationChunk{}, werrors.Corruptf(
			"ACTIVATION_STREAM chunk frame at slot %d declares chunk_index %d",
			index, chunk.Meta.ChunkIndex)
	}
	if chunk.Meta.ChunkCount != uint64(len(s.frames)) {
		return DecodedActivationChunk{}, werrors.Corruptf(
			"ACTIVATION_STREAM chunk declares chunk_count %d but the set has %d frames",
			chunk.Meta.ChunkCount, len(s.frames))
	}
	s.decodedChunks++
	return chunk, nil
}

// ReadChunkRange decodes a byte sub-range inside exactly one chunk.
func (s *ActivationChunkSet) ReadChunkRange(index, start, end uint64, lim limits.Limits) ([]byte, error) {
	chunk, err := s.DecodeChunk(index, lim)
	if err != nil {
		return nil, err
	}
	length := uint64(len(chunk.Bytes))
	if start > end || end > length {
		return nil, werrors.Malformedf(
			"ACTIVATION_STREAM range %d..%d is outside the %d-byte chunk",
			start, end, length)
	}
	return append([]byte(nil), chunk.Bytes[start:end]...), nil
}
