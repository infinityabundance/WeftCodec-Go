// Package modes implements mode 12 KV_CACHE (WEFT-AI-CODEC §13).
package modes

import (
	"math"
	"slices"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// KvCacheSchema is the canonical descriptor schema.
const KvCacheSchema = "weft.ai.kv_cache.v1"

// KvCachePlaneCodec is the only plane coder implemented in v1.
const KvCachePlaneCodec = HuffmanCoderID

// KvDtype describes one KV cache element dtype.
type KvDtype struct {
	Name      string
	WordBytes uint64
	Planes    uint64
}

// ParseKvDtype parses a canonical KV_CACHE dtype name.
func ParseKvDtype(name string) (KvDtype, error) {
	switch name {
	case "bf16":
		return KvDtype{Name: name, WordBytes: 2, Planes: 2}, nil
	case "fp16":
		return KvDtype{Name: name, WordBytes: 2, Planes: 2}, nil
	case "fp32":
		return KvDtype{Name: name, WordBytes: 4, Planes: 4}, nil
	default:
		return KvDtype{}, werrors.Malformedf(
			"unknown KV_CACHE dtype %q (v1 allows bf16, fp16, fp32)", name)
	}
}

// KvCacheMeta is a parsed mode-12 descriptor.
type KvCacheMeta struct {
	Model      string
	Layer      uint32
	Session    string
	PageIndex  uint64
	PageCount  uint64
	SeqStart   uint64
	SeqEnd     uint64
	Dtype      KvDtype
	HeadCount  uint32
	HeadDim    uint32
	Shape      []uint64
	PageBytes  uint64
	PlaneCodec string
	PayloadLen uint64
}

// KvCacheInput describes one page to encode.
type KvCacheInput struct {
	Model     string
	Layer     uint32
	Session   string
	PageIndex uint64
	PageCount uint64
	SeqStart  uint64
	SeqEnd    uint64
	Dtype     KvDtype
	HeadCount uint32
	HeadDim   uint32
}

// DecodedKvPage is one decoded page.
type DecodedKvPage struct {
	Meta  KvCacheMeta
	Bytes []byte
}

func kvMetaText(m map[string]wcb.Value, key string) (string, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.Text {
		return "", werrors.Malformedf("KV_CACHE metadata %q must be text", key)
	}
	return v.Str, nil
}

func kvMetaU64(m map[string]wcb.Value, key string) (uint64, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.U64 {
		return 0, werrors.Malformedf("KV_CACHE metadata %q must be u64", key)
	}
	return v.U, nil
}

func kvMetaU32(m map[string]wcb.Value, key string) (uint32, error) {
	value, err := kvMetaU64(m, key)
	if err != nil {
		return 0, err
	}
	if value > math.MaxUint32 {
		return 0, werrors.Malformedf(
			"KV_CACHE metadata %q = %d exceeds u32", key, value)
	}
	return uint32(value), nil
}

func kvMetaShape(m map[string]wcb.Value) ([]uint64, error) {
	v, ok := m["shape"]
	if !ok || v.Kind != wcb.Array {
		return nil, werrors.NewMalformed("KV_CACHE metadata \"shape\" must be an array")
	}
	shape := make([]uint64, 0, len(v.Array))
	for _, item := range v.Array {
		if item.Kind != wcb.U64 {
			return nil, werrors.NewMalformed(
				"KV_CACHE metadata \"shape\" elements must be u64")
		}
		shape = append(shape, item.U)
	}
	return shape, nil
}

func kvExactKeys(m map[string]wcb.Value, expected []string) error {
	for key := range m {
		if !slices.Contains(expected, key) {
			return werrors.Malformedf("KV_CACHE metadata has unknown key %q", key)
		}
	}
	for _, key := range expected {
		if _, ok := m[key]; !ok {
			return werrors.Malformedf("KV_CACHE metadata is missing key %q", key)
		}
	}
	return nil
}

// ValidateKvCacheMeta validates the descriptor and returns the page byte
// length.
func ValidateKvCacheMeta(meta KvCacheMeta, lim limits.Limits) (uint64, error) {
	if meta.PlaneCodec != KvCachePlaneCodec {
		return 0, werrors.Malformedf(
			"KV_CACHE plane_codec %q is not implemented (only %s)",
			meta.PlaneCodec, KvCachePlaneCodec)
	}
	if meta.PageCount == 0 {
		return 0, werrors.NewMalformed("KV_CACHE page_count must be at least 1")
	}
	if meta.PageIndex >= meta.PageCount {
		return 0, werrors.Malformedf(
			"KV_CACHE page_index %d is not below page_count %d",
			meta.PageIndex, meta.PageCount)
	}
	if meta.SeqStart >= meta.SeqEnd {
		return 0, werrors.Malformedf(
			"KV_CACHE seq_start %d must be less than seq_end %d",
			meta.SeqStart, meta.SeqEnd)
	}
	rows := meta.SeqEnd - meta.SeqStart
	if meta.HeadCount == 0 {
		return 0, werrors.NewMalformed("KV_CACHE head_count must be greater than zero")
	}
	if meta.HeadDim == 0 {
		return 0, werrors.NewMalformed("KV_CACHE head_dim must be greater than zero")
	}
	expectedShape := []uint64{rows, uint64(meta.HeadCount), uint64(meta.HeadDim)}
	if !slices.Equal(meta.Shape, expectedShape) {
		return 0, werrors.Malformedf(
			"KV_CACHE shape %v must equal [page_rows, head_count, head_dim] = %v",
			meta.Shape, expectedShape)
	}
	if uint64(len(meta.Shape)) > lim.MaxTensorRank {
		return 0, werrors.LimitExceededf(
			"KV_CACHE rank %d exceeds limit %d", len(meta.Shape), lim.MaxTensorRank)
	}
	elementCount, ok := mulChecked(rows, uint64(meta.HeadCount))
	if ok {
		elementCount, ok = mulChecked(elementCount, uint64(meta.HeadDim))
	}
	if !ok {
		return 0, werrors.NewLimitExceeded("KV_CACHE element count overflows u64")
	}
	if elementCount > lim.MaxTensorElements {
		return 0, werrors.LimitExceededf(
			"KV_CACHE element count %d exceeds limit %d", elementCount, lim.MaxTensorElements)
	}
	computed, ok := mulChecked(elementCount, meta.Dtype.WordBytes)
	if !ok {
		return 0, werrors.NewLimitExceeded("KV_CACHE page byte count overflows u64")
	}
	if meta.PageBytes != computed {
		return 0, werrors.Malformedf(
			"KV_CACHE page_bytes %d does not equal page_rows * head_count * head_dim * word_bytes = %d",
			meta.PageBytes, computed)
	}
	if err := limits.CheckLen(meta.PageBytes, lim.MaxDecodedBytes, "KV_CACHE page"); err != nil {
		return 0, err
	}
	return computed, nil
}

func mulChecked(a, b uint64) (uint64, bool) {
	if b != 0 && a > math.MaxUint64/b {
		return 0, false
	}
	return a * b, true
}

// kvCacheDescriptorValue builds the canonical WCB descriptor value.
func kvCacheDescriptorValue(meta KvCacheMeta) wcb.Value {
	shape := make([]wcb.Value, 0, len(meta.Shape))
	for _, dim := range meta.Shape {
		shape = append(shape, wcb.U64Value(dim))
	}
	return wcb.MapValue(map[string]wcb.Value{
		"schema":      wcb.TextValue(KvCacheSchema),
		"model":       wcb.TextValue(meta.Model),
		"layer":       wcb.U64Value(uint64(meta.Layer)),
		"session":     wcb.TextValue(meta.Session),
		"page_index":  wcb.U64Value(meta.PageIndex),
		"page_count":  wcb.U64Value(meta.PageCount),
		"seq_start":   wcb.U64Value(meta.SeqStart),
		"seq_end":     wcb.U64Value(meta.SeqEnd),
		"dtype":       wcb.TextValue(meta.Dtype.Name),
		"head_count":  wcb.U64Value(uint64(meta.HeadCount)),
		"head_dim":    wcb.U64Value(uint64(meta.HeadDim)),
		"shape":       wcb.ArrayValue(shape),
		"page_bytes":  wcb.U64Value(meta.PageBytes),
		"plane_codec": wcb.TextValue(meta.PlaneCodec),
		"payload_len": wcb.U64Value(meta.PayloadLen),
	})
}

func parseKvCacheDescriptor(bytes []byte, lim limits.Limits) (KvCacheMeta, error) {
	value, err := wcb.Decode(bytes, lim)
	if err != nil {
		return KvCacheMeta{}, err
	}
	if value.Kind != wcb.Map {
		return KvCacheMeta{}, werrors.NewMalformed("KV_CACHE metadata must be a WCB map")
	}
	mapValue := value.Map
	keys := []string{
		"schema", "model", "layer", "session", "page_index", "page_count",
		"seq_start", "seq_end", "dtype", "head_count", "head_dim", "shape",
		"page_bytes", "plane_codec", "payload_len",
	}
	if err := kvExactKeys(mapValue, keys); err != nil {
		return KvCacheMeta{}, err
	}
	schema, err := kvMetaText(mapValue, "schema")
	if err != nil {
		return KvCacheMeta{}, err
	}
	if schema != KvCacheSchema {
		return KvCacheMeta{}, werrors.Malformedf("unsupported KV_CACHE schema %q", schema)
	}
	model, err := kvMetaText(mapValue, "model")
	if err != nil {
		return KvCacheMeta{}, err
	}
	layer, err := kvMetaU32(mapValue, "layer")
	if err != nil {
		return KvCacheMeta{}, err
	}
	session, err := kvMetaText(mapValue, "session")
	if err != nil {
		return KvCacheMeta{}, err
	}
	pageIndex, err := kvMetaU64(mapValue, "page_index")
	if err != nil {
		return KvCacheMeta{}, err
	}
	pageCount, err := kvMetaU64(mapValue, "page_count")
	if err != nil {
		return KvCacheMeta{}, err
	}
	seqStart, err := kvMetaU64(mapValue, "seq_start")
	if err != nil {
		return KvCacheMeta{}, err
	}
	seqEnd, err := kvMetaU64(mapValue, "seq_end")
	if err != nil {
		return KvCacheMeta{}, err
	}
	dtypeName, err := kvMetaText(mapValue, "dtype")
	if err != nil {
		return KvCacheMeta{}, err
	}
	dtype, err := ParseKvDtype(dtypeName)
	if err != nil {
		return KvCacheMeta{}, err
	}
	headCount, err := kvMetaU32(mapValue, "head_count")
	if err != nil {
		return KvCacheMeta{}, err
	}
	headDim, err := kvMetaU32(mapValue, "head_dim")
	if err != nil {
		return KvCacheMeta{}, err
	}
	shape, err := kvMetaShape(mapValue)
	if err != nil {
		return KvCacheMeta{}, err
	}
	pageBytes, err := kvMetaU64(mapValue, "page_bytes")
	if err != nil {
		return KvCacheMeta{}, err
	}
	planeCodec, err := kvMetaText(mapValue, "plane_codec")
	if err != nil {
		return KvCacheMeta{}, err
	}
	payloadLen, err := kvMetaU64(mapValue, "payload_len")
	if err != nil {
		return KvCacheMeta{}, err
	}
	meta := KvCacheMeta{
		Model: model, Layer: layer, Session: session,
		PageIndex: pageIndex, PageCount: pageCount,
		SeqStart: seqStart, SeqEnd: seqEnd, Dtype: dtype,
		HeadCount: headCount, HeadDim: headDim, Shape: shape,
		PageBytes: pageBytes, PlaneCodec: planeCodec, PayloadLen: payloadLen,
	}
	if _, err := ValidateKvCacheMeta(meta, lim); err != nil {
		return KvCacheMeta{}, err
	}
	return meta, nil
}

// KvCacheMetadata parses and validates a KV_CACHE descriptor from a frame.
func KvCacheMetadata(f *frame.Frame, lim limits.Limits) (KvCacheMeta, error) {
	if f.Header.Mode != frame.ModeKvCache {
		return KvCacheMeta{}, werrors.Malformedf(
			"expected a KV_CACHE frame, got mode %s", f.Header.Mode)
	}
	descriptor, _, err := splitAIBodyPrefix(f.Body)
	if err != nil {
		return KvCacheMeta{}, err
	}
	return parseKvCacheDescriptor(descriptor, lim)
}

func kvElementWord(dtype KvDtype, bytes []byte, index int) uint32 {
	if dtype.WordBytes == 2 {
		offset := index * 2
		return uint32(bytes[offset]) | (uint32(bytes[offset+1]) << 8)
	}
	offset := index * 4
	return uint32(bytes[offset]) |
		(uint32(bytes[offset+1]) << 8) |
		(uint32(bytes[offset+2]) << 16) |
		(uint32(bytes[offset+3]) << 24)
}

func buildKvPlanes(dtype KvDtype, bytes []byte, elementCount int) [][]byte {
	planes := make([][]byte, 0, dtype.Planes)
	for planeIndex := 0; planeIndex < int(dtype.Planes); planeIndex++ {
		plane := make([]byte, 0, elementCount)
		for index := 0; index < elementCount; index++ {
			word := kvElementWord(dtype, bytes, index)
			plane = append(plane, byte(word>>(8*uint(planeIndex))))
		}
		planes = append(planes, plane)
	}
	return planes
}

func decodeKvPlanes(dtype KvDtype, payload []byte, pageBytes uint64) ([]byte, error) {
	if pageBytes%dtype.WordBytes != 0 {
		return nil, werrors.NewMalformed(
			"KV_CACHE page byte length is not a whole number of elements")
	}
	elementCount, err := limits.ToInt(pageBytes/dtype.WordBytes, "KV_CACHE element count")
	if err != nil {
		return nil, err
	}
	planes := make([][]byte, 0, dtype.Planes)
	offset := 0
	for planeIndex := uint64(0); planeIndex < dtype.Planes; planeIndex++ {
		if offset >= len(payload) {
			return nil, werrors.NewCorrupt(
				"KV_CACHE payload ends before all declared planes")
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
			"KV_CACHE payload has trailing bytes after the declared planes")
	}
	bytes := make([]byte, 0, int(pageBytes))
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

// EncodeKvCacheDescriptorFrame builds a frame from an explicit descriptor and
// stored payload. It does not re-validate descriptor semantics.
func EncodeKvCacheDescriptorFrame(
	meta KvCacheMeta,
	payload []byte,
	decodedLen uint64,
	contentHash hash.Hash32,
	lim limits.Limits,
) ([]byte, error) {
	body, err := buildAIBody(kvCacheDescriptorValue(meta), payload, lim)
	if err != nil {
		return nil, err
	}
	return frame.Encode(frame.Parts{
		FormatMajor:  frame.FormatMajor,
		FormatMinor:  frame.FormatMinor,
		Domain:       frame.DomainPayload,
		Mode:         frame.ModeKvCache,
		Flags:        0,
		Level:        0,
		DecodedLen:   decodedLen,
		ContentHash:  contentHash,
		Dependencies: nil,
		Chunks:       nil,
		Body:         body,
	}, lim)
}

// EncodeKvPage encodes one KV cache page from canonical row-major
// little-endian element bytes.
func EncodeKvPage(input KvCacheInput, elements []byte, lim limits.Limits) ([]byte, error) {
	if input.SeqStart >= input.SeqEnd {
		return nil, werrors.Malformedf(
			"KV_CACHE seq_start %d must be less than seq_end %d",
			input.SeqStart, input.SeqEnd)
	}
	rows := input.SeqEnd - input.SeqStart
	elementCount, ok := mulChecked(rows, uint64(input.HeadCount))
	if ok {
		elementCount, ok = mulChecked(elementCount, uint64(input.HeadDim))
	}
	if !ok {
		return nil, werrors.NewLimitExceeded("KV_CACHE element count overflows u64")
	}
	pageBytes, ok := mulChecked(elementCount, input.Dtype.WordBytes)
	if !ok {
		return nil, werrors.NewLimitExceeded("KV_CACHE page byte count overflows u64")
	}
	if uint64(len(elements)) != pageBytes {
		return nil, werrors.Malformedf(
			"KV_CACHE page is %d bytes but the descriptor requires %d",
			len(elements), pageBytes)
	}
	if err := limits.CheckLen(pageBytes, lim.MaxDecodedBytes, "KV_CACHE page"); err != nil {
		return nil, err
	}
	meta := KvCacheMeta{
		Model: input.Model, Layer: input.Layer, Session: input.Session,
		PageIndex: input.PageIndex, PageCount: input.PageCount,
		SeqStart: input.SeqStart, SeqEnd: input.SeqEnd, Dtype: input.Dtype,
		HeadCount: input.HeadCount, HeadDim: input.HeadDim,
		Shape:     []uint64{rows, uint64(input.HeadCount), uint64(input.HeadDim)},
		PageBytes: pageBytes, PlaneCodec: KvCachePlaneCodec, PayloadLen: 0,
	}
	if _, err := ValidateKvCacheMeta(meta, lim); err != nil {
		return nil, err
	}
	count, err := limits.ToInt(elementCount, "KV_CACHE element count")
	if err != nil {
		return nil, err
	}
	planes := buildKvPlanes(input.Dtype, elements, count)
	var payload []byte
	for _, plane := range planes {
		payload = append(payload, HuffmanEncode(plane)...)
	}
	meta.PayloadLen = uint64(len(payload))
	return EncodeKvCacheDescriptorFrame(
		meta, payload, uint64(len(elements)), hash.Domain(hash.DomainChunk, elements), lim)
}

// DecodeKvPage decodes one KV cache page, verifying its descriptor, planes, and
// content hash. No other page is read or decoded.
func DecodeKvPage(f *frame.Frame, lim limits.Limits) (DecodedKvPage, error) {
	if f.Header.Mode != frame.ModeKvCache {
		return DecodedKvPage{}, werrors.Malformedf(
			"expected a KV_CACHE frame, got mode %s", f.Header.Mode)
	}
	descriptor, payload, err := splitAIBodyPrefix(f.Body)
	if err != nil {
		return DecodedKvPage{}, err
	}
	meta, err := parseKvCacheDescriptor(descriptor, lim)
	if err != nil {
		return DecodedKvPage{}, err
	}
	pageBytes, err := ValidateKvCacheMeta(meta, lim)
	if err != nil {
		return DecodedKvPage{}, err
	}
	if f.Header.DecodedLen != pageBytes {
		return DecodedKvPage{}, werrors.Corruptf(
			"KV_CACHE decoded_len is %d but the descriptor requires %d",
			f.Header.DecodedLen, pageBytes)
	}
	if meta.PayloadLen != uint64(len(payload)) {
		return DecodedKvPage{}, werrors.Malformedf(
			"KV_CACHE payload is %d bytes but payload_len declares %d",
			len(payload), meta.PayloadLen)
	}
	bytes, err := decodeKvPlanes(meta.Dtype, payload, pageBytes)
	if err != nil {
		return DecodedKvPage{}, err
	}
	if uint64(len(bytes)) != pageBytes {
		return DecodedKvPage{}, werrors.NewCorrupt(
			"KV_CACHE reconstructed page length is inconsistent")
	}
	if hash.Domain(hash.DomainChunk, bytes) != f.Header.ContentHash {
		return DecodedKvPage{}, werrors.NewCorrupt(
			"KV_CACHE content_hash mismatch against the reconstructed page")
	}
	return DecodedKvPage{Meta: meta, Bytes: bytes}, nil
}

// DecodeKvPageRange decodes a byte sub-range inside one page.
func DecodeKvPageRange(f *frame.Frame, start, end uint64, lim limits.Limits) ([]byte, error) {
	page, err := DecodeKvPage(f, lim)
	if err != nil {
		return nil, err
	}
	length := uint64(len(page.Bytes))
	if start > end || end > length {
		return nil, werrors.Malformedf(
			"KV_CACHE range %d..%d is outside the %d-byte page", start, end, length)
	}
	return append([]byte(nil), page.Bytes[start:end]...), nil
}

// KvPageSet is an ordered set of page frames that decodes one page at a time.
type KvPageSet struct {
	frames       [][]byte
	decodedPages uint64
}

// NewKvPageSet builds a set from page frames in page-index order.
func NewKvPageSet(frames [][]byte) *KvPageSet {
	return &KvPageSet{frames: frames}
}

// Len reports the number of page frames.
func (s *KvPageSet) Len() int { return len(s.frames) }

// DecodedPages reports page-level decodes performed through this set.
func (s *KvPageSet) DecodedPages() uint64 { return s.decodedPages }

// DecodePage decodes exactly one page by index. Other frames are untouched.
func (s *KvPageSet) DecodePage(index uint64, lim limits.Limits) (DecodedKvPage, error) {
	if index >= uint64(len(s.frames)) {
		return DecodedKvPage{}, werrors.Malformedf(
			"KV_CACHE page index %d is outside the %d-page set", index, len(s.frames))
	}
	f, err := frame.Parse(s.frames[index], lim)
	if err != nil {
		return DecodedKvPage{}, err
	}
	page, err := DecodeKvPage(f, lim)
	if err != nil {
		return DecodedKvPage{}, err
	}
	if page.Meta.PageIndex != index {
		return DecodedKvPage{}, werrors.Corruptf(
			"KV_CACHE page frame at slot %d declares page_index %d",
			index, page.Meta.PageIndex)
	}
	if page.Meta.PageCount != uint64(len(s.frames)) {
		return DecodedKvPage{}, werrors.Corruptf(
			"KV_CACHE page declares page_count %d but the set has %d frames",
			page.Meta.PageCount, len(s.frames))
	}
	s.decodedPages++
	return page, nil
}

// ReadPageRange decodes a byte sub-range inside exactly one page.
func (s *KvPageSet) ReadPageRange(index, start, end uint64, lim limits.Limits) ([]byte, error) {
	page, err := s.DecodePage(index, lim)
	if err != nil {
		return nil, err
	}
	length := uint64(len(page.Bytes))
	if start > end || end > length {
		return nil, werrors.Malformedf(
			"KV_CACHE range %d..%d is outside the %d-byte page", start, end, length)
	}
	return append([]byte(nil), page.Bytes[start:end]...), nil
}
