// Mode 20 ENGRAM_SHARD — incrementally updateable, region-addressable
// row-group tables (WEFT-AI-CODEC §21, spec/WEFT-ENGRAM.md).
//
// A rank-2 table is stored as ordered row groups. Each group is split into
// byte planes and each plane is coded with the frozen huffman-canonical-v1
// coder, exactly as mode 15 codes a tile. A group can be stored inline or
// emitted as a reference to an earlier artifact's group root; the latter is
// what makes an incremental update write only the changed groups.
package modes

import (
	"context"
	"math"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// EngramSchema is the canonical mode-20 descriptor schema.
const EngramSchema = "weft.ai.engram_shard.v1"

// EngramLayout is the only table layout (canonical row-major).
const EngramLayout = "row_major"

// Engram hash-function registry.
const (
	EngramHashSHA256 = "ngram-sha256-v1"
	EngramHashBLAKE3 = "ngram-blake3-v1"
)

// EngramQuantization is a declared, lossless quantization block.
type EngramQuantization struct {
	Scheme   string
	Lossless bool
}

// EngramPrecision is the exact storage precision of a table.
type EngramPrecision struct {
	StorageDtype string
	Quantization *EngramQuantization
}

// ExactEngramPrecision returns a precision with no declared transformation.
func ExactEngramPrecision(dtype string) EngramPrecision {
	return EngramPrecision{StorageDtype: dtype}
}

// EngramStorage names how one group's bytes are obtained.
const (
	EngramStorageInline    = "inline"
	EngramStorageReference = "reference"
)

// EngramGroup is one row group in a mode-20 descriptor.
type EngramGroup struct {
	Index         uint64
	RowStart      uint64
	RowEnd        uint64
	Storage       string
	Offset        uint64
	CompressedLen uint64
	Dependency    uint64
	HasDependency bool
	GroupRoot     hash.Hash32
	Supersedes    *hash.Hash32
}

// EngramMeta is a parsed mode-20 descriptor.
type EngramMeta struct {
	HashFunction  string
	VocabSize     uint64
	TableShape    []uint64
	Dtype         string
	Precision     EngramPrecision
	RowGroupEdges []uint64
	PayloadLen    uint64
	PayloadHash   hash.Hash32
	Groups        []EngramGroup
}

// Rows returns the table row count.
func (m EngramMeta) Rows() uint64 { return m.TableShape[0] }

// Dim returns the table row width.
func (m EngramMeta) Dim() uint64 { return m.TableShape[1] }

// ElementCount returns rows*dim.
func (m EngramMeta) ElementCount() uint64 { return m.Rows() * m.Dim() }

// DecodedLen returns the exact decoded table length in bytes.
func (m EngramMeta) DecodedLen() (uint64, error) {
	size, err := tensorElementSize(m.Dtype)
	if err != nil {
		return 0, err
	}
	if m.ElementCount() != 0 && size > math.MaxUint64/m.ElementCount() {
		return 0, werrors.NewLimitExceeded("engram decoded bytes overflow u64")
	}
	return m.ElementCount() * size, nil
}

// DescriptorRoot is H("weft/engram/v1", WCB(descriptor)).
func (m EngramMeta) DescriptorRoot(lim limits.Limits) (hash.Hash32, error) {
	encoded, err := wcb.Encode(engramMetaValue(m), lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(hash.DomainEngram, encoded), nil
}

// EngramDescriptor is a parsed frame with its stored payload slice.
type EngramDescriptor struct {
	Meta    EngramMeta
	Payload []byte
}

// EngramReference names an earlier artifact's group.
type EngramReference struct {
	GroupRoot  hash.Hash32
	Dependency uint64
}

// EngramInput is the encoder input.
type EngramInput struct {
	HashFunction  string
	VocabSize     uint64
	TableShape    []uint64
	Dtype         string
	Precision     EngramPrecision
	RowGroupEdges []uint64
	References    map[uint64]EngramReference
	Supersedes    []*hash.Hash32
}

// InlineEngramInput returns a fully inline table with no supersession edges.
func InlineEngramInput(dtype string, tableShape, rowGroupEdges []uint64, hashFunction string, vocabSize uint64) EngramInput {
	return EngramInput{
		HashFunction:  hashFunction,
		VocabSize:     vocabSize,
		TableShape:    append([]uint64(nil), tableShape...),
		Dtype:         dtype,
		Precision:     ExactEngramPrecision(dtype),
		RowGroupEdges: append([]uint64(nil), rowGroupEdges...),
		References:    map[uint64]EngramReference{},
	}
}

func engramValidateHashFunction(name string) error {
	switch name {
	case EngramHashSHA256, EngramHashBLAKE3:
		return nil
	default:
		return werrors.UnsupportedModef(
			"unknown engram hash_function %q (v1 allows %q or %q)",
			name, EngramHashSHA256, EngramHashBLAKE3)
	}
}

func engramValidatePrecision(dtype string, precision EngramPrecision) error {
	if precision.StorageDtype != dtype {
		return werrors.UnsupportedFeaturef(
			"engram storage_dtype %q differs from element_dtype %q; v1 has no storage transformation",
			precision.StorageDtype, dtype)
	}
	if precision.Quantization != nil {
		if precision.Quantization.Scheme != "none" {
			return werrors.UnsupportedFeaturef(
				"engram quantization scheme %q is not implemented (v1 allows none)",
				precision.Quantization.Scheme)
		}
		if !precision.Quantization.Lossless {
			return werrors.NewUnsupportedFeature(
				"engram quantization block declares lossless=false; a lossy table is rejected")
		}
	}
	return nil
}

func engramValidateEdges(shape, edges []uint64, lim limits.Limits) (uint64, uint64, error) {
	if len(shape) != 2 {
		return 0, 0, werrors.NewMalformed("engram table_shape must have rank exactly 2")
	}
	rows, dim := shape[0], shape[1]
	if rows == 0 || dim == 0 {
		return 0, 0, werrors.NewMalformed("engram table_shape extents must be greater than zero")
	}
	if len(edges) < 2 {
		return 0, 0, werrors.NewMalformed("engram row_group_edges must contain at least two edges")
	}
	if edges[0] != 0 {
		return 0, 0, werrors.NewMalformed("engram row_group_edges must start at 0")
	}
	if edges[len(edges)-1] != rows {
		return 0, 0, werrors.Malformedf(
			"engram last edge %d must equal the row count %d", edges[len(edges)-1], rows)
	}
	groupCount := uint64(len(edges) - 1)
	if groupCount > lim.MaxStructuredItems {
		return 0, 0, werrors.LimitExceededf(
			"engram group count %d exceeds limit %d", groupCount, lim.MaxStructuredItems)
	}
	for index := 1; index < len(edges); index++ {
		if edges[index] <= edges[index-1] {
			return 0, 0, werrors.Malformedf(
				"engram row_group_edges are not strictly increasing at %d: %d then %d",
				index, edges[index-1], edges[index])
		}
		groupRows := edges[index] - edges[index-1]
		if groupRows != 0 && dim > math.MaxUint64/groupRows {
			return 0, 0, werrors.NewLimitExceeded("engram group element count overflows u64")
		}
		elements := groupRows * dim
		if elements > lim.MaxEngramGroupElements {
			return 0, 0, werrors.LimitExceededf(
				"engram group %d has %d elements, limit is %d",
				index-1, elements, lim.MaxEngramGroupElements)
		}
	}
	return rows, dim, nil
}

func encodeEngramGroupPlanes(group []byte, elementSize int) []byte {
	if elementSize == 0 {
		return nil
	}
	elements := len(group) / elementSize
	out := make([]byte, 0, len(group))
	for plane := 0; plane < elementSize; plane++ {
		planeBytes := make([]byte, 0, elements)
		for index := plane; index < len(group); index += elementSize {
			planeBytes = append(planeBytes, group[index])
		}
		out = append(out, HuffmanEncode(planeBytes)...)
	}
	return out
}

func decodeEngramGroupPlanes(stored []byte, elementSize, elementLen int) ([]byte, error) {
	if elementSize == 0 || elementLen%elementSize != 0 {
		return nil, werrors.NewMalformed(
			"engram group length is not a whole number of elements")
	}
	elements := elementLen / elementSize
	planes := make([][]byte, 0, elementSize)
	cursor := 0
	for plane := 0; plane < elementSize; plane++ {
		if cursor > len(stored) {
			return nil, werrors.NewCorrupt("engram group plane is truncated")
		}
		decoded, consumed, err := HuffmanDecodePrefix(stored[cursor:], elements)
		if err != nil {
			return nil, err
		}
		cursor += consumed
		planes = append(planes, decoded)
	}
	if cursor != len(stored) {
		return nil, werrors.NewCorrupt(
			"engram group planes leave trailing bytes inside the group range")
	}
	out := make([]byte, elementLen)
	for planeIndex, plane := range planes {
		for element, b := range plane {
			out[element*elementSize+planeIndex] = b
		}
	}
	return out, nil
}

// EngramGroupRoot is H("weft/engram/group/v1", WCB({group_index, row_start,
// row_end, dtype, precision, values_hash})).
func EngramGroupRoot(dtype, precision string, groupIndex, rowStart, rowEnd uint64, groupBytes []byte, lim limits.Limits) (hash.Hash32, error) {
	valuesHash := hash.Domain(tensor.DomainTensorValues, groupBytes)
	value := wcb.MapValue(map[string]wcb.Value{
		"group_index": wcb.U64Value(groupIndex),
		"row_start":   wcb.U64Value(rowStart),
		"row_end":     wcb.U64Value(rowEnd),
		"dtype":       wcb.TextValue(dtype),
		"precision":   wcb.TextValue(precision),
		"values_hash": wcb.BytesValue(valuesHash.Bytes()),
	})
	encoded, err := wcb.Encode(value, lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(hash.DomainEngramGroup, encoded), nil
}

func prepareEngramGroups(table []byte, input EngramInput, lim limits.Limits) (EngramMeta, []byte, error) {
	if err := engramValidateHashFunction(input.HashFunction); err != nil {
		return EngramMeta{}, nil, err
	}
	if err := engramValidatePrecision(input.Dtype, input.Precision); err != nil {
		return EngramMeta{}, nil, err
	}
	rows, dim, err := engramValidateEdges(input.TableShape, input.RowGroupEdges, lim)
	if err != nil {
		return EngramMeta{}, nil, err
	}
	elementCount, err := tensor.ValidateShape(input.TableShape, lim)
	if err != nil {
		return EngramMeta{}, nil, err
	}
	if input.VocabSize == 0 || input.VocabSize > rows {
		return EngramMeta{}, nil, werrors.Malformedf(
			"engram vocab_size %d must be in 1..=%d", input.VocabSize, rows)
	}
	elementSize, err := tensorElementSize(input.Dtype)
	if err != nil {
		return EngramMeta{}, nil, err
	}
	if elementCount != 0 && elementSize > math.MaxUint64/elementCount {
		return EngramMeta{}, nil, werrors.NewLimitExceeded("engram decoded bytes overflow u64")
	}
	expectedLen := elementCount * elementSize
	if uint64(len(table)) != expectedLen {
		return EngramMeta{}, nil, werrors.Malformedf(
			"engram table is %d bytes but shape %v x %s requires %d",
			len(table), input.TableShape, input.Dtype, expectedLen)
	}
	if err := limits.CheckLen(uint64(len(table)), lim.MaxDecodedBytes, "engram table"); err != nil {
		return EngramMeta{}, nil, err
	}
	groupCount := len(input.RowGroupEdges) - 1
	var supersedes []*hash.Hash32
	if len(input.Supersedes) == 0 {
		supersedes = make([]*hash.Hash32, groupCount)
	} else if len(input.Supersedes) == groupCount {
		supersedes = input.Supersedes
	} else {
		return EngramMeta{}, nil, werrors.Malformedf(
			"engram supersedes has %d entries but there are %d groups",
			len(input.Supersedes), groupCount)
	}
	for index, reference := range input.References {
		if index >= uint64(groupCount) {
			return EngramMeta{}, nil, werrors.Malformedf(
				"engram reference names group %d but there are %d groups", index, groupCount)
		}
		if reference.GroupRoot == (hash.Hash32{}) {
			return EngramMeta{}, nil, werrors.NewMalformed(
				"engram reference group root must not be all zero")
		}
	}
	if dim != 0 && elementSize > math.MaxUint64/dim {
		return EngramMeta{}, nil, werrors.NewLimitExceeded("engram row byte length overflows u64")
	}
	rowBytes := dim * elementSize
	stored := make([]byte, 0, len(table))
	groups := make([]EngramGroup, 0, groupCount)
	for index := 0; index < groupCount; index++ {
		rowStart := input.RowGroupEdges[index]
		rowEnd := input.RowGroupEdges[index+1]
		if rowStart != 0 && rowBytes > math.MaxUint64/rowStart {
			return EngramMeta{}, nil, werrors.NewLimitExceeded("engram group start overflows u64")
		}
		byteStart := rowStart * rowBytes
		if rowEnd != 0 && rowBytes > math.MaxUint64/rowEnd {
			return EngramMeta{}, nil, werrors.NewLimitExceeded("engram group end overflows u64")
		}
		byteEnd := rowEnd * rowBytes
		if byteEnd < byteStart || byteEnd > uint64(len(table)) {
			return EngramMeta{}, nil, werrors.NewCorrupt(
				"engram group range is outside the table")
		}
		groupBytes := table[byteStart:byteEnd]
		if reference, ok := input.References[uint64(index)]; ok {
			groups = append(groups, EngramGroup{
				Index:         uint64(index),
				RowStart:      rowStart,
				RowEnd:        rowEnd,
				Storage:       EngramStorageReference,
				Dependency:    reference.Dependency,
				HasDependency: true,
				GroupRoot:     reference.GroupRoot,
				Supersedes:    supersedes[index],
			})
			continue
		}
		encoded := encodeEngramGroupPlanes(groupBytes, int(elementSize))
		root, err := EngramGroupRoot(
			input.Dtype, input.Precision.StorageDtype, uint64(index), rowStart, rowEnd, groupBytes, lim)
		if err != nil {
			return EngramMeta{}, nil, err
		}
		groups = append(groups, EngramGroup{
			Index:         uint64(index),
			RowStart:      rowStart,
			RowEnd:        rowEnd,
			Storage:       EngramStorageInline,
			Offset:        uint64(len(stored)),
			CompressedLen: uint64(len(encoded)),
			GroupRoot:     root,
			Supersedes:    supersedes[index],
		})
		stored = append(stored, encoded...)
	}
	payloadHash := hash.Domain(hash.DomainEngramPayload, stored)
	meta := EngramMeta{
		HashFunction:  input.HashFunction,
		VocabSize:     input.VocabSize,
		TableShape:    append([]uint64(nil), input.TableShape...),
		Dtype:         input.Dtype,
		Precision:     input.Precision,
		RowGroupEdges: append([]uint64(nil), input.RowGroupEdges...),
		PayloadLen:    uint64(len(stored)),
		PayloadHash:   payloadHash,
		Groups:        groups,
	}
	return meta, stored, nil
}

// EncodeEngram encodes a fully inline mode-20 frame.
func EncodeEngram(data []byte, input EngramInput, lim limits.Limits) ([]byte, error) {
	if len(input.References) != 0 {
		return nil, werrors.NewMalformed(
			"EncodeEngram cannot emit references without a dependency table; use EncodeEngramWithDependencies")
	}
	meta, stored, err := prepareEngramGroups(data, input, lim)
	if err != nil {
		return nil, err
	}
	body, err := buildEngramBody(meta, stored, lim)
	if err != nil {
		return nil, err
	}
	return frame.Encode(frame.PayloadParts(
		frame.ModeEngramShard, 0, 0, uint64(len(data)),
		hash.Domain(hash.DomainChunk, data), body), lim)
}

// EncodeEngramWithDependencies encodes a frame that references earlier
// artifacts. dependencyHashes[i] is the content hash named by
// EngramReference.Dependency == i.
func EncodeEngramWithDependencies(data []byte, input EngramInput, dependencyHashes []hash.Hash32, lim limits.Limits) ([]byte, error) {
	if uint64(len(dependencyHashes)) > lim.MaxDependencyCount {
		return nil, werrors.LimitExceededf(
			"engram dependency count %d exceeds limit %d",
			len(dependencyHashes), lim.MaxDependencyCount)
	}
	meta, stored, err := prepareEngramGroups(data, input, lim)
	if err != nil {
		return nil, err
	}
	for index, reference := range input.References {
		if reference.Dependency >= uint64(len(dependencyHashes)) {
			return nil, werrors.Malformedf(
				"engram reference group %d names dependency %d but only %d are supplied",
				index, reference.Dependency, len(dependencyHashes))
		}
	}
	body, err := buildEngramBody(meta, stored, lim)
	if err != nil {
		return nil, err
	}
	parts := frame.PayloadParts(
		frame.ModeEngramShard, 0, 0, uint64(len(data)),
		hash.Domain(hash.DomainChunk, data), body)
	deps := make([]frame.Dependency, 0, len(dependencyHashes))
	for _, h := range dependencyHashes {
		deps = append(deps, frame.NewDependency(h, uint64(len(data)), frame.DepDomainPayload, 20))
	}
	parts.Dependencies = deps
	return frame.Encode(parts, lim)
}

func buildEngramBody(meta EngramMeta, stored []byte, lim limits.Limits) ([]byte, error) {
	descriptor, err := wcb.Encode(engramMetaValue(meta), lim)
	if err != nil {
		return nil, err
	}
	if uint64(len(descriptor)) > math.MaxUint32 {
		return nil, werrors.NewLimitExceeded("engram descriptor does not fit u32")
	}
	body := make([]byte, 0, 4+len(descriptor)+len(stored))
	body = append(body, byte(len(descriptor)), byte(len(descriptor)>>8), byte(len(descriptor)>>16), byte(len(descriptor)>>24))
	body = append(body, descriptor...)
	body = append(body, stored...)
	return body, nil
}

func engramPrecisionValue(precision EngramPrecision) wcb.Value {
	m := map[string]wcb.Value{
		"storage_dtype": wcb.TextValue(precision.StorageDtype),
	}
	if precision.Quantization != nil {
		m["quantization"] = wcb.MapValue(map[string]wcb.Value{
			"scheme":   wcb.TextValue(precision.Quantization.Scheme),
			"lossless": wcb.BoolValue(precision.Quantization.Lossless),
		})
	}
	return wcb.MapValue(m)
}

func engramGroupValue(group EngramGroup) wcb.Value {
	dependency := wcb.NullValue()
	if group.HasDependency {
		dependency = wcb.U64Value(group.Dependency)
	}
	supersedes := wcb.NullValue()
	if group.Supersedes != nil {
		supersedes = wcb.BytesValue(group.Supersedes.Bytes())
	}
	return wcb.MapValue(map[string]wcb.Value{
		"index":          wcb.U64Value(group.Index),
		"row_start":      wcb.U64Value(group.RowStart),
		"row_end":        wcb.U64Value(group.RowEnd),
		"storage":        wcb.TextValue(group.Storage),
		"offset":         wcb.U64Value(group.Offset),
		"compressed_len": wcb.U64Value(group.CompressedLen),
		"group_root":     wcb.BytesValue(group.GroupRoot.Bytes()),
		"supersedes":     supersedes,
		"dependency":     dependency,
	})
}

func engramMetaValue(meta EngramMeta) wcb.Value {
	shape := make([]wcb.Value, 0, len(meta.TableShape))
	for _, dim := range meta.TableShape {
		shape = append(shape, wcb.U64Value(dim))
	}
	edges := make([]wcb.Value, 0, len(meta.RowGroupEdges))
	for _, edge := range meta.RowGroupEdges {
		edges = append(edges, wcb.U64Value(edge))
	}
	groups := make([]wcb.Value, 0, len(meta.Groups))
	for _, group := range meta.Groups {
		groups = append(groups, engramGroupValue(group))
	}
	return wcb.MapValue(map[string]wcb.Value{
		"schema":          wcb.TextValue(EngramSchema),
		"hash_function":   wcb.TextValue(meta.HashFunction),
		"vocab_size":      wcb.U64Value(meta.VocabSize),
		"table_shape":     wcb.ArrayValue(shape),
		"element_dtype":   wcb.TextValue(meta.Dtype),
		"precision":       engramPrecisionValue(meta.Precision),
		"row_group_edges": wcb.ArrayValue(edges),
		"group_count":     wcb.U64Value(uint64(len(meta.Groups))),
		"groups":          wcb.ArrayValue(groups),
		"payload_len":     wcb.U64Value(meta.PayloadLen),
		"payload_hash":    wcb.BytesValue(meta.PayloadHash.Bytes()),
	})
}

func engramSplitBody(body []byte) ([]byte, []byte, error) {
	if len(body) < 4 {
		return nil, nil, werrors.NewMalformed(
			"engram body is shorter than its 4-byte descriptor length")
	}
	length := int(uint32(body[0]) | uint32(body[1])<<8 | uint32(body[2])<<16 | uint32(body[3])<<24)
	end := 4 + length
	if end < 4 || end > len(body) {
		return nil, nil, werrors.NewMalformed("engram descriptor extends past the frame body")
	}
	return body[4:end], body[end:], nil
}

func engramExactKeys(m map[string]wcb.Value, keys ...string) error {
	allowed := make(map[string]bool, len(keys))
	for _, key := range keys {
		allowed[key] = true
		if _, ok := m[key]; !ok {
			return werrors.Malformedf("engram record is missing key %q", key)
		}
	}
	for key := range m {
		if !allowed[key] {
			return werrors.Malformedf("engram record has unknown key %q", key)
		}
	}
	return nil
}

func engramText(m map[string]wcb.Value, key string) (string, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.Text {
		return "", werrors.Malformedf("%q must be text", key)
	}
	return v.Str, nil
}

func engramU64(m map[string]wcb.Value, key string) (uint64, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.U64 {
		return 0, werrors.Malformedf("%q must be u64", key)
	}
	return v.U, nil
}

func engramBytes(m map[string]wcb.Value, key string) ([]byte, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.Bytes {
		return nil, werrors.Malformedf("%q must be bytes", key)
	}
	return v.Bytes, nil
}

func engramU64Array(m map[string]wcb.Value, key string) ([]uint64, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.Array {
		return nil, werrors.Malformedf("%q must be an array", key)
	}
	out := make([]uint64, 0, len(v.Array))
	for _, item := range v.Array {
		if item.Kind != wcb.U64 {
			return nil, werrors.Malformedf("%q elements must be u64", key)
		}
		out = append(out, item.U)
	}
	return out, nil
}

func engramHash32(b []byte, what string) (hash.Hash32, error) {
	if len(b) != hash.HashLen {
		return hash.Hash32{}, werrors.Malformedf("%s must be 32 bytes", what)
	}
	var raw [hash.HashLen]byte
	copy(raw[:], b)
	h := hash.Hash32(raw)
	if h == (hash.Hash32{}) {
		return hash.Hash32{}, werrors.Malformedf("%s must not be all zero", what)
	}
	return h, nil
}

// ParseEngramDescriptor parses a mode-20 descriptor from its canonical WCB
// bytes.
func ParseEngramDescriptor(descriptorBytes []byte, decodedLen, payloadLen uint64, lim limits.Limits) (EngramMeta, error) {
	value, err := wcb.Decode(descriptorBytes, lim)
	if err != nil {
		return EngramMeta{}, err
	}
	if value.Kind != wcb.Map {
		return EngramMeta{}, werrors.NewMalformed("engram descriptor must be a WCB map")
	}
	m := value.Map
	if err := engramExactKeys(m,
		"schema", "hash_function", "vocab_size", "table_shape", "element_dtype",
		"precision", "row_group_edges", "group_count", "groups", "payload_len",
		"payload_hash"); err != nil {
		return EngramMeta{}, err
	}
	schema, err := engramText(m, "schema")
	if err != nil {
		return EngramMeta{}, err
	}
	if schema != EngramSchema {
		return EngramMeta{}, werrors.Malformedf("unsupported engram schema %q", schema)
	}
	hashFunction, err := engramText(m, "hash_function")
	if err != nil {
		return EngramMeta{}, err
	}
	if err := engramValidateHashFunction(hashFunction); err != nil {
		return EngramMeta{}, err
	}
	tableShape, err := engramU64Array(m, "table_shape")
	if err != nil {
		return EngramMeta{}, err
	}
	dtype, err := engramText(m, "element_dtype")
	if err != nil {
		return EngramMeta{}, err
	}
	precisionValue, ok := m["precision"]
	if !ok || precisionValue.Kind != wcb.Map {
		return EngramMeta{}, werrors.NewMalformed("engram precision must be a map")
	}
	storageDtype, err := engramText(precisionValue.Map, "storage_dtype")
	if err != nil {
		return EngramMeta{}, err
	}
	if _, err := tensorElementSize(storageDtype); err != nil {
		return EngramMeta{}, werrors.UnsupportedFeaturef(
			"unknown engram storage_dtype %q", storageDtype)
	}
	var quantization *EngramQuantization
	if q, ok := precisionValue.Map["quantization"]; ok && q.Kind != wcb.Null {
		if q.Kind != wcb.Map {
			return EngramMeta{}, werrors.NewMalformed("engram quantization must be a map or null")
		}
		if err := engramExactKeys(q.Map, "scheme", "lossless"); err != nil {
			return EngramMeta{}, err
		}
		scheme, err := engramText(q.Map, "scheme")
		if err != nil {
			return EngramMeta{}, err
		}
		losslessValue, ok := q.Map["lossless"]
		if !ok || losslessValue.Kind != wcb.Bool {
			return EngramMeta{}, werrors.NewMalformed(
				"engram quantization lossless must be a bool")
		}
		quantization = &EngramQuantization{Scheme: scheme, Lossless: losslessValue.Bool}
	}
	for key := range precisionValue.Map {
		if key != "storage_dtype" && key != "quantization" {
			return EngramMeta{}, werrors.Malformedf(
				"engram precision has unknown key %q", key)
		}
	}
	precision := EngramPrecision{StorageDtype: storageDtype, Quantization: quantization}
	if err := engramValidatePrecision(dtype, precision); err != nil {
		return EngramMeta{}, err
	}
	edges, err := engramU64Array(m, "row_group_edges")
	if err != nil {
		return EngramMeta{}, err
	}
	rows, _, err := engramValidateEdges(tableShape, edges, lim)
	if err != nil {
		return EngramMeta{}, err
	}
	elementCount, err := tensor.ValidateShape(tableShape, lim)
	if err != nil {
		return EngramMeta{}, err
	}
	elementSize, err := tensorElementSize(dtype)
	if err != nil {
		return EngramMeta{}, err
	}
	if elementCount != 0 && elementSize > math.MaxUint64/elementCount {
		return EngramMeta{}, werrors.NewLimitExceeded("engram decoded bytes overflow u64")
	}
	expectedLen := elementCount * elementSize
	if decodedLen != expectedLen {
		return EngramMeta{}, werrors.Corruptf(
			"engram decoded_len is %d but the descriptor requires %d", decodedLen, expectedLen)
	}
	vocabSize, err := engramU64(m, "vocab_size")
	if err != nil {
		return EngramMeta{}, err
	}
	if vocabSize == 0 || vocabSize > rows {
		return EngramMeta{}, werrors.Malformedf(
			"engram vocab_size %d must be in 1..=%d", vocabSize, rows)
	}
	groupCount, err := engramU64(m, "group_count")
	if err != nil {
		return EngramMeta{}, err
	}
	if groupCount != uint64(len(edges)-1) {
		return EngramMeta{}, werrors.Malformedf(
			"engram group_count %d does not match %d edges", groupCount, len(edges)-1)
	}
	declaredPayloadLen, err := engramU64(m, "payload_len")
	if err != nil {
		return EngramMeta{}, err
	}
	if declaredPayloadLen != payloadLen {
		return EngramMeta{}, werrors.Malformedf(
			"engram payload is %d bytes but the descriptor declares %d",
			payloadLen, declaredPayloadLen)
	}
	payloadHashBytes, err := engramBytes(m, "payload_hash")
	if err != nil {
		return EngramMeta{}, err
	}
	payloadHash, err := engramHash32(payloadHashBytes, "engram payload_hash")
	if err != nil {
		return EngramMeta{}, err
	}
	groupValue, ok := m["groups"]
	if !ok || groupValue.Kind != wcb.Array {
		return EngramMeta{}, werrors.NewMalformed("engram groups must be an array")
	}
	if uint64(len(groupValue.Array)) != groupCount {
		return EngramMeta{}, werrors.Malformedf(
			"engram has %d group entries but group_count is %d",
			len(groupValue.Array), groupCount)
	}
	groups := make([]EngramGroup, 0, len(groupValue.Array))
	var nextOffset uint64
	for position, item := range groupValue.Array {
		if item.Kind != wcb.Map {
			return EngramMeta{}, werrors.NewMalformed("engram group must be a WCB map")
		}
		entry := item.Map
		if err := engramExactKeys(entry,
			"index", "row_start", "row_end", "storage", "offset", "compressed_len",
			"group_root", "supersedes", "dependency"); err != nil {
			return EngramMeta{}, err
		}
		index, err := engramU64(entry, "index")
		if err != nil {
			return EngramMeta{}, err
		}
		if index != uint64(position) {
			return EngramMeta{}, werrors.Malformedf(
				"engram group entry %d declares index %d; entries must be in canonical order",
				position, index)
		}
		rowStart, err := engramU64(entry, "row_start")
		if err != nil {
			return EngramMeta{}, err
		}
		rowEnd, err := engramU64(entry, "row_end")
		if err != nil {
			return EngramMeta{}, err
		}
		if rowStart != edges[position] || rowEnd != edges[position+1] {
			return EngramMeta{}, werrors.Malformedf(
				"engram group %d rows [%d,%d) do not match edges [%d,%d)",
				position, rowStart, rowEnd, edges[position], edges[position+1])
		}
		storage, err := engramText(entry, "storage")
		if err != nil {
			return EngramMeta{}, err
		}
		offset, err := engramU64(entry, "offset")
		if err != nil {
			return EngramMeta{}, err
		}
		compressedLen, err := engramU64(entry, "compressed_len")
		if err != nil {
			return EngramMeta{}, err
		}
		var dependency uint64
		hasDependency := false
		switch dep := entry["dependency"]; dep.Kind {
		case wcb.Null:
		case wcb.U64:
			dependency, hasDependency = dep.U, true
		default:
			return EngramMeta{}, werrors.NewMalformed(
				"engram group dependency must be u64 or null")
		}
		switch storage {
		case EngramStorageInline:
			if hasDependency {
				return EngramMeta{}, werrors.Malformedf(
					"engram inline group %d must not name a dependency", position)
			}
			if compressedLen == 0 {
				return EngramMeta{}, werrors.Malformedf(
					"engram inline group %d has an empty stored range", position)
			}
			if offset != nextOffset {
				return EngramMeta{}, werrors.Malformedf(
					"engram group %d offset %d is not the contiguous offset %d",
					position, offset, nextOffset)
			}
			if compressedLen > math.MaxUint64-nextOffset {
				return EngramMeta{}, werrors.NewLimitExceeded("engram group offset overflows u64")
			}
			nextOffset += compressedLen
		case EngramStorageReference:
			if offset != 0 || compressedLen != 0 {
				return EngramMeta{}, werrors.Malformedf(
					"engram reference group %d must have offset 0 and compressed_len 0", position)
			}
			if !hasDependency {
				return EngramMeta{}, werrors.Malformedf(
					"engram reference group %d must name a dependency", position)
			}
		default:
			return EngramMeta{}, werrors.Malformedf(
				"engram group %d has unknown storage %q", position, storage)
		}
		groupRootBytes, err := engramBytes(entry, "group_root")
		if err != nil {
			return EngramMeta{}, err
		}
		groupRoot, err := engramHash32(groupRootBytes, "engram group_root")
		if err != nil {
			return EngramMeta{}, err
		}
		var supersedes *hash.Hash32
		switch sup := entry["supersedes"]; sup.Kind {
		case wcb.Null:
		case wcb.Bytes:
			h, err := engramHash32(sup.Bytes, "engram supersedes")
			if err != nil {
				return EngramMeta{}, err
			}
			supersedes = &h
		default:
			return EngramMeta{}, werrors.NewMalformed(
				"engram group supersedes must be bytes32 or null")
		}
		groups = append(groups, EngramGroup{
			Index:         index,
			RowStart:      rowStart,
			RowEnd:        rowEnd,
			Storage:       storage,
			Offset:        offset,
			CompressedLen: compressedLen,
			Dependency:    dependency,
			HasDependency: hasDependency,
			GroupRoot:     groupRoot,
			Supersedes:    supersedes,
		})
	}
	if nextOffset != declaredPayloadLen {
		return EngramMeta{}, werrors.Malformedf(
			"engram inline group ranges cover %d bytes but payload_len is %d",
			nextOffset, declaredPayloadLen)
	}
	return EngramMeta{
		HashFunction:  hashFunction,
		VocabSize:     vocabSize,
		TableShape:    tableShape,
		Dtype:         dtype,
		Precision:     precision,
		RowGroupEdges: edges,
		PayloadLen:    declaredPayloadLen,
		PayloadHash:   payloadHash,
		Groups:        groups,
	}, nil
}

// ParseEngram parses a mode-20 frame and returns its descriptor and payload.
func ParseEngram(f *frame.Frame, lim limits.Limits) (EngramDescriptor, error) {
	if f.Header.Mode != frame.ModeEngramShard {
		return EngramDescriptor{}, werrors.Malformedf(
			"frame mode %s is not engram_shard", f.Header.Mode)
	}
	descriptorBytes, payload, err := engramSplitBody(f.Body)
	if err != nil {
		return EngramDescriptor{}, err
	}
	meta, err := ParseEngramDescriptor(descriptorBytes, f.Header.DecodedLen, uint64(len(payload)), lim)
	if err != nil {
		return EngramDescriptor{}, err
	}
	decodedLen, err := meta.DecodedLen()
	if err != nil {
		return EngramDescriptor{}, err
	}
	for index, dep := range f.Dependencies {
		if dep.DepDomain != frame.DepDomainPayload {
			return EngramDescriptor{}, werrors.Malformedf(
				"engram dependency %d must use dep_domain 0 (payload)", index)
		}
		if dep.DecodedLen != decodedLen {
			return EngramDescriptor{}, werrors.Malformedf(
				"engram dependency %d declares %d bytes but the table is %d",
				index, dep.DecodedLen, decodedLen)
		}
	}
	for _, group := range meta.Groups {
		if group.Storage == EngramStorageReference && group.Dependency >= uint64(len(f.Dependencies)) {
			return EngramDescriptor{}, werrors.Malformedf(
				"engram group %d references dependency %d but the frame has %d dependencies",
				group.Index, group.Dependency, len(f.Dependencies))
		}
	}
	if hash.Domain(hash.DomainEngramPayload, payload) != meta.PayloadHash {
		return EngramDescriptor{}, werrors.NewCorrupt(
			"engram payload hash does not match the descriptor")
	}
	return EngramDescriptor{Meta: meta, Payload: payload}, nil
}

// EngramMetadata returns a mode-20 frame's descriptor.
func EngramMetadata(f *frame.Frame, lim limits.Limits) (EngramMeta, error) {
	descriptor, err := ParseEngram(f, lim)
	if err != nil {
		return EngramMeta{}, err
	}
	return descriptor.Meta, nil
}

func engramGroupByteLen(meta EngramMeta, group EngramGroup) (uint64, error) {
	size, err := tensorElementSize(meta.Dtype)
	if err != nil {
		return 0, err
	}
	rows := group.RowEnd - group.RowStart
	if rows != 0 && meta.Dim() > math.MaxUint64/rows {
		return 0, werrors.NewLimitExceeded("engram group byte length overflows u64")
	}
	elements := rows * meta.Dim()
	if elements != 0 && size > math.MaxUint64/elements {
		return 0, werrors.NewLimitExceeded("engram group byte length overflows u64")
	}
	return elements * size, nil
}

func decodeEngramInlineGroup(meta EngramMeta, group EngramGroup, payload []byte, lim limits.Limits) ([]byte, error) {
	if group.Offset > uint64(len(payload)) || group.CompressedLen > uint64(len(payload))-group.Offset {
		return nil, werrors.NewCorrupt("engram group range is outside the payload")
	}
	stored := payload[group.Offset : group.Offset+group.CompressedLen]
	elementLen, err := engramGroupByteLen(meta, group)
	if err != nil {
		return nil, err
	}
	elementSize, err := tensorElementSize(meta.Dtype)
	if err != nil {
		return nil, err
	}
	bytes, err := decodeEngramGroupPlanes(stored, int(elementSize), int(elementLen))
	if err != nil {
		return nil, err
	}
	root, err := EngramGroupRoot(
		meta.Dtype, meta.Precision.StorageDtype, group.Index, group.RowStart, group.RowEnd, bytes, lim)
	if err != nil {
		return nil, err
	}
	if root != group.GroupRoot {
		return nil, werrors.Corruptf(
			"engram group %d root %s does not match the descriptor's %s",
			group.Index, root, group.GroupRoot)
	}
	return bytes, nil
}

type foundEngramGroup struct {
	meta  EngramMeta
	group EngramGroup
	bytes []byte
}

func engramCompatibleReference(a EngramMeta, ga EngramGroup, b EngramMeta, gb EngramGroup) bool {
	return ga.RowStart == gb.RowStart && ga.RowEnd == gb.RowEnd &&
		a.Dtype == b.Dtype && a.Precision.StorageDtype == b.Precision.StorageDtype
}

func engramCompatibleSupersession(a, b EngramMeta) bool {
	return a.Dtype == b.Dtype && a.Precision.StorageDtype == b.Precision.StorageDtype
}

func findEngramGroup(ctx context.Context, f *frame.Frame, root hash.Hash32, source dependency.FrameSource, lim limits.Limits, visiting map[hash.Hash32]bool, depth uint64) (*foundEngramGroup, error) {
	if depth == 0 {
		return nil, nil
	}
	descriptor, err := ParseEngram(f, lim)
	if err != nil {
		return nil, err
	}
	for _, group := range descriptor.Meta.Groups {
		if group.GroupRoot != root {
			continue
		}
		var bytes []byte
		switch group.Storage {
		case EngramStorageInline:
			bytes, err = decodeEngramInlineGroup(descriptor.Meta, group, descriptor.Payload, lim)
			if err != nil {
				return nil, err
			}
		case EngramStorageReference:
			if group.Dependency >= uint64(len(f.Dependencies)) {
				return nil, werrors.NewCorrupt("engram reference dependency index is out of range")
			}
			dep := f.Dependencies[group.Dependency]
			if visiting[dep.Hash] {
				return nil, werrors.NewDependencyCycle(
					"engram dependency cycle while resolving a referenced group")
			}
			visiting[dep.Hash] = true
			var nested []byte
			var ok bool
			if source != nil {
				nested, ok, err = source.GetFrame(ctx, dep.Hash)
				if err != nil {
					delete(visiting, dep.Hash)
					return nil, err
				}
			}
			var found *foundEngramGroup
			if ok {
				parsed, parseErr := frame.Parse(nested, lim)
				if parseErr != nil {
					delete(visiting, dep.Hash)
					return nil, parseErr
				}
				found, err = findEngramGroup(ctx, parsed, root, source, lim, visiting, depth-1)
				if err != nil {
					delete(visiting, dep.Hash)
					return nil, err
				}
			}
			delete(visiting, dep.Hash)
			if found == nil {
				return nil, werrors.DependencyMissingf(
					"engram referenced group %s is absent from dependency %s", root, dep.Hash)
			}
			bytes = found.bytes
		}
		return &foundEngramGroup{meta: descriptor.Meta, group: group, bytes: bytes}, nil
	}
	for _, dep := range f.Dependencies {
		if visiting[dep.Hash] {
			return nil, werrors.NewDependencyCycle(
				"engram dependency cycle while resolving a group")
		}
		if source == nil {
			continue
		}
		visiting[dep.Hash] = true
		nested, ok, err := source.GetFrame(ctx, dep.Hash)
		if err != nil {
			delete(visiting, dep.Hash)
			return nil, err
		}
		var found *foundEngramGroup
		if ok {
			parsed, parseErr := frame.Parse(nested, lim)
			if parseErr != nil {
				delete(visiting, dep.Hash)
				return nil, parseErr
			}
			found, err = findEngramGroup(ctx, parsed, root, source, lim, visiting, depth-1)
			if err != nil {
				delete(visiting, dep.Hash)
				return nil, err
			}
		}
		delete(visiting, dep.Hash)
		if found != nil {
			return found, nil
		}
	}
	return nil, nil
}

// ReadEngramGroups reads explicit group indices. It returns the concatenated
// canonical group bytes in ascending index order and the touched indices.
// Only touched groups are decoded; the payload hash, trailer and whole-table
// content hash are not checked on this path.
func ReadEngramGroups(ctx context.Context, f *frame.Frame, descriptor EngramDescriptor, indices []uint32, source dependency.FrameSource, lim limits.Limits) ([]byte, []uint32, error) {
	seen := map[uint32]bool{}
	unique := make([]uint32, 0, len(indices))
	for _, index := range indices {
		if int(index) >= len(descriptor.Meta.Groups) {
			return nil, nil, werrors.Malformedf(
				"engram group index %d is outside the descriptor", index)
		}
		if !seen[index] {
			seen[index] = true
			unique = append(unique, index)
		}
	}
	sortU32(unique)
	out := make([]byte, 0)
	for _, index := range unique {
		group := descriptor.Meta.Groups[index]
		switch group.Storage {
		case EngramStorageInline:
			bytes, err := decodeEngramInlineGroup(descriptor.Meta, group, descriptor.Payload, lim)
			if err != nil {
				return nil, nil, err
			}
			out = append(out, bytes...)
		case EngramStorageReference:
			found, err := findEngramGroup(ctx, f, group.GroupRoot, source, lim, map[hash.Hash32]bool{}, lim.MaxDependencyDepth)
			if err != nil {
				return nil, nil, err
			}
			if found == nil {
				return nil, nil, werrors.DependencyMissingf(
					"engram referenced group %d (%s) is absent", group.Index, group.GroupRoot)
			}
			if !engramCompatibleReference(descriptor.Meta, group, found.meta, found.group) {
				return nil, nil, werrors.Corruptf(
					"engram referenced group %d is incompatible with its source group", group.Index)
			}
			out = append(out, found.bytes...)
		}
	}
	return out, unique, nil
}

func sortU32(values []uint32) {
	for i := 1; i < len(values); i++ {
		for j := i; j > 0 && values[j] < values[j-1]; j-- {
			values[j], values[j-1] = values[j-1], values[j]
		}
	}
}

// EngramRegionGroups returns the groups whose row intervals intersect
// [rowStart, rowEnd).
func EngramRegionGroups(meta EngramMeta, rowStart, rowEnd uint64) ([]uint32, error) {
	if rowStart > rowEnd {
		return nil, werrors.NonCanonicalf(
			"engram row region start %d is after end %d", rowStart, rowEnd)
	}
	if rowEnd > meta.Rows() {
		return nil, werrors.NonCanonicalf(
			"engram row region end %d exceeds %d rows", rowEnd, meta.Rows())
	}
	out := []uint32{}
	if rowStart == rowEnd {
		return out, nil
	}
	for _, group := range meta.Groups {
		if group.RowStart < rowEnd && rowStart < group.RowEnd {
			out = append(out, uint32(group.Index))
		}
	}
	return out, nil
}

// ReadEngramRegion reads a row region, decoding only intersecting groups.
func ReadEngramRegion(ctx context.Context, f *frame.Frame, descriptor EngramDescriptor, rowStart, rowEnd uint64, source dependency.FrameSource, lim limits.Limits) ([]byte, []uint32, error) {
	indices, err := EngramRegionGroups(descriptor.Meta, rowStart, rowEnd)
	if err != nil {
		return nil, nil, err
	}
	if len(indices) == 0 {
		return []byte{}, []uint32{}, nil
	}
	bytes, touched, err := ReadEngramGroups(ctx, f, descriptor, indices, source, lim)
	if err != nil {
		return nil, nil, err
	}
	size, err := tensorElementSize(descriptor.Meta.Dtype)
	if err != nil {
		return nil, nil, err
	}
	if descriptor.Meta.Dim() != 0 && size > math.MaxUint64/descriptor.Meta.Dim() {
		return nil, nil, werrors.NewLimitExceeded("engram row byte length overflows u64")
	}
	rowBytes := descriptor.Meta.Dim() * size
	first := descriptor.Meta.Groups[indices[0]]
	last := descriptor.Meta.Groups[indices[len(indices)-1]]
	lead := (rowStart - first.RowStart) * rowBytes
	tail := (last.RowEnd - rowEnd) * rowBytes
	if tail > uint64(len(bytes)) || lead > uint64(len(bytes))-tail {
		return nil, nil, werrors.NewCorrupt(
			"engram region trim exceeds the decoded group bytes")
	}
	return bytes[lead : uint64(len(bytes))-tail], touched, nil
}

func verifyEngramSupersession(ctx context.Context, meta EngramMeta, f *frame.Frame, source dependency.FrameSource, lim limits.Limits) error {
	for _, group := range meta.Groups {
		seen := map[hash.Hash32]bool{group.GroupRoot: true}
		current := group.Supersedes
		for current != nil {
			root := *current
			if seen[root] {
				return werrors.Corruptf("engram supersession cycle at group %d", group.Index)
			}
			seen[root] = true
			inFrame := false
			for _, parent := range meta.Groups {
				if parent.GroupRoot != root {
					continue
				}
				if !engramCompatibleSupersession(meta, meta) {
					return werrors.Corruptf(
						"engram supersession for group %d names an incompatible in-frame group",
						group.Index)
				}
				current = parent.Supersedes
				inFrame = true
				break
			}
			if inFrame {
				continue
			}
			found, err := findEngramGroup(ctx, f, root, source, lim, map[hash.Hash32]bool{}, lim.MaxDependencyDepth)
			if err != nil {
				return err
			}
			if found == nil {
				return werrors.Corruptf(
					"engram supersession root %s does not resolve to a group root", root)
			}
			if !engramCompatibleSupersession(meta, found.meta) {
				return werrors.Corruptf(
					"engram supersession for group %d names an incompatible group", group.Index)
			}
			break
		}
	}
	return nil
}

type emptyEngramFrameSource struct{}

func (emptyEngramFrameSource) GetFrame(context.Context, hash.Hash32) ([]byte, bool, error) {
	return nil, false, nil
}

// DecodeEngram decodes a complete mode-20 frame. Every group root, every
// supersession edge, the payload hash, the exact length and the whole-table
// content hash are verified.
func DecodeEngram(ctx context.Context, f *frame.Frame, source dependency.FrameSource, lim limits.Limits) ([]byte, error) {
	descriptor, err := ParseEngram(f, lim)
	if err != nil {
		return nil, err
	}
	meta := descriptor.Meta
	total, err := meta.DecodedLen()
	if err != nil {
		return nil, err
	}
	if uint64(int(total)) != total {
		return nil, werrors.NewLimitExceeded("engram decoded bytes do not fit int")
	}
	out := make([]byte, int(total))
	size, err := tensorElementSize(meta.Dtype)
	if err != nil {
		return nil, err
	}
	if meta.Dim() != 0 && size > math.MaxUint64/meta.Dim() {
		return nil, werrors.NewLimitExceeded("engram row byte length overflows u64")
	}
	rowBytes := meta.Dim() * size
	for _, group := range meta.Groups {
		var bytes []byte
		switch group.Storage {
		case EngramStorageInline:
			bytes, err = decodeEngramInlineGroup(meta, group, descriptor.Payload, lim)
			if err != nil {
				return nil, err
			}
		case EngramStorageReference:
			found, findErr := findEngramGroup(ctx, f, group.GroupRoot, source, lim, map[hash.Hash32]bool{}, lim.MaxDependencyDepth)
			if findErr != nil {
				return nil, findErr
			}
			if found == nil {
				return nil, werrors.DependencyMissingf(
					"engram referenced group %d (%s) is absent", group.Index, group.GroupRoot)
			}
			if !engramCompatibleReference(meta, group, found.meta, found.group) {
				return nil, werrors.Corruptf(
					"engram referenced group %d is incompatible with its source group", group.Index)
			}
			bytes = found.bytes
		}
		start := group.RowStart * rowBytes
		end := start + uint64(len(bytes))
		if end < start || end > uint64(len(out)) {
			return nil, werrors.NewCorrupt(
				"engram group destination is outside the assembled table")
		}
		copy(out[start:end], bytes)
	}
	if err := verifyEngramSupersession(ctx, meta, f, source, lim); err != nil {
		return nil, err
	}
	if hash.Domain(hash.DomainChunk, out) != f.Header.ContentHash {
		return nil, werrors.NewCorrupt("engram whole-table content hash mismatch")
	}
	return out, nil
}

// DecodeEngramWithoutDependencies decodes a frame that has no references.
// Any reference (or external supersession edge) fails closed.
func DecodeEngramWithoutDependencies(ctx context.Context, f *frame.Frame, lim limits.Limits) ([]byte, error) {
	return DecodeEngram(ctx, f, emptyEngramFrameSource{}, lim)
}

// DecodeEngramFrame is the generic mode-dispatch entry point.
func DecodeEngramFrame(ctx context.Context, f *frame.Frame, resolver dependency.Resolver, lim limits.Limits) ([]byte, error) {
	source, _ := resolver.(dependency.FrameSource)
	return DecodeEngram(ctx, f, source, lim)
}
