// Package tensor implements the logical tensor layer of WEFT-TENSOR v1:
// layout-independent tensor identity, physical layout descriptors, region
// addressing, and shard-container reassembly.
//
// The logical identity is
//
//	values_hash       = H("weft/tensor/values/v1",  row-major element bytes)
//	LogicalTensorRoot = H("weft/tensor/logical/v1",
//	                      WCB({schema:"weft.tensor.logical.v1", dtype, shape,
//	                           element_count, values_hash}))
//
// and it deliberately ignores sharding, blocking, strides, and storage layout.
//
// This package is additive: it does not change mode-6 TENSOR frames, whose
// `layout` field remains frozen to `row_major` (WEFT-AI-CODEC §2).
package tensor

import (
	"encoding/binary"
	"math"

	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// Canonical schema strings.
const (
	LogicalSchema = "weft.tensor.logical.v1"
	LayoutSchema  = "weft.tensor.layout.v1"
	ShardSchema   = "weft.tensor.shard.v1"
)

// Domain strings.
const (
	DomainTensorLogical      = "weft/tensor/logical/v1"
	DomainTensorValues       = "weft/tensor/values/v1"
	DomainTensorShardPayload = "weft/tensor/shard-payload/v1"
)

// DtypeSize parses a dtype name and returns its element size. Unknown dtypes
// are UnsupportedFeature in the tensor-identity layer.
func DtypeSize(dtype string) (uint64, error) {
	size, err := modes.TensorElementSize(dtype)
	if err != nil {
		return 0, werrors.UnsupportedFeaturef("unknown tensor dtype %q", dtype)
	}
	return size, nil
}

// ValidateShape checks a logical shape and returns its element count.
// Rank 0, zero dimensions, overflow, and limit violations are rejected.
func ValidateShape(shape []uint64, lim limits.Limits) (uint64, error) {
	if len(shape) == 0 {
		return 0, werrors.NewNonCanonical(
			"tensor shape must have rank >= 1 (rank 0 is not a v1 logical tensor)")
	}
	if uint64(len(shape)) > lim.MaxTensorRank {
		return 0, werrors.LimitExceededf(
			"tensor rank %d exceeds limit %d", len(shape), lim.MaxTensorRank)
	}
	count := uint64(1)
	for axis, dim := range shape {
		if dim == 0 {
			return 0, werrors.NonCanonicalf("tensor shape dimension %d is zero", axis)
		}
		if count > math.MaxUint64/dim {
			return 0, werrors.NewLimitExceeded("tensor shape product overflows u64")
		}
		count *= dim
	}
	if count > lim.MaxTensorElements {
		return 0, werrors.LimitExceededf(
			"tensor element count %d exceeds limit %d", count, lim.MaxTensorElements)
	}
	return count, nil
}

// PayloadLen is the checked element byte length.
func PayloadLen(elementCount uint64, dtype string) (uint64, error) {
	size, err := DtypeSize(dtype)
	if err != nil {
		return 0, err
	}
	if elementCount != 0 && size > math.MaxUint64/elementCount {
		return 0, werrors.NewLimitExceeded("tensor payload byte count overflows u64")
	}
	return elementCount * size, nil
}

// CheckPayload verifies that bytes is exactly the payload for shape x dtype.
// A byte-length mismatch is Malformed; a payload above MaxDecodedBytes is
// LimitExceeded before any allocation.
func CheckPayload(dtype string, shape []uint64, bytes []byte, lim limits.Limits) (uint64, error) {
	count, err := ValidateShape(shape, lim)
	if err != nil {
		return 0, err
	}
	expected, err := PayloadLen(count, dtype)
	if err != nil {
		return 0, err
	}
	if err := limits.CheckLen(expected, lim.MaxDecodedBytes, "tensor payload"); err != nil {
		return 0, err
	}
	if expected != uint64(len(bytes)) {
		return 0, werrors.Malformedf(
			"tensor payload is %d bytes but shape %v x %s requires %d",
			len(bytes), shape, dtype, expected)
	}
	return count, nil
}

// ValuesHash is H("weft/tensor/values/v1", rowMajor).
func ValuesHash(rowMajor []byte) hash.Hash32 {
	return hash.Domain(DomainTensorValues, rowMajor)
}

// LogicalRecord is the canonical layout-independent identity record.
type LogicalRecord struct {
	Dtype        string
	Shape        []uint64
	ElementCount uint64
	ValuesHash   hash.Hash32
}

// NewLogicalRecord builds a record from canonical row-major element bytes.
func NewLogicalRecord(dtype string, shape []uint64, rowMajor []byte, lim limits.Limits) (LogicalRecord, error) {
	count, err := CheckPayload(dtype, shape, rowMajor, lim)
	if err != nil {
		return LogicalRecord{}, err
	}
	return LogicalRecord{
		Dtype:        dtype,
		Shape:        append([]uint64(nil), shape...),
		ElementCount: count,
		ValuesHash:   ValuesHash(rowMajor),
	}, nil
}

// Equal reports structural equality (slices are compared element-wise).
func (r LogicalRecord) Equal(other LogicalRecord) bool {
	return r.Dtype == other.Dtype &&
		r.ElementCount == other.ElementCount &&
		r.ValuesHash == other.ValuesHash &&
		u64SliceEqual(r.Shape, other.Shape)
}

// Validate checks every record invariant.
func (r LogicalRecord) Validate(lim limits.Limits) error {
	count, err := ValidateShape(r.Shape, lim)
	if err != nil {
		return err
	}
	if r.ElementCount != count {
		return werrors.NonCanonicalf(
			"tensor element_count %d does not match shape product %d", r.ElementCount, count)
	}
	_, err = PayloadLen(count, r.Dtype)
	return err
}

// WCBValue renders the canonical WCB map.
func (r LogicalRecord) WCBValue() wcb.Value {
	shape := make([]wcb.Value, 0, len(r.Shape))
	for _, dim := range r.Shape {
		shape = append(shape, wcb.U64Value(dim))
	}
	return wcb.MapValue(map[string]wcb.Value{
		"schema":        wcb.TextValue(LogicalSchema),
		"dtype":         wcb.TextValue(r.Dtype),
		"shape":         wcb.ArrayValue(shape),
		"element_count": wcb.U64Value(r.ElementCount),
		"values_hash":   wcb.BytesValue(r.ValuesHash.Bytes()),
	})
}

// CanonicalBytes renders canonical WCB bytes.
func (r LogicalRecord) CanonicalBytes(lim limits.Limits) ([]byte, error) {
	if err := r.Validate(lim); err != nil {
		return nil, err
	}
	return wcb.Encode(r.WCBValue(), lim)
}

// Root is H("weft/tensor/logical/v1", canonical bytes).
func (r LogicalRecord) Root(lim limits.Limits) (hash.Hash32, error) {
	canonical, err := r.CanonicalBytes(lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(DomainTensorLogical, canonical), nil
}

// ParseLogicalRecord decodes canonical bytes, rejecting non-canonical forms.
func ParseLogicalRecord(bytes []byte, lim limits.Limits) (LogicalRecord, error) {
	value, err := wcb.Decode(bytes, lim)
	if err != nil {
		return LogicalRecord{}, err
	}
	if value.Kind != wcb.Map {
		return LogicalRecord{}, werrors.NewNonCanonical("logical tensor record must be a WCB map")
	}
	m := value.Map
	if err := checkAllowedKeys(m, []string{
		"schema", "dtype", "shape", "element_count", "values_hash",
	}, "logical tensor record"); err != nil {
		return LogicalRecord{}, err
	}
	schema, err := textField(m, "schema", "logical tensor record")
	if err != nil {
		return LogicalRecord{}, err
	}
	if schema != LogicalSchema {
		return LogicalRecord{}, werrors.UnknownSchemaf(
			"unknown logical tensor schema %q", schema)
	}
	dtype, err := textField(m, "dtype", "logical tensor record")
	if err != nil {
		return LogicalRecord{}, err
	}
	if _, err := DtypeSize(dtype); err != nil {
		return LogicalRecord{}, err
	}
	shape, err := u64ArrayField(m, "shape", "logical tensor record")
	if err != nil {
		return LogicalRecord{}, err
	}
	elementCount, err := u64Field(m, "element_count", "logical tensor record")
	if err != nil {
		return LogicalRecord{}, err
	}
	valuesHash, err := hashField(m, "values_hash", "logical tensor record")
	if err != nil {
		return LogicalRecord{}, err
	}
	record := LogicalRecord{
		Dtype:        dtype,
		Shape:        shape,
		ElementCount: elementCount,
		ValuesHash:   valuesHash,
	}
	if err := record.Validate(lim); err != nil {
		return LogicalRecord{}, err
	}
	reencoded, err := wcb.Encode(record.WCBValue(), lim)
	if err != nil {
		return LogicalRecord{}, err
	}
	if !bytesEqual(reencoded, bytes) {
		return LogicalRecord{}, werrors.NewNonCanonical(
			"logical tensor bytes are not in canonical form")
	}
	return record, nil
}

// LogicalRoot computes the logical root from canonical row-major bytes.
func LogicalRoot(dtype string, shape []uint64, rowMajor []byte, lim limits.Limits) (hash.Hash32, error) {
	record, err := NewLogicalRecord(dtype, shape, rowMajor, lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	return record.Root(lim)
}

// LayoutKind enumerates the physical layouts.
type LayoutKind uint8

const (
	RowMajor LayoutKind = iota
	ColumnMajor
	Blocked
	Sharded
)

// Layout is the layout-specific part of a descriptor.
type Layout struct {
	Kind       LayoutKind
	BlockShape []uint64
	Axis       uint32
	Shards     uint32
	Index      uint32
}

// Equal reports structural equality.
func (l Layout) Equal(other Layout) bool {
	return l.Kind == other.Kind &&
		l.Axis == other.Axis &&
		l.Shards == other.Shards &&
		l.Index == other.Index &&
		u64SliceEqual(l.BlockShape, other.BlockShape)
}

// Name is the canonical layout name.
func (l Layout) Name() string {
	switch l.Kind {
	case RowMajor:
		return "row-major"
	case ColumnMajor:
		return "column-major"
	case Blocked:
		return "blocked"
	case Sharded:
		return "sharded"
	default:
		return "unknown"
	}
}

// Descriptor is the canonical weft.tensor.layout.v1 record.
type Descriptor struct {
	Dtype  string
	Shape  []uint64
	Layout Layout
}

// RowMajorDescriptor builds a row-major descriptor.
func RowMajorDescriptor(dtype string, shape []uint64) Descriptor {
	return Descriptor{Dtype: dtype, Shape: append([]uint64(nil), shape...), Layout: Layout{Kind: RowMajor}}
}

// ColumnMajorDescriptor builds a column-major descriptor.
func ColumnMajorDescriptor(dtype string, shape []uint64) Descriptor {
	return Descriptor{Dtype: dtype, Shape: append([]uint64(nil), shape...), Layout: Layout{Kind: ColumnMajor}}
}

// BlockedDescriptor builds a blocked descriptor.
func BlockedDescriptor(dtype string, shape, blockShape []uint64) Descriptor {
	return Descriptor{
		Dtype:  dtype,
		Shape:  append([]uint64(nil), shape...),
		Layout: Layout{Kind: Blocked, BlockShape: append([]uint64(nil), blockShape...)},
	}
}

// ShardedDescriptor builds a sharded descriptor.
func ShardedDescriptor(dtype string, shape []uint64, axis, shards, index uint32) Descriptor {
	return Descriptor{
		Dtype:  dtype,
		Shape:  append([]uint64(nil), shape...),
		Layout: Layout{Kind: Sharded, Axis: axis, Shards: shards, Index: index},
	}
}

// Equal reports structural equality.
func (d Descriptor) Equal(other Descriptor) bool {
	return d.Dtype == other.Dtype &&
		u64SliceEqual(d.Shape, other.Shape) &&
		d.Layout.Equal(other.Layout)
}

// Validate checks shape and layout invariants.
func (d Descriptor) Validate(lim limits.Limits) error {
	if len(d.Shape) == 0 {
		if d.Layout.Kind == ColumnMajor {
			return werrors.NewNonCanonical(
				"column-major cannot express a rank-0 shape (a scalar has no column order)")
		}
		return werrors.NewNonCanonical(
			"tensor shape must have rank >= 1 (rank 0 is not a v1 tensor)")
	}
	if _, err := ValidateShape(d.Shape, lim); err != nil {
		return err
	}
	if _, err := DtypeSize(d.Dtype); err != nil {
		return err
	}
	switch d.Layout.Kind {
	case RowMajor, ColumnMajor:
		return nil
	case Blocked:
		if len(d.Layout.BlockShape) != len(d.Shape) {
			return werrors.NonCanonicalf(
				"blocked block_shape rank %d does not equal shape rank %d",
				len(d.Layout.BlockShape), len(d.Shape))
		}
		for axis, block := range d.Layout.BlockShape {
			if block == 0 {
				return werrors.NonCanonicalf(
					"blocked block_shape dimension %d is zero", axis)
			}
		}
		return nil
	case Sharded:
		if err := validateShardParameters(d.Shape, d.Layout.Axis, d.Layout.Shards); err != nil {
			return err
		}
		if d.Layout.Index >= d.Layout.Shards {
			return werrors.NonCanonicalf(
				"shard index %d is not below shard count %d", d.Layout.Index, d.Layout.Shards)
		}
		return nil
	default:
		return werrors.UnsupportedFeaturef("unknown tensor layout kind %d", d.Layout.Kind)
	}
}

// WCBValue renders the canonical layout record.
func (d Descriptor) WCBValue() wcb.Value {
	return wcb.MapValue(descriptorMap(d))
}

func descriptorMap(d Descriptor) map[string]wcb.Value {
	shape := make([]wcb.Value, 0, len(d.Shape))
	for _, dim := range d.Shape {
		shape = append(shape, wcb.U64Value(dim))
	}
	m := map[string]wcb.Value{
		"schema": wcb.TextValue(LayoutSchema),
		"dtype":  wcb.TextValue(d.Dtype),
		"shape":  wcb.ArrayValue(shape),
		"layout": wcb.TextValue(d.Layout.Name()),
	}
	switch d.Layout.Kind {
	case Blocked:
		blocks := make([]wcb.Value, 0, len(d.Layout.BlockShape))
		for _, dim := range d.Layout.BlockShape {
			blocks = append(blocks, wcb.U64Value(dim))
		}
		m["block_shape"] = wcb.ArrayValue(blocks)
	case Sharded:
		m["axis"] = wcb.U64Value(uint64(d.Layout.Axis))
		m["shards"] = wcb.U64Value(uint64(d.Layout.Shards))
		m["index"] = wcb.U64Value(uint64(d.Layout.Index))
	}
	return m
}

// CanonicalBytes renders canonical WCB bytes.
func (d Descriptor) CanonicalBytes(lim limits.Limits) ([]byte, error) {
	if err := d.Validate(lim); err != nil {
		return nil, err
	}
	return wcb.Encode(d.WCBValue(), lim)
}

// ParseDescriptor decodes canonical descriptor bytes.
func ParseDescriptor(bytes []byte, lim limits.Limits) (Descriptor, error) {
	value, err := wcb.Decode(bytes, lim)
	if err != nil {
		return Descriptor{}, err
	}
	if value.Kind != wcb.Map {
		return Descriptor{}, werrors.NewNonCanonical("layout descriptor must be a WCB map")
	}
	m := value.Map
	if err := checkAllowedKeys(m, []string{
		"schema", "dtype", "shape", "layout", "block_shape", "axis", "shards", "index",
	}, "layout descriptor"); err != nil {
		return Descriptor{}, err
	}
	schema, err := textField(m, "schema", "layout descriptor")
	if err != nil {
		return Descriptor{}, err
	}
	if schema != LayoutSchema {
		return Descriptor{}, werrors.UnknownSchemaf(
			"unknown layout descriptor schema %q", schema)
	}
	descriptor, err := descriptorFromMap(m, lim)
	if err != nil {
		return Descriptor{}, err
	}
	if err := descriptor.Validate(lim); err != nil {
		return Descriptor{}, err
	}
	reencoded, err := wcb.Encode(descriptor.WCBValue(), lim)
	if err != nil {
		return Descriptor{}, err
	}
	if !bytesEqual(reencoded, bytes) {
		return Descriptor{}, werrors.NewNonCanonical(
			"layout descriptor bytes are not in canonical form")
	}
	return descriptor, nil
}

func descriptorFromMap(m map[string]wcb.Value, lim limits.Limits) (Descriptor, error) {
	dtype, err := textField(m, "dtype", "layout descriptor")
	if err != nil {
		return Descriptor{}, err
	}
	if _, err := DtypeSize(dtype); err != nil {
		return Descriptor{}, err
	}
	shape, err := u64ArrayField(m, "shape", "layout descriptor")
	if err != nil {
		return Descriptor{}, err
	}
	layoutName, err := textField(m, "layout", "layout descriptor")
	if err != nil {
		return Descriptor{}, err
	}
	layout := Layout{}
	switch layoutName {
	case "row-major":
		layout.Kind = RowMajor
		if err := forbidFields(m, layoutName, "block_shape", "axis", "shards", "index"); err != nil {
			return Descriptor{}, err
		}
	case "column-major":
		layout.Kind = ColumnMajor
		if err := forbidFields(m, layoutName, "block_shape", "axis", "shards", "index"); err != nil {
			return Descriptor{}, err
		}
	case "blocked":
		layout.Kind = Blocked
		if err := forbidFields(m, layoutName, "axis", "shards", "index"); err != nil {
			return Descriptor{}, err
		}
		blockShape, err := u64ArrayField(m, "block_shape", "layout descriptor")
		if err != nil {
			return Descriptor{}, err
		}
		layout.BlockShape = blockShape
	case "sharded":
		layout.Kind = Sharded
		if err := forbidFields(m, layoutName, "block_shape"); err != nil {
			return Descriptor{}, err
		}
		axis, err := u32Field(m, "axis", "layout descriptor")
		if err != nil {
			return Descriptor{}, err
		}
		shards, err := u32Field(m, "shards", "layout descriptor")
		if err != nil {
			return Descriptor{}, err
		}
		index, err := u32Field(m, "index", "layout descriptor")
		if err != nil {
			return Descriptor{}, err
		}
		layout.Axis, layout.Shards, layout.Index = axis, shards, index
	default:
		return Descriptor{}, werrors.UnsupportedFeaturef(
			"unknown tensor layout %q (v1 allows row-major, column-major, blocked, sharded)",
			layoutName)
	}
	descriptor := Descriptor{Dtype: dtype, Shape: shape, Layout: layout}
	if err := descriptor.Validate(lim); err != nil {
		return Descriptor{}, err
	}
	return descriptor, nil
}

func validateShardParameters(shape []uint64, axis, shards uint32) error {
	if shards == 0 {
		return werrors.NewNonCanonical("shard count must be at least 1")
	}
	if int(axis) >= len(shape) {
		return werrors.NonCanonicalf("shard axis %d is not below rank %d", axis, len(shape))
	}
	if shape[axis]%uint64(shards) != 0 {
		return werrors.NonCanonicalf(
			"shape[%d] = %d is not divisible by shard count %d", axis, shape[axis], shards)
	}
	return nil
}

func forbidFields(m map[string]wcb.Value, layout string, fields ...string) error {
	for _, field := range fields {
		if _, ok := m[field]; ok {
			return werrors.NonCanonicalf(
				"layout %q must not carry the %q field", layout, field)
		}
	}
	return nil
}

// Region is a half-open hyper-rectangle in logical coordinates.
type Region struct {
	Start []uint64
	End   []uint64
}

// NewRegion constructs an unchecked region.
func NewRegion(start, end []uint64) Region {
	return Region{Start: append([]uint64(nil), start...), End: append([]uint64(nil), end...)}
}

// FullRegion is the whole tensor.
func FullRegion(shape []uint64) Region {
	return Region{
		Start: make([]uint64, len(shape)),
		End:   append([]uint64(nil), shape...),
	}
}

// Extent is the per-axis extent.
func (r Region) Extent() []uint64 {
	out := make([]uint64, len(r.Start))
	for axis := range r.Start {
		if r.End[axis] > r.Start[axis] {
			out[axis] = r.End[axis] - r.Start[axis]
		}
	}
	return out
}

// IsEmpty reports a zero-extent axis.
func (r Region) IsEmpty() bool {
	for axis := range r.Start {
		if r.Start[axis] >= r.End[axis] {
			return true
		}
	}
	return false
}

// Validate checks rank and bounds against shape.
func (r Region) Validate(shape []uint64) error {
	if len(r.Start) != len(shape) || len(r.End) != len(shape) {
		return werrors.NonCanonicalf(
			"region rank (%d, %d) does not equal shape rank %d",
			len(r.Start), len(r.End), len(shape))
	}
	for axis := range shape {
		if r.Start[axis] > r.End[axis] {
			return werrors.NonCanonicalf(
				"region axis %d start %d is after end %d", axis, r.Start[axis], r.End[axis])
		}
		if r.End[axis] > shape[axis] {
			return werrors.NonCanonicalf(
				"region axis %d end %d exceeds shape extent %d", axis, r.End[axis], shape[axis])
		}
	}
	return nil
}

// RegionRead is the result of a region read.
type RegionRead struct {
	Shape        []uint64
	Bytes        []byte
	UnitsTouched uint64
}

// ---------------------------------------------------------------------------
// Shard containers
// ---------------------------------------------------------------------------

// DecodedShard is a decoded self-describing shard.
type DecodedShard struct {
	Descriptor Descriptor
	Bytes      []byte
}

// DescriptorPayloadLen is the expected container payload length.
func DescriptorPayloadLen(d Descriptor, lim limits.Limits) (uint64, error) {
	if err := d.Validate(lim); err != nil {
		return 0, err
	}
	count, err := ValidateShape(d.Shape, lim)
	if err != nil {
		return 0, err
	}
	elements := count
	if d.Layout.Kind == Sharded {
		shardShape, err := shardShapeFor(d.Shape, d.Layout.Axis, d.Layout.Shards)
		if err != nil {
			return 0, err
		}
		elements, err = checkedProduct(shardShape)
		if err != nil {
			return 0, err
		}
	}
	return PayloadLen(elements, d.Dtype)
}

// ShardPayloadHash is H("weft/tensor/shard-payload/v1", payload).
func ShardPayloadHash(payload []byte) hash.Hash32 {
	return hash.Domain(DomainTensorShardPayload, payload)
}

// EncodeShard encodes u32le(descriptor_len) || WCB(shard) || payload.
func EncodeShard(d Descriptor, payload []byte, lim limits.Limits) ([]byte, error) {
	if err := d.Validate(lim); err != nil {
		return nil, err
	}
	expected, err := DescriptorPayloadLen(d, lim)
	if err != nil {
		return nil, err
	}
	if expected != uint64(len(payload)) {
		return nil, werrors.Malformedf(
			"shard payload is %d bytes but the descriptor requires %d", len(payload), expected)
	}
	descriptorBytes, err := wcb.Encode(wcb.MapValue(shardMap(d, expected, ShardPayloadHash(payload))), lim)
	if err != nil {
		return nil, err
	}
	if uint64(len(descriptorBytes)) > math.MaxUint32 {
		return nil, werrors.NewLimitExceeded("shard descriptor does not fit u32")
	}
	out := make([]byte, 0, 4+len(descriptorBytes)+len(payload))
	var prefix [4]byte
	binary.LittleEndian.PutUint32(prefix[:], uint32(len(descriptorBytes)))
	out = append(out, prefix[:]...)
	out = append(out, descriptorBytes...)
	out = append(out, payload...)
	return out, nil
}

func shardMap(d Descriptor, payloadBytes uint64, payloadHash hash.Hash32) map[string]wcb.Value {
	m := descriptorMap(d)
	m["schema"] = wcb.TextValue(ShardSchema)
	m["payload_bytes"] = wcb.U64Value(payloadBytes)
	m["payload_hash"] = wcb.BytesValue(payloadHash.Bytes())
	return m
}

// DecodeShard decodes a container and verifies descriptor, length, and hash.
func DecodeShard(bytes []byte, lim limits.Limits) (DecodedShard, error) {
	if len(bytes) < 4 {
		return DecodedShard{}, werrors.NewMalformed(
			"shard container is shorter than its 4-byte descriptor length")
	}
	length := uint64(binary.LittleEndian.Uint32(bytes[0:4]))
	if length > uint64(len(bytes)-4) {
		return DecodedShard{}, werrors.NewMalformed("shard descriptor extends past the container")
	}
	descriptorBytes := bytes[4 : 4+length]
	payload := bytes[4+length:]
	value, err := wcb.Decode(descriptorBytes, lim)
	if err != nil {
		return DecodedShard{}, err
	}
	if value.Kind != wcb.Map {
		return DecodedShard{}, werrors.NewNonCanonical("shard descriptor must be a WCB map")
	}
	m := value.Map
	if err := checkAllowedKeys(m, []string{
		"schema", "dtype", "shape", "layout", "block_shape", "axis", "shards", "index",
		"payload_bytes", "payload_hash",
	}, "shard descriptor"); err != nil {
		return DecodedShard{}, err
	}
	schema, err := textField(m, "schema", "shard descriptor")
	if err != nil {
		return DecodedShard{}, err
	}
	if schema != ShardSchema {
		return DecodedShard{}, werrors.UnknownSchemaf(
			"unknown shard descriptor schema %q", schema)
	}
	descriptor, err := descriptorFromMap(m, lim)
	if err != nil {
		return DecodedShard{}, err
	}
	declaredBytes, err := u64Field(m, "payload_bytes", "shard descriptor")
	if err != nil {
		return DecodedShard{}, err
	}
	expected, err := DescriptorPayloadLen(descriptor, lim)
	if err != nil {
		return DecodedShard{}, err
	}
	if declaredBytes != expected {
		return DecodedShard{}, werrors.NonCanonicalf(
			"shard payload_bytes %d does not match layout length %d", declaredBytes, expected)
	}
	if declaredBytes != uint64(len(payload)) {
		return DecodedShard{}, werrors.Malformedf(
			"shard payload is %d bytes but the descriptor declares %d", len(payload), declaredBytes)
	}
	declaredHash, err := hashField(m, "payload_hash", "shard descriptor")
	if err != nil {
		return DecodedShard{}, err
	}
	if declaredHash != ShardPayloadHash(payload) {
		return DecodedShard{}, werrors.NewCorrupt("shard payload_hash mismatch")
	}
	reencoded, err := wcb.Encode(wcb.MapValue(shardMap(descriptor, declaredBytes, declaredHash)), lim)
	if err != nil {
		return DecodedShard{}, err
	}
	if !bytesEqual(reencoded, descriptorBytes) {
		return DecodedShard{}, werrors.NewNonCanonical(
			"shard descriptor bytes are not in canonical form")
	}
	return DecodedShard{Descriptor: descriptor, Bytes: payload}, nil
}

// ShardRoot verifies a complete shard set and returns the logical root.
//
// Shard order is not significant: the descriptor's `index` is authoritative and
// shards are reordered deterministically by it. Duplicate, missing,
// out-of-range, and inconsistent shards are rejected.
func ShardRoot(shards []DecodedShard, lim limits.Limits) (hash.Hash32, error) {
	if len(shards) == 0 {
		return hash.Hash32{}, werrors.NewMalformed("shard set is empty")
	}
	first := shards[0].Descriptor
	if first.Layout.Kind != Sharded {
		return hash.Hash32{}, werrors.Malformedf(
			"shard root requires sharded descriptors, got %q", first.Layout.Name())
	}
	axis, count := first.Layout.Axis, first.Layout.Shards
	if count == 0 {
		return hash.Hash32{}, werrors.NewNonCanonical("shard count must be at least 1")
	}
	if len(shards) != int(count) {
		return hash.Hash32{}, werrors.Malformedf(
			"shard set has %d entries but the descriptor declares %d", len(shards), count)
	}
	ordered := make([][]byte, count)
	seen := make([]bool, count)
	for _, shard := range shards {
		descriptor := shard.Descriptor
		if descriptor.Layout.Kind != Sharded {
			return hash.Hash32{}, werrors.Malformedf(
				"shard root requires sharded descriptors, got %q", descriptor.Layout.Name())
		}
		if descriptor.Dtype != first.Dtype ||
			!u64SliceEqual(descriptor.Shape, first.Shape) ||
			descriptor.Layout.Axis != axis ||
			descriptor.Layout.Shards != count {
			return hash.Hash32{}, werrors.NewNonCanonical(
				"shard descriptors disagree on dtype, shape, axis, or shard count")
		}
		if err := descriptor.Validate(lim); err != nil {
			return hash.Hash32{}, err
		}
		index := descriptor.Layout.Index
		if index >= count {
			return hash.Hash32{}, werrors.NonCanonicalf(
				"shard index %d is not below shard count %d", index, count)
		}
		if seen[index] {
			return hash.Hash32{}, werrors.NonCanonicalf("duplicate shard index %d", index)
		}
		shardShape, err := shardShapeFor(descriptor.Shape, axis, count)
		if err != nil {
			return hash.Hash32{}, err
		}
		elementCount, err := checkedProduct(shardShape)
		if err != nil {
			return hash.Hash32{}, err
		}
		expected, err := PayloadLen(elementCount, descriptor.Dtype)
		if err != nil {
			return hash.Hash32{}, err
		}
		if uint64(len(shard.Bytes)) != expected {
			return hash.Hash32{}, werrors.Malformedf(
				"shard %d is %d bytes but requires %d", index, len(shard.Bytes), expected)
		}
		ordered[index] = append([]byte(nil), shard.Bytes...)
		seen[index] = true
	}
	for index, ok := range seen {
		if !ok {
			return hash.Hash32{}, werrors.NonCanonicalf("missing shard index %d", index)
		}
	}
	physical, err := NewSharded(first.Dtype, first.Shape, axis, ordered, lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	return physical.LogicalRoot(lim)
}

// AssembleShards reassembles a shard set to canonical row-major bytes.
func AssembleShards(shards []DecodedShard, lim limits.Limits) ([]byte, error) {
	if len(shards) == 0 {
		return nil, werrors.NewMalformed("shard set is empty")
	}
	first := shards[0].Descriptor
	if first.Layout.Kind != Sharded {
		return nil, werrors.Malformedf(
			"shard reassembly requires sharded descriptors, got %q", first.Layout.Name())
	}
	count := first.Layout.Shards
	ordered := make([][]byte, count)
	for _, shard := range shards {
		if shard.Descriptor.Layout.Kind != Sharded {
			return nil, werrors.NewMalformed("shard reassembly requires shard descriptors")
		}
		index := shard.Descriptor.Layout.Index
		if index >= count || ordered[index] != nil {
			return nil, werrors.NewNonCanonical("shard indices are out of range or duplicated")
		}
		ordered[index] = append([]byte(nil), shard.Bytes...)
	}
	for index, payload := range ordered {
		if payload == nil {
			return nil, werrors.NonCanonicalf("shard set is missing index %d", index)
		}
	}
	physical, err := NewSharded(first.Dtype, first.Shape, first.Layout.Axis, ordered, lim)
	if err != nil {
		return nil, err
	}
	return physical.AssembleWithLimits(lim)
}

// ---------------------------------------------------------------------------
// WCB accessors
// ---------------------------------------------------------------------------

func checkAllowedKeys(m map[string]wcb.Value, allowed []string, what string) error {
	for key := range m {
		ok := false
		for _, candidate := range allowed {
			if key == candidate {
				ok = true
				break
			}
		}
		if !ok {
			return werrors.NonCanonicalf("%s has unknown key %q", what, key)
		}
	}
	return nil
}

func textField(m map[string]wcb.Value, key, what string) (string, error) {
	value, ok := m[key]
	if !ok || value.Kind != wcb.Text {
		return "", werrors.NonCanonicalf("%s %q must be text", what, key)
	}
	return value.Str, nil
}

func u64Field(m map[string]wcb.Value, key, what string) (uint64, error) {
	value, ok := m[key]
	if !ok || value.Kind != wcb.U64 {
		return 0, werrors.NonCanonicalf("%s %q must be u64", what, key)
	}
	return value.U, nil
}

func u32Field(m map[string]wcb.Value, key, what string) (uint32, error) {
	value, err := u64Field(m, key, what)
	if err != nil {
		return 0, err
	}
	if value > math.MaxUint32 {
		return 0, werrors.NonCanonicalf("%s %q must fit u32, got %d", what, key, value)
	}
	return uint32(value), nil
}

func u64ArrayField(m map[string]wcb.Value, key, what string) ([]uint64, error) {
	value, ok := m[key]
	if !ok || value.Kind != wcb.Array {
		return nil, werrors.NonCanonicalf("%s %q must be an array", what, key)
	}
	out := make([]uint64, 0, len(value.Array))
	for _, item := range value.Array {
		if item.Kind != wcb.U64 {
			return nil, werrors.NonCanonicalf("%s %q elements must be u64", what, key)
		}
		out = append(out, item.U)
	}
	return out, nil
}

func hashField(m map[string]wcb.Value, key, what string) (hash.Hash32, error) {
	value, ok := m[key]
	if !ok || value.Kind != wcb.Bytes {
		return hash.Hash32{}, werrors.NonCanonicalf("%s %q must be bytes", what, key)
	}
	h, err := hash.FromSlice(value.Bytes)
	if err != nil {
		return hash.Hash32{}, werrors.NonCanonicalf("%s %q must be 32 bytes", what, key)
	}
	return h, nil
}

func u64SliceEqual(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
