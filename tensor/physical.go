package tensor

import (
	"math"

	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// Physical is an in-memory physical arrangement of one logical tensor.
type Physical struct {
	kind       LayoutKind
	dtype      string
	shape      []uint64
	bytes      []byte   // row-major or column-major
	blockShape []uint64 // blocked
	blocks     [][]byte // blocked, canonical grid order
	axis       uint32   // sharded
	shards     [][]byte // sharded, index order
}

// NewRowMajor wraps canonical row-major bytes.
func NewRowMajor(dtype string, shape []uint64, bytes []byte, lim limits.Limits) (*Physical, error) {
	if _, err := CheckPayload(dtype, shape, bytes, lim); err != nil {
		return nil, err
	}
	return &Physical{
		kind:  RowMajor,
		dtype: dtype,
		shape: append([]uint64(nil), shape...),
		bytes: append([]byte(nil), bytes...),
	}, nil
}

// NewColumnMajor wraps column-major bytes.
func NewColumnMajor(dtype string, shape []uint64, bytes []byte, lim limits.Limits) (*Physical, error) {
	if _, err := CheckPayload(dtype, shape, bytes, lim); err != nil {
		return nil, err
	}
	return &Physical{
		kind:  ColumnMajor,
		dtype: dtype,
		shape: append([]uint64(nil), shape...),
		bytes: append([]byte(nil), bytes...),
	}, nil
}

// NewBlocked wraps canonical block payloads.
func NewBlocked(dtype string, shape, blockShape []uint64, blocks [][]byte, lim limits.Limits) (*Physical, error) {
	descriptor := BlockedDescriptor(dtype, shape, blockShape)
	if err := descriptor.Validate(lim); err != nil {
		return nil, err
	}
	count, err := ValidateShape(shape, lim)
	if err != nil {
		return nil, err
	}
	payload, err := PayloadLen(count, dtype)
	if err != nil {
		return nil, err
	}
	if err := limits.CheckLen(payload, lim.MaxDecodedBytes, "blocked tensor"); err != nil {
		return nil, err
	}
	extents, err := BlockExtents(shape, blockShape)
	if err != nil {
		return nil, err
	}
	if len(extents) != len(blocks) {
		return nil, werrors.Malformedf(
			"blocked tensor has %d block payloads but the grid needs %d", len(blocks), len(extents))
	}
	for index, extent := range extents {
		elementCount, err := checkedProduct(extent)
		if err != nil {
			return nil, err
		}
		expected, err := PayloadLen(elementCount, dtype)
		if err != nil {
			return nil, err
		}
		if uint64(len(blocks[index])) != expected {
			return nil, werrors.Malformedf(
				"block %d is %d bytes but requires %d", index, len(blocks[index]), expected)
		}
	}
	return &Physical{
		kind:       Blocked,
		dtype:      dtype,
		shape:      append([]uint64(nil), shape...),
		blockShape: append([]uint64(nil), blockShape...),
		blocks:     cloneBlocks(blocks),
	}, nil
}

// NewSharded wraps shard payloads in index order.
func NewSharded(dtype string, shape []uint64, axis uint32, shards [][]byte, lim limits.Limits) (*Physical, error) {
	if _, err := ValidateShape(shape, lim); err != nil {
		return nil, err
	}
	if len(shards) > math.MaxUint32 {
		return nil, werrors.NewLimitExceeded("shard count does not fit u32")
	}
	count := uint32(len(shards))
	if err := validateShardParameters(shape, axis, count); err != nil {
		return nil, err
	}
	shardShape, err := shardShapeFor(shape, axis, count)
	if err != nil {
		return nil, err
	}
	elementCount, err := checkedProduct(shardShape)
	if err != nil {
		return nil, err
	}
	expected, err := PayloadLen(elementCount, dtype)
	if err != nil {
		return nil, err
	}
	if count != 0 && expected > math.MaxUint64/uint64(count) {
		return nil, werrors.NewLimitExceeded("sharded tensor size overflows u64")
	}
	if err := limits.CheckLen(expected*uint64(count), lim.MaxDecodedBytes, "sharded tensor"); err != nil {
		return nil, err
	}
	for index, payload := range shards {
		if uint64(len(payload)) != expected {
			return nil, werrors.Malformedf(
				"shard %d is %d bytes but requires %d", index, len(payload), expected)
		}
	}
	return &Physical{
		kind:   Sharded,
		dtype:  dtype,
		shape:  append([]uint64(nil), shape...),
		axis:   axis,
		shards: cloneBlocks(shards),
	}, nil
}

// Kind reports the physical layout kind.
func (p *Physical) Kind() LayoutKind { return p.kind }

// Dtype reports the element dtype.
func (p *Physical) Dtype() string { return p.dtype }

// Shape reports the logical shape.
func (p *Physical) Shape() []uint64 { return append([]uint64(nil), p.shape...) }

// Assemble reassembles canonical row-major logical bytes.
func (p *Physical) Assemble() ([]byte, error) {
	return p.AssembleWithLimits(limits.Default())
}

// AssembleWithLimits reassembles canonical row-major bytes under explicit limits.
func (p *Physical) AssembleWithLimits(lim limits.Limits) ([]byte, error) {
	size, err := DtypeSize(p.dtype)
	if err != nil {
		return nil, err
	}
	esize := int(size)
	count, err := checkedProduct(p.shape)
	if err != nil {
		return nil, err
	}
	payload, err := PayloadLen(count, p.dtype)
	if err != nil {
		return nil, err
	}
	if err := limits.CheckLen(payload, lim.MaxDecodedBytes, "assembled tensor"); err != nil {
		return nil, err
	}
	total := int(payload)
	switch p.kind {
	case RowMajor:
		return append([]byte(nil), p.bytes...), nil
	case ColumnMajor:
		out := make([]byte, total)
		strideF, err := columnMajorStrides(p.shape)
		if err != nil {
			return nil, err
		}
		position := 0
		forEachIndex(make([]uint64, len(p.shape)), p.shape, func(idx []uint64) {
			if position+esize > len(out) {
				return
			}
			var off uint64
			for axis := range idx {
				off += idx[axis] * strideF[axis]
			}
			source := int(off) * esize
			copy(out[position:position+esize], p.bytes[source:source+esize])
			position += esize
		})
		return out, nil
	case Blocked:
		out := make([]byte, total)
		extents, err := BlockExtents(p.shape, p.blockShape)
		if err != nil {
			return nil, err
		}
		grid, err := blockGrid(p.shape, p.blockShape)
		if err != nil {
			return nil, err
		}
		zeros := make([]uint64, len(p.shape))
		for blockIndex, extent := range extents {
			coords := gridCoords(grid, blockIndex)
			dstStart := make([]uint64, len(p.shape))
			for axis := range p.shape {
				dstStart[axis] = coords[axis] * p.blockShape[axis]
			}
			if err := copyBoxRowMajorToRowMajor(
				out, p.shape, dstStart,
				p.blocks[blockIndex], extent, zeros,
				extent, esize); err != nil {
				return nil, err
			}
		}
		return out, nil
	case Sharded:
		out := make([]byte, total)
		shardShape, err := shardShapeFor(p.shape, p.axis, uint32(len(p.shards)))
		if err != nil {
			return nil, err
		}
		plane := p.shape[p.axis] / uint64(len(p.shards))
		zeros := make([]uint64, len(p.shape))
		for index, payload := range p.shards {
			dstStart := make([]uint64, len(p.shape))
			dstStart[p.axis] = uint64(index) * plane
			if err := copyBoxRowMajorToRowMajor(
				out, p.shape, dstStart,
				payload, shardShape, zeros,
				shardShape, esize); err != nil {
				return nil, err
			}
		}
		return out, nil
	default:
		return nil, werrors.UnsupportedFeaturef("unknown tensor layout kind %d", p.kind)
	}
}

// LogicalRoot reassembles and returns the logical tensor root.
func (p *Physical) LogicalRoot(lim limits.Limits) (hash.Hash32, error) {
	bytes, err := p.AssembleWithLimits(lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	return LogicalRoot(p.dtype, p.shape, bytes, lim)
}

// ReadRegion reads a region, touching only the physical units it intersects.
func (p *Physical) ReadRegion(region Region) (RegionRead, error) {
	return p.ReadRegionWithLimits(region, limits.Default())
}

// ReadRegionWithLimits reads a region under explicit allocation limits.
func (p *Physical) ReadRegionWithLimits(region Region, lim limits.Limits) (RegionRead, error) {
	if err := region.Validate(p.shape); err != nil {
		return RegionRead{}, err
	}
	size, err := DtypeSize(p.dtype)
	if err != nil {
		return RegionRead{}, err
	}
	esize := int(size)
	extent := region.Extent()
	if region.IsEmpty() {
		return RegionRead{Shape: extent, Bytes: []byte{}, UnitsTouched: 0}, nil
	}
	count, err := checkedProduct(extent)
	if err != nil {
		return RegionRead{}, err
	}
	payload, err := PayloadLen(count, p.dtype)
	if err != nil {
		return RegionRead{}, err
	}
	if err := limits.CheckLen(payload, lim.MaxDecodedBytes, "region payload"); err != nil {
		return RegionRead{}, err
	}
	out := make([]byte, int(payload))
	zeros := make([]uint64, len(p.shape))
	var touched uint64
	switch p.kind {
	case RowMajor:
		if err := copyBoxRowMajorToRowMajor(
			out, extent, zeros, p.bytes, p.shape, region.Start, extent, esize); err != nil {
			return RegionRead{}, err
		}
		touched = 1
	case ColumnMajor:
		if err := copyBoxColumnMajorToRowMajor(
			out, extent, zeros, p.bytes, p.shape, region.Start, extent, esize); err != nil {
			return RegionRead{}, err
		}
		touched = 1
	case Blocked:
		extents, err := BlockExtents(p.shape, p.blockShape)
		if err != nil {
			return RegionRead{}, err
		}
		grid, err := blockGrid(p.shape, p.blockShape)
		if err != nil {
			return RegionRead{}, err
		}
		for blockIndex, blockExtent := range extents {
			coords := gridCoords(grid, blockIndex)
			blockStart := make([]uint64, len(p.shape))
			for axis := range p.shape {
				blockStart[axis] = coords[axis] * p.blockShape[axis]
			}
			interStart := make([]uint64, len(p.shape))
			interExtent := make([]uint64, len(p.shape))
			any := true
			for axis := range p.shape {
				blockEnd := blockStart[axis] + blockExtent[axis]
				lo := maxU64(region.Start[axis], blockStart[axis])
				hi := minU64(region.End[axis], blockEnd)
				if lo >= hi {
					any = false
					break
				}
				interStart[axis] = lo
				interExtent[axis] = hi - lo
			}
			if !any {
				continue
			}
			touched++
			srcStart := make([]uint64, len(p.shape))
			dstStart := make([]uint64, len(p.shape))
			for axis := range p.shape {
				srcStart[axis] = interStart[axis] - blockStart[axis]
				dstStart[axis] = interStart[axis] - region.Start[axis]
			}
			if err := copyBoxRowMajorToRowMajor(
				out, extent, dstStart, p.blocks[blockIndex], blockExtent, srcStart,
				interExtent, esize); err != nil {
				return RegionRead{}, err
			}
		}
	case Sharded:
		plane := p.shape[p.axis] / uint64(len(p.shards))
		first := region.Start[p.axis] / plane
		last := (region.End[p.axis] - 1) / plane
		shardShape, err := shardShapeFor(p.shape, p.axis, uint32(len(p.shards)))
		if err != nil {
			return RegionRead{}, err
		}
		for shardIndex := first; shardIndex <= last; shardIndex++ {
			payloadBytes := p.shards[shardIndex]
			srcStart := append([]uint64(nil), region.Start...)
			dstStart := make([]uint64, len(p.shape))
			interExtent := append([]uint64(nil), extent...)
			slabStart := shardIndex * plane
			slabEnd := slabStart + plane
			lo := maxU64(region.Start[p.axis], slabStart)
			hi := minU64(region.End[p.axis], slabEnd)
			srcStart[p.axis] = lo - slabStart
			dstStart[p.axis] = lo - region.Start[p.axis]
			interExtent[p.axis] = hi - lo
			if err := copyBoxRowMajorToRowMajor(
				out, extent, dstStart, payloadBytes, shardShape, srcStart,
				interExtent, esize); err != nil {
				return RegionRead{}, err
			}
		}
		touched = last - first + 1
	default:
		return RegionRead{}, werrors.UnsupportedFeaturef("unknown tensor layout kind %d", p.kind)
	}
	return RegionRead{Shape: extent, Bytes: out, UnitsTouched: touched}, nil
}

// ColumnMajorBytes converts canonical row-major bytes to column-major bytes.
func ColumnMajorBytes(dtype string, shape []uint64, rowMajor []byte, lim limits.Limits) ([]byte, error) {
	if _, err := CheckPayload(dtype, shape, rowMajor, lim); err != nil {
		return nil, err
	}
	size, _ := DtypeSize(dtype)
	esize := int(size)
	out := make([]byte, len(rowMajor))
	strideF, err := columnMajorStrides(shape)
	if err != nil {
		return nil, err
	}
	position := 0
	forEachIndex(make([]uint64, len(shape)), shape, func(idx []uint64) {
		if position+esize > len(rowMajor) {
			return
		}
		var off uint64
		for axis := range idx {
			off += idx[axis] * strideF[axis]
		}
		target := int(off) * esize
		copy(out[target:target+esize], rowMajor[position:position+esize])
		position += esize
	})
	return out, nil
}

// BlockExtents returns block extents in canonical block-grid order.
func BlockExtents(shape, blockShape []uint64) ([][]uint64, error) {
	if len(shape) != len(blockShape) {
		return nil, werrors.NewNonCanonical("block_shape rank must equal shape rank")
	}
	for _, block := range blockShape {
		if block == 0 {
			return nil, werrors.NewNonCanonical("block_shape dimensions must be >= 1")
		}
	}
	grid, err := blockGrid(shape, blockShape)
	if err != nil {
		return nil, err
	}
	gridCount, err := checkedProduct(grid)
	if err != nil {
		return nil, err
	}
	extents := make([][]uint64, 0, gridCount)
	for blockIndex := 0; blockIndex < int(gridCount); blockIndex++ {
		coords := gridCoords(grid, blockIndex)
		extent := make([]uint64, len(shape))
		for axis := range shape {
			start := coords[axis] * blockShape[axis]
			remaining := shape[axis] - start
			extent[axis] = minU64(blockShape[axis], remaining)
		}
		extents = append(extents, extent)
	}
	return extents, nil
}

// BlockedBytes splits canonical row-major bytes into block payloads.
func BlockedBytes(dtype string, shape, blockShape []uint64, rowMajor []byte, lim limits.Limits) ([][]byte, error) {
	if _, err := CheckPayload(dtype, shape, rowMajor, lim); err != nil {
		return nil, err
	}
	size, _ := DtypeSize(dtype)
	esize := int(size)
	extents, err := BlockExtents(shape, blockShape)
	if err != nil {
		return nil, err
	}
	grid, err := blockGrid(shape, blockShape)
	if err != nil {
		return nil, err
	}
	zeros := make([]uint64, len(shape))
	blocks := make([][]byte, 0, len(extents))
	for blockIndex, extent := range extents {
		coords := gridCoords(grid, blockIndex)
		srcStart := make([]uint64, len(shape))
		for axis := range shape {
			srcStart[axis] = coords[axis] * blockShape[axis]
		}
		elementCount, err := checkedProduct(extent)
		if err != nil {
			return nil, err
		}
		blockBytes, err := PayloadLen(elementCount, dtype)
		if err != nil {
			return nil, err
		}
		block := make([]byte, int(blockBytes))
		if err := copyBoxRowMajorToRowMajor(
			block, extent, zeros, rowMajor, shape, srcStart, extent, esize); err != nil {
			return nil, err
		}
		blocks = append(blocks, block)
	}
	return blocks, nil
}

// ShardShape returns the logical shape of shard index.
func ShardShape(shape []uint64, axis, shards, index uint32) ([]uint64, error) {
	if err := validateShardParameters(shape, axis, shards); err != nil {
		return nil, err
	}
	if index >= shards {
		return nil, werrors.NonCanonicalf(
			"shard index %d is not below shard count %d", index, shards)
	}
	return shardShapeFor(shape, axis, shards)
}

func shardShapeFor(shape []uint64, axis, shards uint32) ([]uint64, error) {
	if err := validateShardParameters(shape, axis, shards); err != nil {
		return nil, err
	}
	out := append([]uint64(nil), shape...)
	out[axis] /= uint64(shards)
	return out, nil
}

// ShardBytes extracts one shard's canonical row-major bytes.
func ShardBytes(dtype string, shape []uint64, axis, shards, index uint32, rowMajor []byte, lim limits.Limits) ([]byte, error) {
	if _, err := CheckPayload(dtype, shape, rowMajor, lim); err != nil {
		return nil, err
	}
	shardShape, err := ShardShape(shape, axis, shards, index)
	if err != nil {
		return nil, err
	}
	size, _ := DtypeSize(dtype)
	esize := int(size)
	plane := shape[axis] / uint64(shards)
	srcStart := make([]uint64, len(shape))
	srcStart[axis] = uint64(index) * plane
	elementCount, err := checkedProduct(shardShape)
	if err != nil {
		return nil, err
	}
	shardBytes, err := PayloadLen(elementCount, dtype)
	if err != nil {
		return nil, err
	}
	out := make([]byte, int(shardBytes))
	if err := copyBoxRowMajorToRowMajor(
		out, shardShape, make([]uint64, len(shape)),
		rowMajor, shape, srcStart, shardShape, esize); err != nil {
		return nil, err
	}
	return out, nil
}

// SplitShards splits canonical row-major bytes into all shards in index order.
func SplitShards(dtype string, shape []uint64, axis, shards uint32, rowMajor []byte, lim limits.Limits) ([][]byte, error) {
	if _, err := CheckPayload(dtype, shape, rowMajor, lim); err != nil {
		return nil, err
	}
	if err := validateShardParameters(shape, axis, shards); err != nil {
		return nil, err
	}
	out := make([][]byte, 0, shards)
	for index := uint32(0); index < shards; index++ {
		payload, err := ShardBytes(dtype, shape, axis, shards, index, rowMajor, lim)
		if err != nil {
			return nil, err
		}
		out = append(out, payload)
	}
	return out, nil
}

// SplitBlockedPayload splits a concatenated blocked payload into block payloads.
func SplitBlockedPayload(dtype string, shape, blockShape []uint64, payload []byte, lim limits.Limits) ([][]byte, error) {
	if _, err := CheckPayload(dtype, shape, payload, lim); err != nil {
		return nil, err
	}
	extents, err := BlockExtents(shape, blockShape)
	if err != nil {
		return nil, err
	}
	blocks := make([][]byte, 0, len(extents))
	position := 0
	for _, extent := range extents {
		elementCount, err := checkedProduct(extent)
		if err != nil {
			return nil, err
		}
		blockBytes, err := PayloadLen(elementCount, dtype)
		if err != nil {
			return nil, err
		}
		end := position + int(blockBytes)
		if end > len(payload) {
			return nil, werrors.NewMalformed("blocked payload is truncated")
		}
		blocks = append(blocks, append([]byte(nil), payload[position:end]...))
		position = end
	}
	if position != len(payload) {
		return nil, werrors.NewMalformed("blocked payload has trailing bytes")
	}
	return blocks, nil
}

// ---------------------------------------------------------------------------
// Index and copy helpers
// ---------------------------------------------------------------------------

func checkedProduct(values []uint64) (uint64, error) {
	product := uint64(1)
	for _, value := range values {
		if value != 0 && product > math.MaxUint64/value {
			return 0, werrors.NewLimitExceeded("tensor dimension product overflows u64")
		}
		product *= value
	}
	return product, nil
}

func rowMajorStrides(shape []uint64) ([]uint64, error) {
	strides := make([]uint64, len(shape))
	if len(shape) == 0 {
		return strides, nil
	}
	strides[len(shape)-1] = 1
	for axis := len(shape) - 2; axis >= 0; axis-- {
		if shape[axis+1] != 0 && strides[axis+1] > math.MaxUint64/shape[axis+1] {
			return nil, werrors.NewLimitExceeded("tensor stride overflows u64")
		}
		strides[axis] = strides[axis+1] * shape[axis+1]
	}
	return strides, nil
}

func columnMajorStrides(shape []uint64) ([]uint64, error) {
	strides := make([]uint64, len(shape))
	if len(shape) == 0 {
		return strides, nil
	}
	strides[0] = 1
	for axis := 1; axis < len(shape); axis++ {
		if shape[axis-1] != 0 && strides[axis-1] > math.MaxUint64/shape[axis-1] {
			return nil, werrors.NewLimitExceeded("tensor stride overflows u64")
		}
		strides[axis] = strides[axis-1] * shape[axis-1]
	}
	return strides, nil
}

// forEachIndex visits every multi-index in [start, start+extent) in row-major
// order. A rank-0 box is a single point and calls f exactly once.
func forEachIndex(start, ext []uint64, f func([]uint64)) {
	for _, value := range ext {
		if value == 0 {
			return
		}
	}
	index := append([]uint64(nil), start...)
	if len(index) == 0 {
		f(nil)
		return
	}
	for {
		f(index)
		axis := len(index)
		for {
			if axis == 0 {
				return
			}
			axis--
			index[axis]++
			if index[axis] < start[axis]+ext[axis] {
				break
			}
			index[axis] = start[axis]
			if axis == 0 {
				return
			}
		}
	}
}

func copyBoxRowMajorToRowMajor(
	dst []byte, dstShape, dstStart []uint64,
	src []byte, srcShape, srcStart []uint64,
	ext []uint64, esize int,
) error {
	if len(ext) == 0 {
		return nil
	}
	for _, value := range ext {
		if value == 0 {
			return nil
		}
	}
	dstStrides, err := rowMajorStrides(dstShape)
	if err != nil {
		return err
	}
	srcStrides, err := rowMajorStrides(srcShape)
	if err != nil {
		return err
	}
	prefix := len(ext) - 1
	run := int(ext[prefix]) * esize
	zeros := make([]uint64, prefix)
	var failure error
	forEachIndex(zeros, ext[:prefix], func(local []uint64) {
		if failure != nil {
			return
		}
		dstOff := dstStart[prefix]
		srcOff := srcStart[prefix]
		for axis := 0; axis < prefix; axis++ {
			dstOff += (dstStart[axis] + local[axis]) * dstStrides[axis]
			srcOff += (srcStart[axis] + local[axis]) * srcStrides[axis]
		}
		dstByte := int(dstOff) * esize
		srcByte := int(srcOff) * esize
		if dstByte+run > len(dst) || srcByte+run > len(src) {
			failure = werrors.NewMalformed("internal tensor copy exceeded its buffers")
			return
		}
		copy(dst[dstByte:dstByte+run], src[srcByte:srcByte+run])
	})
	return failure
}

func copyBoxColumnMajorToRowMajor(
	dst []byte, dstShape, dstStart []uint64,
	src []byte, srcShape, srcStart []uint64,
	ext []uint64, esize int,
) error {
	if len(ext) == 0 {
		return nil
	}
	for _, value := range ext {
		if value == 0 {
			return nil
		}
	}
	dstStrides, err := rowMajorStrides(dstShape)
	if err != nil {
		return err
	}
	srcStrides, err := columnMajorStrides(srcShape)
	if err != nil {
		return err
	}
	rank := len(ext)
	var failure error
	forEachIndex(dstStart, ext, func(index []uint64) {
		if failure != nil {
			return
		}
		var dstOff uint64
		var srcOff uint64
		for axis := 0; axis < rank; axis++ {
			local := index[axis] - dstStart[axis]
			dstOff += index[axis] * dstStrides[axis]
			srcOff += (srcStart[axis] + local) * srcStrides[axis]
		}
		dstByte := int(dstOff) * esize
		srcByte := int(srcOff) * esize
		if dstByte+esize > len(dst) || srcByte+esize > len(src) {
			failure = werrors.NewMalformed("internal tensor copy exceeded its buffers")
			return
		}
		copy(dst[dstByte:dstByte+esize], src[srcByte:srcByte+esize])
	})
	return failure
}

func blockGrid(shape, blockShape []uint64) ([]uint64, error) {
	if len(shape) != len(blockShape) {
		return nil, werrors.NewNonCanonical("block_shape rank must equal shape rank")
	}
	grid := make([]uint64, len(shape))
	for axis := range shape {
		if blockShape[axis] == 0 {
			return nil, werrors.NewNonCanonical("block_shape dimensions must be >= 1")
		}
		grid[axis] = shape[axis] / blockShape[axis]
		if shape[axis]%blockShape[axis] != 0 {
			grid[axis]++
		}
	}
	return grid, nil
}

func gridCoords(grid []uint64, index int) []uint64 {
	coords := make([]uint64, len(grid))
	position := index
	for axis := len(grid) - 1; axis >= 0; axis-- {
		extent := int(grid[axis])
		coords[axis] = uint64(position % extent)
		position /= extent
	}
	return coords
}

func cloneBlocks(blocks [][]byte) [][]byte {
	out := make([][]byte, len(blocks))
	for i, block := range blocks {
		out[i] = append([]byte(nil), block...)
	}
	return out
}

func minU64(a, b uint64) uint64 {
	if a < b {
		return a
	}
	return b
}

func maxU64(a, b uint64) uint64 {
	if a > b {
		return a
	}
	return b
}
