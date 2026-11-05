package modes

// Mode 15 TENSOR_TILED (WEFT-AI-CODEC §16).
//
// A frame carries one tensor split into independently decodable tiles. The
// descriptor names the whole logical shape, the tile grid, and the byte range
// of every stored tile. Each tile stores its canonical row-major element bytes
// as byte planes, every plane coded with `huffman-canonical-v1` and its own
// table. Decoding one tile verifies that tile's root; a full decode rebuilds
// the whole tensor and verifies the logical root.

import (
	"math"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// TensorTiledSchema is the canonical descriptor schema.
const TensorTiledSchema = "weft.ai.tensor_tiled.v1"

// TensorTileLayout is the only tile layout implemented in v1.
const TensorTileLayout = "row_major"

// TensorTiledEntry is one independently decodable tile.
type TensorTiledEntry struct {
	Index         uint64
	Dims          []uint64
	Offset        uint64
	CompressedLen uint64
	ElementLen    uint64
	TileRoot      hash.Hash32
}

// TensorTiledMeta is a parsed mode-15 descriptor.
type TensorTiledMeta struct {
	Dtype        string
	Shape        []uint64
	ElementCount uint64
	TileShape    []uint64
	TileCounts   []uint64
	Tiles        []TensorTiledEntry
	PayloadLen   uint64
	LogicalRoot  hash.Hash32
}

// TensorTiledDescriptor is a parsed mode-15 frame with its stored payload.
type TensorTiledDescriptor struct {
	Meta    TensorTiledMeta
	Payload []byte
}

func tiledProduct(shape []uint64, lim limits.Limits, what string) (uint64, error) {
	if len(shape) == 0 {
		return 0, werrors.Malformedf("%s shape rank must be at least 1", what)
	}
	if uint64(len(shape)) > lim.MaxTensorRank {
		return 0, werrors.LimitExceededf(
			"%s rank %d exceeds limit %d", what, len(shape), lim.MaxTensorRank)
	}
	product := uint64(1)
	for _, dim := range shape {
		if dim == 0 {
			return 0, werrors.Malformedf("%s shape dimensions must be greater than zero", what)
		}
		if product > math.MaxUint64/dim {
			return 0, werrors.LimitExceededf("%s shape product overflows u64", what)
		}
		product *= dim
	}
	if product > lim.MaxTensorElements {
		return 0, werrors.LimitExceededf(
			"%s element count %d exceeds limit %d", what, product, lim.MaxTensorElements)
	}
	return product, nil
}

func tileCounts(shape, tileShape []uint64) ([]uint64, error) {
	if len(shape) != len(tileShape) {
		return nil, werrors.Malformedf(
			"tensor tile rank %d does not equal shape rank %d", len(tileShape), len(shape))
	}
	counts := make([]uint64, len(shape))
	for axis := range shape {
		if tileShape[axis] == 0 {
			return nil, werrors.Malformedf(
				"tensor tile shape axis %d must be greater than zero", axis)
		}
		if shape[axis]%tileShape[axis] != 0 {
			return nil, werrors.Malformedf(
				"tensor tile shape axis %d extent %d does not divide shape extent %d",
				axis, tileShape[axis], shape[axis])
		}
		counts[axis] = shape[axis] / tileShape[axis]
	}
	return counts, nil
}

func tileCountOf(counts []uint64, lim limits.Limits) (uint64, error) {
	total := uint64(1)
	for _, count := range counts {
		if total > math.MaxUint64/count {
			return 0, werrors.NewLimitExceeded("tensor tile count overflows u64")
		}
		total *= count
	}
	if total > lim.MaxStructuredItems {
		return 0, werrors.LimitExceededf(
			"tensor tile count %d exceeds limit %d", total, lim.MaxStructuredItems)
	}
	return total, nil
}

func tileCoords(counts []uint64, index uint64) []uint64 {
	coords := make([]uint64, len(counts))
	remainder := index
	for axis := len(counts) - 1; axis >= 0; axis-- {
		coords[axis] = remainder % counts[axis]
		remainder /= counts[axis]
	}
	return coords
}

func rowMajorByteStrides(shape []uint64, elementSize uint64) ([]uint64, error) {
	strides := make([]uint64, len(shape))
	stride := elementSize
	for axis := len(shape) - 1; axis >= 0; axis-- {
		strides[axis] = stride
		if shape[axis] != 0 && stride > math.MaxUint64/shape[axis] {
			return nil, werrors.NewLimitExceeded("tensor byte stride overflows u64")
		}
		stride *= shape[axis]
	}
	return strides, nil
}

// copyBoxRows copies a row-major box from src into the row-major dst.
func copyBoxRows(dst, src []byte, srcShape, srcStart, extent []uint64, elementSize uint64) error {
	rank := len(srcShape)
	if rank == 0 || len(extent) != rank || len(srcStart) != rank {
		return werrors.NewMalformed("box copy rank mismatch")
	}
	strides, err := rowMajorByteStrides(srcShape, elementSize)
	if err != nil {
		return err
	}
	rowLenU64 := extent[rank-1] * elementSize
	if extent[rank-1] != 0 && rowLenU64/extent[rank-1] != elementSize {
		return werrors.NewLimitExceeded("box row length overflows u64")
	}
	rowLen := int(rowLenU64)
	if uint64(rowLen) != rowLenU64 {
		return werrors.NewLimitExceeded("box row length does not fit int")
	}
	outer := extent[:rank-1]
	coords := make([]uint64, len(outer))
	dstOff := 0
	for {
		srcOff := uint64(0)
		for axis := 0; axis < rank-1; axis++ {
			position := srcStart[axis] + coords[axis]
			term := position * strides[axis]
			if strides[axis] != 0 && term/strides[axis] != position {
				return werrors.NewLimitExceeded("box source offset overflows u64")
			}
			if srcOff > math.MaxUint64-term {
				return werrors.NewLimitExceeded("box source offset overflows u64")
			}
			srcOff += term
		}
		last := srcStart[rank-1] * strides[rank-1]
		if strides[rank-1] != 0 && last/strides[rank-1] != srcStart[rank-1] {
			return werrors.NewLimitExceeded("box source offset overflows u64")
		}
		if srcOff > math.MaxUint64-last {
			return werrors.NewLimitExceeded("box source offset overflows u64")
		}
		srcOff += last
		if srcOff > uint64(len(src)) || rowLen > len(src)-int(srcOff) {
			return werrors.NewMalformed("box copy is outside its source buffer")
		}
		if dstOff > len(dst) || rowLen > len(dst)-dstOff {
			return werrors.NewMalformed("box copy is outside its destination buffer")
		}
		copy(dst[dstOff:dstOff+rowLen], src[srcOff:srcOff+uint64(rowLen)])
		dstOff += rowLen

		axis := len(outer)
		for {
			if axis == 0 {
				return nil
			}
			axis--
			coords[axis]++
			if coords[axis] < outer[axis] {
				break
			}
			coords[axis] = 0
		}
	}
}

// copyBoxInto scatters a row-major box into a larger row-major destination.
func copyBoxInto(dst []byte, dstExtent []uint64, dstStart []uint64, src []byte, srcExtent []uint64, elementSize uint64) error {
	rank := len(dstExtent)
	if rank == 0 || len(srcExtent) != rank || len(dstStart) != rank {
		return werrors.NewMalformed("box scatter rank mismatch")
	}
	dstStrides := make([]uint64, rank)
	stride := uint64(1)
	for axis := rank - 1; axis >= 0; axis-- {
		dstStrides[axis] = stride
		if stride > math.MaxUint64/dstExtent[axis] {
			return werrors.NewLimitExceeded("destination stride overflows u64")
		}
		stride *= dstExtent[axis]
	}
	rowLen := srcExtent[rank-1] * elementSize
	if srcExtent[rank-1] != 0 && rowLen/srcExtent[rank-1] != elementSize {
		return werrors.NewLimitExceeded("box row length overflows u64")
	}
	outer := srcExtent[:rank-1]
	coords := make([]uint64, len(outer))
	srcOff := 0
	for {
		dstElements := uint64(0)
		for axis := 0; axis < rank-1; axis++ {
			position := dstStart[axis] + coords[axis]
			term := position * dstStrides[axis]
			if dstStrides[axis] != 0 && term/dstStrides[axis] != position {
				return werrors.NewLimitExceeded("destination offset overflows u64")
			}
			if dstElements > math.MaxUint64-term {
				return werrors.NewLimitExceeded("destination offset overflows u64")
			}
			dstElements += term
		}
		if dstElements > math.MaxUint64-dstStart[rank-1] {
			return werrors.NewLimitExceeded("destination offset overflows u64")
		}
		dstElements += dstStart[rank-1]
		dstOff := dstElements * elementSize
		if elementSize != 0 && dstOff/elementSize != dstElements {
			return werrors.NewLimitExceeded("destination byte offset overflows u64")
		}
		if dstOff > uint64(len(dst)) || rowLen > uint64(len(dst))-dstOff {
			return werrors.NewMalformed("box scatter is outside its destination buffer")
		}
		if srcOff > len(src) || int(rowLen) > len(src)-srcOff {
			return werrors.NewMalformed("box scatter is outside its source buffer")
		}
		copy(dst[dstOff:dstOff+rowLen], src[srcOff:srcOff+int(rowLen)])
		srcOff += int(rowLen)

		axis := len(outer)
		for {
			if axis == 0 {
				return nil
			}
			axis--
			coords[axis]++
			if coords[axis] < outer[axis] {
				break
			}
			coords[axis] = 0
		}
	}
}

func encodeTilePlanes(tile []byte, elementSize int) []byte {
	elements := len(tile) / elementSize
	out := make([]byte, 0, len(tile))
	for plane := 0; plane < elementSize; plane++ {
		planeBytes := make([]byte, 0, elements)
		for index := plane; index < len(tile); index += elementSize {
			planeBytes = append(planeBytes, tile[index])
		}
		out = append(out, HuffmanEncode(planeBytes)...)
	}
	return out
}

func decodeTilePlanes(stored []byte, elementSize, elementLen int) ([]byte, error) {
	if elementSize == 0 || elementLen%elementSize != 0 {
		return nil, werrors.NewMalformed(
			"tensor tile element length is not a whole number of elements")
	}
	elements := elementLen / elementSize
	planes := make([][]byte, 0, elementSize)
	cursor := 0
	for plane := 0; plane < elementSize; plane++ {
		if cursor > len(stored) {
			return nil, werrors.NewCorrupt("tensor tile plane is truncated")
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
			"tensor tile planes leave trailing bytes inside the tile range")
	}
	out := make([]byte, elementLen)
	for plane, values := range planes {
		for element, value := range values {
			out[element*elementSize+plane] = value
		}
	}
	return out, nil
}

// TensorTileRoot is
// H("weft/tensor/tile/v1", WCB({dtype, dims, element_len, values_hash})).
func TensorTileRoot(dtype string, dims []uint64, tileBytes []byte, lim limits.Limits) (hash.Hash32, error) {
	dimsItems := make([]wcb.Value, 0, len(dims))
	for _, dim := range dims {
		dimsItems = append(dimsItems, wcb.U64Value(dim))
	}
	valuesHash := hash.Domain(tensor.DomainTensorValues, tileBytes)
	meta := wcb.MapValue(map[string]wcb.Value{
		"dtype":       wcb.TextValue(dtype),
		"dims":        wcb.ArrayValue(dimsItems),
		"element_len": wcb.U64Value(uint64(len(tileBytes))),
		"values_hash": wcb.BytesValue(valuesHash.Bytes()),
	})
	encoded, err := wcb.Encode(meta, lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(hash.DomainTensorTile, encoded), nil
}

func tiledMetaValue(meta TensorTiledMeta) wcb.Value {
	tiles := make([]wcb.Value, 0, len(meta.Tiles))
	for _, entry := range meta.Tiles {
		dims := make([]wcb.Value, 0, len(entry.Dims))
		for _, dim := range entry.Dims {
			dims = append(dims, wcb.U64Value(dim))
		}
		tiles = append(tiles, wcb.MapValue(map[string]wcb.Value{
			"index":          wcb.U64Value(entry.Index),
			"dims":           wcb.ArrayValue(dims),
			"offset":         wcb.U64Value(entry.Offset),
			"compressed_len": wcb.U64Value(entry.CompressedLen),
			"element_len":    wcb.U64Value(entry.ElementLen),
			"tile_root":      wcb.BytesValue(entry.TileRoot.Bytes()),
		}))
	}
	shape := make([]wcb.Value, 0, len(meta.Shape))
	for _, dim := range meta.Shape {
		shape = append(shape, wcb.U64Value(dim))
	}
	tileShape := make([]wcb.Value, 0, len(meta.TileShape))
	for _, dim := range meta.TileShape {
		tileShape = append(tileShape, wcb.U64Value(dim))
	}
	counts := make([]wcb.Value, 0, len(meta.TileCounts))
	for _, count := range meta.TileCounts {
		counts = append(counts, wcb.U64Value(count))
	}
	return wcb.MapValue(map[string]wcb.Value{
		"schema":        wcb.TextValue(TensorTiledSchema),
		"dtype":         wcb.TextValue(meta.Dtype),
		"shape":         wcb.ArrayValue(shape),
		"element_count": wcb.U64Value(meta.ElementCount),
		"layout":        wcb.TextValue(TensorTileLayout),
		"tile_shape":    wcb.ArrayValue(tileShape),
		"tile_counts":   wcb.ArrayValue(counts),
		"tile_codec":    wcb.TextValue(HuffmanCoderID),
		"tiles":         wcb.ArrayValue(tiles),
		"payload_len":   wcb.U64Value(meta.PayloadLen),
		"logical_root":  wcb.BytesValue(meta.LogicalRoot.Bytes()),
	})
}

// EncodeTiledTensor encodes a mode-15 TENSOR_TILED frame.
func EncodeTiledTensor(payload []byte, dtype string, shape, tileShape []uint64, lim limits.Limits) ([]byte, error) {
	if err := limits.CheckLen(uint64(len(payload)), lim.MaxDecodedBytes, "tiled tensor payload"); err != nil {
		return nil, err
	}
	elementSize, err := tensorElementSize(dtype)
	if err != nil {
		return nil, err
	}
	elementCount, err := tiledProduct(shape, lim, "tiled tensor")
	if err != nil {
		return nil, err
	}
	if elementCount != 0 && elementSize > math.MaxUint64/elementCount {
		return nil, werrors.NewLimitExceeded("tiled tensor decoded bytes overflow u64")
	}
	decodedBytes := elementCount * elementSize
	if decodedBytes != uint64(len(payload)) {
		return nil, werrors.Malformedf(
			"tiled tensor payload is %d bytes but shape %v x %s requires %d",
			len(payload), shape, dtype, decodedBytes)
	}
	counts, err := tileCounts(shape, tileShape)
	if err != nil {
		return nil, err
	}
	tileElements, err := tiledProduct(tileShape, lim, "tensor tile")
	if err != nil {
		return nil, err
	}
	if tileElements != 0 && elementSize > math.MaxUint64/tileElements {
		return nil, werrors.NewLimitExceeded("tensor tile byte length overflows u64")
	}
	tileElementLen := tileElements * elementSize
	tileCount, err := tileCountOf(counts, lim)
	if err != nil {
		return nil, err
	}
	if uint64(int(tileElementLen)) != tileElementLen {
		return nil, werrors.NewLimitExceeded("tensor tile byte length does not fit int")
	}
	stored := make([]byte, 0, len(payload))
	tiles := make([]TensorTiledEntry, 0, int(tileCount))
	for index := uint64(0); index < tileCount; index++ {
		coords := tileCoords(counts, index)
		startBytes := make([]uint64, len(coords))
		for axis := range coords {
			startBytes[axis] = coords[axis] * tileShape[axis]
		}
		tile := make([]byte, int(tileElementLen))
		if err := copyBoxRows(tile, payload, shape, startBytes, tileShape, elementSize); err != nil {
			return nil, err
		}
		encoded := encodeTilePlanes(tile, int(elementSize))
		root, err := TensorTileRoot(dtype, tileShape, tile, lim)
		if err != nil {
			return nil, err
		}
		tiles = append(tiles, TensorTiledEntry{
			Index:         index,
			Dims:          append([]uint64(nil), tileShape...),
			Offset:        uint64(len(stored)),
			CompressedLen: uint64(len(encoded)),
			ElementLen:    tileElementLen,
			TileRoot:      root,
		})
		stored = append(stored, encoded...)
	}
	logicalRoot, err := tensor.LogicalRoot(dtype, shape, payload, lim)
	if err != nil {
		return nil, err
	}
	meta := TensorTiledMeta{
		Dtype:        dtype,
		Shape:        append([]uint64(nil), shape...),
		ElementCount: elementCount,
		TileShape:    append([]uint64(nil), tileShape...),
		TileCounts:   counts,
		Tiles:        tiles,
		PayloadLen:   uint64(len(stored)),
		LogicalRoot:  logicalRoot,
	}
	body, err := buildAIBody(tiledMetaValue(meta), stored, lim)
	if err != nil {
		return nil, err
	}
	return frame.Encode(frame.PayloadParts(
		frame.ModeTensorTiled, 0, 0, decodedBytes, hash.Domain(hash.DomainChunk, payload), body), lim)
}

// ParseTiledTensor parses a mode-15 descriptor with its stored payload.
func ParseTiledTensor(f *frame.Frame, lim limits.Limits) (TensorTiledDescriptor, error) {
	if f.Header.Mode != frame.ModeTensorTiled {
		return TensorTiledDescriptor{}, werrors.Malformedf(
			"frame mode %s is not tensor_tiled", f.Header.Mode)
	}
	descriptorBytes, payload, err := splitAIBodyPrefix(f.Body)
	if err != nil {
		return TensorTiledDescriptor{}, err
	}
	value, err := wcb.Decode(descriptorBytes, lim)
	if err != nil {
		return TensorTiledDescriptor{}, err
	}
	if value.Kind != wcb.Map {
		return TensorTiledDescriptor{}, werrors.NewMalformed(
			"tiled tensor metadata must be a WCB map")
	}
	m := value.Map
	if err := checkExactKeys(m, []string{
		"schema", "dtype", "shape", "element_count", "layout", "tile_shape",
		"tile_counts", "tile_codec", "tiles", "payload_len", "logical_root",
	}, "tiled tensor"); err != nil {
		return TensorTiledDescriptor{}, err
	}
	schema, err := metaText(m, "schema")
	if err != nil {
		return TensorTiledDescriptor{}, err
	}
	if schema != TensorTiledSchema {
		return TensorTiledDescriptor{}, werrors.Malformedf(
			"unsupported tiled tensor schema %q", schema)
	}
	dtype, err := metaText(m, "dtype")
	if err != nil {
		return TensorTiledDescriptor{}, err
	}
	elementSize, err := tensorElementSize(dtype)
	if err != nil {
		return TensorTiledDescriptor{}, err
	}
	layout, err := metaText(m, "layout")
	if err != nil {
		return TensorTiledDescriptor{}, err
	}
	if layout != TensorTileLayout {
		return TensorTiledDescriptor{}, werrors.Malformedf(
			"tiled tensor layout %q is specified but not implemented in v1 (only %s)",
			layout, TensorTileLayout)
	}
	codec, err := metaText(m, "tile_codec")
	if err != nil {
		return TensorTiledDescriptor{}, err
	}
	if codec != HuffmanCoderID {
		return TensorTiledDescriptor{}, werrors.Malformedf(
			"unsupported tensor tile codec %q (v1 allows %s)", codec, HuffmanCoderID)
	}
	shape, err := metaShape(m)
	if err != nil {
		return TensorTiledDescriptor{}, err
	}
	elementCount, err := tiledProduct(shape, lim, "tiled tensor")
	if err != nil {
		return TensorTiledDescriptor{}, err
	}
	declaredCount, err := metaU64(m, "element_count")
	if err != nil {
		return TensorTiledDescriptor{}, err
	}
	if declaredCount != elementCount {
		return TensorTiledDescriptor{}, werrors.Malformedf(
			"tiled tensor element_count %d does not match shape product %d",
			declaredCount, elementCount)
	}
	decodedBytes := elementCount * elementSize
	if elementCount != 0 && elementSize > math.MaxUint64/elementCount {
		return TensorTiledDescriptor{}, werrors.NewLimitExceeded(
			"tiled tensor decoded bytes overflow u64")
	}
	if err := limits.CheckLen(decodedBytes, lim.MaxDecodedBytes, "tiled tensor decoded bytes"); err != nil {
		return TensorTiledDescriptor{}, err
	}
	if f.Header.DecodedLen != decodedBytes {
		return TensorTiledDescriptor{}, werrors.Corruptf(
			"tiled tensor decoded_len is %d but the descriptor requires %d",
			f.Header.DecodedLen, decodedBytes)
	}
	tileShape, err := metaShapeKey(m, "tile_shape")
	if err != nil {
		return TensorTiledDescriptor{}, err
	}
	counts, err := tileCounts(shape, tileShape)
	if err != nil {
		return TensorTiledDescriptor{}, err
	}
	declaredCounts, err := metaShapeKey(m, "tile_counts")
	if err != nil {
		return TensorTiledDescriptor{}, err
	}
	if !u64SlicesEqual(declaredCounts, counts) {
		return TensorTiledDescriptor{}, werrors.Malformedf(
			"tiled tensor tile_counts %v do not match shape/tile_shape %v",
			declaredCounts, counts)
	}
	tileElements, err := tiledProduct(tileShape, lim, "tensor tile")
	if err != nil {
		return TensorTiledDescriptor{}, err
	}
	if tileElements != 0 && elementSize > math.MaxUint64/tileElements {
		return TensorTiledDescriptor{}, werrors.NewLimitExceeded(
			"tensor tile byte length overflows u64")
	}
	tileElementLen := tileElements * elementSize
	tileCount, err := tileCountOf(counts, lim)
	if err != nil {
		return TensorTiledDescriptor{}, err
	}
	declaredPayloadLen, err := metaU64(m, "payload_len")
	if err != nil {
		return TensorTiledDescriptor{}, err
	}
	if declaredPayloadLen != uint64(len(payload)) {
		return TensorTiledDescriptor{}, werrors.Malformedf(
			"tiled tensor payload is %d bytes but metadata declares %d",
			len(payload), declaredPayloadLen)
	}
	tilesValue, ok := m["tiles"]
	if !ok || tilesValue.Kind != wcb.Array {
		return TensorTiledDescriptor{}, werrors.NewMalformed(
			"tiled tensor \"tiles\" must be an array")
	}
	if uint64(len(tilesValue.Array)) != tileCount {
		return TensorTiledDescriptor{}, werrors.Malformedf(
			"tiled tensor has %d tile entries but the grid requires %d",
			len(tilesValue.Array), tileCount)
	}
	tiles := make([]TensorTiledEntry, 0, len(tilesValue.Array))
	expectedOffset := uint64(0)
	for position, item := range tilesValue.Array {
		if item.Kind != wcb.Map {
			return TensorTiledDescriptor{}, werrors.NewMalformed(
				"tensor tile entry must be a WCB map")
		}
		entry := item.Map
		if err := checkExactKeys(entry, []string{
			"index", "dims", "offset", "compressed_len", "element_len", "tile_root",
		}, "tensor tile"); err != nil {
			return TensorTiledDescriptor{}, err
		}
		index, err := metaU64(entry, "index")
		if err != nil {
			return TensorTiledDescriptor{}, err
		}
		if index != uint64(position) {
			return TensorTiledDescriptor{}, werrors.Malformedf(
				"tiled tensor entry %d declares index %d; entries must be in canonical order",
				position, index)
		}
		dims, err := metaShapeKey(entry, "dims")
		if err != nil {
			return TensorTiledDescriptor{}, err
		}
		if !u64SlicesEqual(dims, tileShape) {
			return TensorTiledDescriptor{}, werrors.Malformedf(
				"tensor tile %d dims %v do not equal tile_shape %v; partial tiles are not defined in v1",
				position, dims, tileShape)
		}
		elementLen, err := metaU64(entry, "element_len")
		if err != nil {
			return TensorTiledDescriptor{}, err
		}
		if elementLen != tileElementLen {
			return TensorTiledDescriptor{}, werrors.Malformedf(
				"tensor tile %d element_len %d does not match %d",
				position, elementLen, tileElementLen)
		}
		offset, err := metaU64(entry, "offset")
		if err != nil {
			return TensorTiledDescriptor{}, err
		}
		if offset != expectedOffset {
			return TensorTiledDescriptor{}, werrors.Malformedf(
				"tensor tile %d offset %d is not the contiguous offset %d",
				position, offset, expectedOffset)
		}
		compressedLen, err := metaU64(entry, "compressed_len")
		if err != nil {
			return TensorTiledDescriptor{}, err
		}
		if compressedLen == 0 {
			return TensorTiledDescriptor{}, werrors.Malformedf(
				"tensor tile %d has an empty stored range", position)
		}
		if expectedOffset > math.MaxUint64-compressedLen {
			return TensorTiledDescriptor{}, werrors.NewLimitExceeded(
				"tensor tile offset overflows u64")
		}
		expectedOffset += compressedLen
		rootBytes, err := metaBytes(entry, "tile_root")
		if err != nil {
			return TensorTiledDescriptor{}, err
		}
		root, err := hash.FromSlice(rootBytes)
		if err != nil {
			return TensorTiledDescriptor{}, werrors.NewMalformed(
				"tensor tile_root must be 32 bytes")
		}
		tiles = append(tiles, TensorTiledEntry{
			Index:         index,
			Dims:          dims,
			Offset:        offset,
			CompressedLen: compressedLen,
			ElementLen:    elementLen,
			TileRoot:      root,
		})
	}
	if expectedOffset != declaredPayloadLen {
		return TensorTiledDescriptor{}, werrors.Malformedf(
			"tensor tile ranges cover %d bytes but payload_len is %d",
			expectedOffset, declaredPayloadLen)
	}
	logicalBytes, err := metaBytes(m, "logical_root")
	if err != nil {
		return TensorTiledDescriptor{}, err
	}
	logicalRoot, err := hash.FromSlice(logicalBytes)
	if err != nil {
		return TensorTiledDescriptor{}, werrors.NewMalformed(
			"tiled tensor logical_root must be 32 bytes")
	}
	return TensorTiledDescriptor{
		Meta: TensorTiledMeta{
			Dtype:        dtype,
			Shape:        shape,
			ElementCount: elementCount,
			TileShape:    tileShape,
			TileCounts:   counts,
			Tiles:        tiles,
			PayloadLen:   declaredPayloadLen,
			LogicalRoot:  logicalRoot,
		},
		Payload: payload,
	}, nil
}

func metaShapeKey(m map[string]wcb.Value, key string) ([]uint64, error) {
	value, ok := m[key]
	if !ok || value.Kind != wcb.Array {
		return nil, werrors.Malformedf("metadata %q must be an array", key)
	}
	out := make([]uint64, 0, len(value.Array))
	for _, item := range value.Array {
		if item.Kind != wcb.U64 {
			return nil, werrors.Malformedf("metadata %q elements must be u64", key)
		}
		out = append(out, item.U)
	}
	return out, nil
}

func u64SlicesEqual(left, right []uint64) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

// TiledTensorMetadata parses and returns only the descriptor.
func TiledTensorMetadata(f *frame.Frame, lim limits.Limits) (TensorTiledMeta, error) {
	descriptor, err := ParseTiledTensor(f, lim)
	if err != nil {
		return TensorTiledMeta{}, err
	}
	return descriptor.Meta, nil
}

// DecodeTiledTensor validates and decodes a mode-15 frame into canonical
// row-major bytes. Every tile root and the whole-tensor logical root are
// recomputed and matched.
func DecodeTiledTensor(f *frame.Frame, lim limits.Limits) ([]byte, error) {
	descriptor, err := ParseTiledTensor(f, lim)
	if err != nil {
		return nil, err
	}
	meta := descriptor.Meta
	elementSize, err := tensorElementSize(meta.Dtype)
	if err != nil {
		return nil, err
	}
	total := meta.ElementCount * elementSize
	if meta.ElementCount != 0 && elementSize > math.MaxUint64/meta.ElementCount {
		return nil, werrors.NewLimitExceeded("tiled tensor decoded bytes overflow u64")
	}
	if uint64(int(total)) != total {
		return nil, werrors.NewLimitExceeded("tiled tensor decoded bytes do not fit int")
	}
	out := make([]byte, int(total))
	tilesBytes := 0
	for _, entry := range meta.Tiles {
		start := entry.Offset
		end := start + entry.CompressedLen
		if end < start || end > uint64(len(descriptor.Payload)) {
			return nil, werrors.NewCorrupt("tensor tile range is outside the payload")
		}
		if entry.ElementLen > uint64(int(^uint(0)>>1)) {
			return nil, werrors.NewLimitExceeded("tensor tile length does not fit int")
		}
		tile, err := decodeTilePlanes(
			descriptor.Payload[start:end], int(elementSize), int(entry.ElementLen))
		if err != nil {
			return nil, err
		}
		root, err := TensorTileRoot(meta.Dtype, entry.Dims, tile, lim)
		if err != nil {
			return nil, err
		}
		if root != entry.TileRoot {
			return nil, werrors.Corruptf(
				"tensor tile %d root %s does not match the descriptor's %s",
				entry.Index, root, entry.TileRoot)
		}
		tilesBytes += len(tile)
		coords := tileCoords(meta.TileCounts, entry.Index)
		dstStart := make([]uint64, len(meta.Shape))
		for axis := range meta.Shape {
			dstStart[axis] = coords[axis] * meta.TileShape[axis]
		}
		if err := copyBoxInto(out, meta.Shape, dstStart, tile, entry.Dims, elementSize); err != nil {
			return nil, err
		}
	}
	if uint64(tilesBytes) != total {
		return nil, werrors.Corruptf(
			"tiled tensor decoded to %d bytes but the descriptor requires %d",
			tilesBytes, total)
	}
	logical, err := tensor.LogicalRoot(meta.Dtype, meta.Shape, out, lim)
	if err != nil {
		return nil, err
	}
	if logical != meta.LogicalRoot {
		return nil, werrors.Corruptf(
			"tiled tensor logical root %s does not match the descriptor's %s",
			logical, meta.LogicalRoot)
	}
	return out, nil
}

// ReadRegion reads one logical region, decoding only the tiles it intersects.
// It returns the region's canonical row-major bytes and the number of tiles
// decoded (whose roots were verified).
func ReadRegion(descriptor TensorTiledDescriptor, region tensor.Region, lim limits.Limits) ([]byte, uint64, error) {
	meta := descriptor.Meta
	if err := region.Validate(meta.Shape); err != nil {
		return nil, 0, err
	}
	extent := region.Extent()
	if region.IsEmpty() {
		return []byte{}, 0, nil
	}
	elementSize, err := tensorElementSize(meta.Dtype)
	if err != nil {
		return nil, 0, err
	}
	count := uint64(1)
	for _, dim := range extent {
		if count > math.MaxUint64/dim {
			return nil, 0, werrors.NewLimitExceeded("region element count overflows u64")
		}
		count *= dim
	}
	byteLen := count * elementSize
	if count != 0 && elementSize > math.MaxUint64/count {
		return nil, 0, werrors.NewLimitExceeded("region payload length overflows u64")
	}
	if err := limits.CheckLen(byteLen, lim.MaxDecodedBytes, "region payload"); err != nil {
		return nil, 0, err
	}
	if uint64(int(byteLen)) != byteLen {
		return nil, 0, werrors.NewLimitExceeded("region payload does not fit int")
	}
	out := make([]byte, int(byteLen))

	first := make([]uint64, len(meta.Shape))
	last := make([]uint64, len(meta.Shape))
	for axis := range meta.Shape {
		tile := meta.TileShape[axis]
		first[axis] = region.Start[axis] / tile
		last[axis] = (region.End[axis] + tile - 1) / tile
	}
	var tilesTouched uint64
	coords := append([]uint64(nil), first...)
	for {
		index := uint64(0)
		for axis := range meta.TileCounts {
			if index > math.MaxUint64/meta.TileCounts[axis] {
				return nil, 0, werrors.NewLimitExceeded("tensor tile index overflows u64")
			}
			index = index*meta.TileCounts[axis] + coords[axis]
		}
		if index >= uint64(len(meta.Tiles)) {
			return nil, 0, werrors.NewCorrupt(
				"tensor tile index is outside the descriptor")
		}
		entry := meta.Tiles[index]
		start := entry.Offset
		end := start + entry.CompressedLen
		if end < start || end > uint64(len(descriptor.Payload)) {
			return nil, 0, werrors.NewCorrupt("tensor tile range is outside the payload")
		}
		tile, err := decodeTilePlanes(
			descriptor.Payload[start:end], int(elementSize), int(entry.ElementLen))
		if err != nil {
			return nil, 0, err
		}
		root, err := TensorTileRoot(meta.Dtype, entry.Dims, tile, lim)
		if err != nil {
			return nil, 0, err
		}
		if root != entry.TileRoot {
			return nil, 0, werrors.Corruptf(
				"tensor tile %d root %s does not match the descriptor's %s",
				index, root, entry.TileRoot)
		}
		tilesTouched++

		localStart := make([]uint64, len(meta.Shape))
		localExtent := make([]uint64, len(meta.Shape))
		dstStart := make([]uint64, len(meta.Shape))
		for axis := range meta.Shape {
			base := coords[axis] * meta.TileShape[axis]
			lo := region.Start[axis]
			if lo < base {
				lo = base
			}
			hi := region.End[axis]
			if limit := base + meta.TileShape[axis]; hi > limit {
				hi = limit
			}
			localStart[axis] = lo - base
			localExtent[axis] = hi - lo
			dstStart[axis] = lo - region.Start[axis]
		}
		boxLen := uint64(1)
		for _, dim := range localExtent {
			boxLen *= dim
		}
		boxed := make([]byte, int(boxLen*elementSize))
		if err := copyBoxRows(boxed, tile, meta.TileShape, localStart, localExtent, elementSize); err != nil {
			return nil, 0, err
		}
		if err := copyBoxInto(out, extent, dstStart, boxed, localExtent, elementSize); err != nil {
			return nil, 0, err
		}

		axis := len(first)
		for {
			if axis == 0 {
				return out, tilesTouched, nil
			}
			axis--
			coords[axis]++
			if coords[axis] < last[axis] {
				break
			}
			coords[axis] = first[axis]
		}
	}
}
