package modes

// FLOAT_TILED (mode 17, WEFT-AI-CODEC §18).
//
// A tile grid where every tile is an exact stand-alone mode-9 split: a
// `huffman-canonical-v1` exponent plane plus a raw sign+mantissa plane,
// computed per tile. Tiles are stored contiguously in canonical row-major
// order with explicit offsets, so a region read decodes only the tiles it
// intersects, and a full decode verifies every tile root and the logical root.

import (
	"math"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// FloatTiledSchema is the canonical mode-17 descriptor schema.
const FloatTiledSchema = "weft.ai.float_tiled.v1"

// FloatTileLayout is the only float-tile layout implemented in v1.
const FloatTileLayout = "row_major"

// FloatTiledEntry is one independently decodable float tile.
type FloatTiledEntry struct {
	Index            uint64
	Dims             []uint64
	Offset           uint64
	CompressedLen    uint64
	ExponentPlaneLen uint64
	MantissaPlaneLen uint64
	ElementLen       uint64
	TileRoot         hash.Hash32
}

// FloatTiledMeta is a parsed mode-17 descriptor.
type FloatTiledMeta struct {
	Dtype        FloatDtype
	Shape        []uint64
	ElementCount uint64
	TileShape    []uint64
	TileCounts   []uint64
	Tiles        []FloatTiledEntry
	PayloadLen   uint64
	LogicalRoot  hash.Hash32
}

// FloatTiledDescriptor is a parsed mode-17 frame with its stored payload.
type FloatTiledDescriptor struct {
	Meta    FloatTiledMeta
	Payload []byte
}

// floatTiledTensorDtype returns the tensor-identity dtype name, or false when
// the float dtype has no logical-tensor layout in v1 (fp8 has none).
func floatTiledTensorDtype(dtype FloatDtype) (string, bool) {
	switch dtype.Name {
	case "bf16":
		return "bf16", true
	case "fp16":
		return "f16", true
	case "fp32":
		return "f32", true
	default:
		return "", false
	}
}

// FloatTileRoot is
// H("weft/tensor/float_tile/v1",
//
//	WCB({dtype, dims, element_len, exponent_plane_len, mantissa_plane_len,
//	      values_hash})).
func FloatTileRoot(
	dtype string, dims []uint64, tileBytes []byte,
	exponentPlaneLen, mantissaPlaneLen uint64, lim limits.Limits,
) (hash.Hash32, error) {
	dimsItems := make([]wcb.Value, 0, len(dims))
	for _, dim := range dims {
		dimsItems = append(dimsItems, wcb.U64Value(dim))
	}
	valuesHash := hash.Domain(tensor.DomainTensorValues, tileBytes)
	meta := wcb.MapValue(map[string]wcb.Value{
		"dtype":              wcb.TextValue(dtype),
		"dims":               wcb.ArrayValue(dimsItems),
		"element_len":        wcb.U64Value(uint64(len(tileBytes))),
		"exponent_plane_len": wcb.U64Value(exponentPlaneLen),
		"mantissa_plane_len": wcb.U64Value(mantissaPlaneLen),
		"values_hash":        wcb.BytesValue(valuesHash.Bytes()),
	})
	encoded, err := wcb.Encode(meta, lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(hash.DomainFloatTile, encoded), nil
}

func floatTiledMetaValue(meta FloatTiledMeta) wcb.Value {
	tiles := make([]wcb.Value, 0, len(meta.Tiles))
	for _, entry := range meta.Tiles {
		dims := make([]wcb.Value, 0, len(entry.Dims))
		for _, dim := range entry.Dims {
			dims = append(dims, wcb.U64Value(dim))
		}
		tiles = append(tiles, wcb.MapValue(map[string]wcb.Value{
			"index":              wcb.U64Value(entry.Index),
			"dims":               wcb.ArrayValue(dims),
			"offset":             wcb.U64Value(entry.Offset),
			"compressed_len":     wcb.U64Value(entry.CompressedLen),
			"exponent_plane_len": wcb.U64Value(entry.ExponentPlaneLen),
			"mantissa_plane_len": wcb.U64Value(entry.MantissaPlaneLen),
			"element_len":        wcb.U64Value(entry.ElementLen),
			"tile_root":          wcb.BytesValue(entry.TileRoot.Bytes()),
		}))
	}
	shape := u64Values(meta.Shape)
	tileShape := u64Values(meta.TileShape)
	counts := u64Values(meta.TileCounts)
	return wcb.MapValue(map[string]wcb.Value{
		"schema":         wcb.TextValue(FloatTiledSchema),
		"dtype":          wcb.TextValue(meta.Dtype.Name),
		"shape":          wcb.ArrayValue(shape),
		"element_count":  wcb.U64Value(meta.ElementCount),
		"layout":         wcb.TextValue(FloatTileLayout),
		"tile_shape":     wcb.ArrayValue(tileShape),
		"tile_counts":    wcb.ArrayValue(counts),
		"exponent_coder": wcb.TextValue(HuffmanCoderID),
		"tiles":          wcb.ArrayValue(tiles),
		"payload_len":    wcb.U64Value(meta.PayloadLen),
		"logical_root":   wcb.BytesValue(meta.LogicalRoot.Bytes()),
	})
}

func u64Values(items []uint64) []wcb.Value {
	out := make([]wcb.Value, 0, len(items))
	for _, item := range items {
		out = append(out, wcb.U64Value(item))
	}
	return out
}

// EncodeFloatTiled encodes a mode-17 FLOAT_TILED frame.
func EncodeFloatTiled(payload []byte, dtype FloatDtype, shape, tileShape []uint64, lim limits.Limits) ([]byte, error) {
	tensorDtypeName, ok := floatTiledTensorDtype(dtype)
	if !ok {
		return nil, werrors.UnsupportedFeaturef(
			"float-tiled v1 requires a tensor identity layout; dtype %q has none", dtype.Name)
	}
	if err := limits.CheckLen(uint64(len(payload)), lim.MaxDecodedBytes, "float tiled payload"); err != nil {
		return nil, err
	}
	elementCount, err := floatTensorProduct(shape, lim)
	if err != nil {
		return nil, err
	}
	elementSize := dtype.ElementSize()
	if elementCount != 0 && elementSize > math.MaxUint64/elementCount {
		return nil, werrors.NewLimitExceeded("float tiled decoded bytes overflow u64")
	}
	decodedBytes := elementCount * elementSize
	if decodedBytes != uint64(len(payload)) {
		return nil, werrors.Malformedf(
			"float tiled payload is %d bytes but shape %v x %s requires %d",
			len(payload), shape, dtype.Name, decodedBytes)
	}
	counts, err := tileCounts(shape, tileShape)
	if err != nil {
		return nil, err
	}
	tileElements, err := tiledProduct(tileShape, lim, "float tile")
	if err != nil {
		return nil, err
	}
	if tileElements != 0 && elementSize > math.MaxUint64/tileElements {
		return nil, werrors.NewLimitExceeded("float tile byte length overflows u64")
	}
	tileElementLen := tileElements * elementSize
	tileCount, err := tileCountOf(counts, lim)
	if err != nil {
		return nil, err
	}
	if uint64(int(tileElementLen)) != tileElementLen {
		return nil, werrors.NewLimitExceeded("float tile byte length does not fit int")
	}
	stored := make([]byte, 0, len(payload))
	tiles := make([]FloatTiledEntry, 0, int(tileCount))
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
		exponentPlane, mantissaPlane, err := SplitFloatTensorPayload(tile, dtype, tileElements)
		if err != nil {
			return nil, err
		}
		root, err := FloatTileRoot(
			dtype.Name, tileShape, tile, uint64(len(exponentPlane)), uint64(len(mantissaPlane)), lim)
		if err != nil {
			return nil, err
		}
		compressedLen := uint64(len(exponentPlane) + len(mantissaPlane))
		tiles = append(tiles, FloatTiledEntry{
			Index:            index,
			Dims:             append([]uint64(nil), tileShape...),
			Offset:           uint64(len(stored)),
			CompressedLen:    compressedLen,
			ExponentPlaneLen: uint64(len(exponentPlane)),
			MantissaPlaneLen: uint64(len(mantissaPlane)),
			ElementLen:       tileElementLen,
			TileRoot:         root,
		})
		stored = append(stored, exponentPlane...)
		stored = append(stored, mantissaPlane...)
	}
	logicalRoot, err := tensor.LogicalRoot(tensorDtypeName, shape, payload, lim)
	if err != nil {
		return nil, err
	}
	meta := FloatTiledMeta{
		Dtype:        dtype,
		Shape:        append([]uint64(nil), shape...),
		ElementCount: elementCount,
		TileShape:    append([]uint64(nil), tileShape...),
		TileCounts:   counts,
		Tiles:        tiles,
		PayloadLen:   uint64(len(stored)),
		LogicalRoot:  logicalRoot,
	}
	body, err := buildAIBody(floatTiledMetaValue(meta), stored, lim)
	if err != nil {
		return nil, err
	}
	return frame.Encode(frame.PayloadParts(
		frame.ModeFloatTiled, 0, 0, decodedBytes, hash.Domain(hash.DomainChunk, payload), body), lim)
}

// ParseFloatTiled parses a mode-17 frame with its stored payload slice.
func ParseFloatTiled(f *frame.Frame, lim limits.Limits) (FloatTiledDescriptor, error) {
	if f.Header.Mode != frame.ModeFloatTiled {
		return FloatTiledDescriptor{}, werrors.Malformedf(
			"frame mode %s is not float_tiled", f.Header.Mode)
	}
	descriptorBytes, payload, err := splitAIBodyPrefix(f.Body)
	if err != nil {
		return FloatTiledDescriptor{}, err
	}
	meta, err := ParseFloatTiledDescriptor(descriptorBytes, f.Header.DecodedLen, uint64(len(payload)), lim)
	if err != nil {
		return FloatTiledDescriptor{}, err
	}
	return FloatTiledDescriptor{Meta: meta, Payload: payload}, nil
}

// ParseFloatTiledDescriptor parses a mode-17 descriptor from canonical WCB
// bytes. This is the prefix form used by a range reader that never
// materialises the whole frame.
func ParseFloatTiledDescriptor(
	descriptorBytes []byte, decodedLen, payloadLen uint64, lim limits.Limits,
) (FloatTiledMeta, error) {
	value, err := wcb.Decode(descriptorBytes, lim)
	if err != nil {
		return FloatTiledMeta{}, err
	}
	if value.Kind != wcb.Map {
		return FloatTiledMeta{}, werrors.NewMalformed(
			"float tiled metadata must be a WCB map")
	}
	m := value.Map
	if err := checkExactKeys(m, []string{
		"schema", "dtype", "shape", "element_count", "layout", "tile_shape",
		"tile_counts", "exponent_coder", "tiles", "payload_len", "logical_root",
	}, "float tiled"); err != nil {
		return FloatTiledMeta{}, err
	}
	schema, err := metaText(m, "schema")
	if err != nil {
		return FloatTiledMeta{}, err
	}
	if schema != FloatTiledSchema {
		return FloatTiledMeta{}, werrors.Malformedf(
			"unsupported float tiled schema %q", schema)
	}
	dtypeName, err := metaText(m, "dtype")
	if err != nil {
		return FloatTiledMeta{}, err
	}
	dtype, err := ParseFloatDtype(dtypeName)
	if err != nil {
		return FloatTiledMeta{}, err
	}
	if _, ok := floatTiledTensorDtype(dtype); !ok {
		return FloatTiledMeta{}, werrors.UnsupportedFeaturef(
			"float-tiled v1 requires a tensor identity layout; dtype %q has none", dtype.Name)
	}
	layout, err := metaText(m, "layout")
	if err != nil {
		return FloatTiledMeta{}, err
	}
	if layout != FloatTileLayout {
		return FloatTiledMeta{}, werrors.Malformedf(
			"float tiled layout %q is specified but not implemented in v1 (only %s)",
			layout, FloatTileLayout)
	}
	coder, err := metaText(m, "exponent_coder")
	if err != nil {
		return FloatTiledMeta{}, err
	}
	if coder != HuffmanCoderID {
		return FloatTiledMeta{}, werrors.Malformedf(
			"unsupported float tile exponent coder %q (v1 allows %s)", coder, HuffmanCoderID)
	}
	shape, err := metaShape(m)
	if err != nil {
		return FloatTiledMeta{}, err
	}
	elementCount, err := floatTensorProduct(shape, lim)
	if err != nil {
		return FloatTiledMeta{}, err
	}
	declaredCount, err := metaU64(m, "element_count")
	if err != nil {
		return FloatTiledMeta{}, err
	}
	if declaredCount != elementCount {
		return FloatTiledMeta{}, werrors.Malformedf(
			"float tiled element_count %d does not match shape product %d",
			declaredCount, elementCount)
	}
	elementSize := dtype.ElementSize()
	if elementCount != 0 && elementSize > math.MaxUint64/elementCount {
		return FloatTiledMeta{}, werrors.NewLimitExceeded(
			"float tiled decoded bytes overflow u64")
	}
	decodedBytes := elementCount * elementSize
	if err := limits.CheckLen(decodedBytes, lim.MaxDecodedBytes, "float tiled decoded bytes"); err != nil {
		return FloatTiledMeta{}, err
	}
	if decodedLen != decodedBytes {
		return FloatTiledMeta{}, werrors.Corruptf(
			"float tiled decoded_len is %d but the descriptor requires %d",
			decodedLen, decodedBytes)
	}
	tileShape, err := metaShapeKey(m, "tile_shape")
	if err != nil {
		return FloatTiledMeta{}, err
	}
	counts, err := tileCounts(shape, tileShape)
	if err != nil {
		return FloatTiledMeta{}, err
	}
	declaredCounts, err := metaShapeKey(m, "tile_counts")
	if err != nil {
		return FloatTiledMeta{}, err
	}
	if !u64SlicesEqual(declaredCounts, counts) {
		return FloatTiledMeta{}, werrors.Malformedf(
			"float tiled tile_counts %v do not match shape/tile_shape %v",
			declaredCounts, counts)
	}
	tileElements, err := tiledProduct(tileShape, lim, "float tile")
	if err != nil {
		return FloatTiledMeta{}, err
	}
	if tileElements != 0 && elementSize > math.MaxUint64/tileElements {
		return FloatTiledMeta{}, werrors.NewLimitExceeded(
			"float tile byte length overflows u64")
	}
	tileElementLen := tileElements * elementSize
	tileCount, err := tileCountOf(counts, lim)
	if err != nil {
		return FloatTiledMeta{}, err
	}
	mantissaBits := dtype.SignBits + dtype.MantissaBits
	if tileElements != 0 && mantissaBits > math.MaxUint64/tileElements {
		return FloatTiledMeta{}, werrors.NewLimitExceeded(
			"float tile mantissa bits overflow u64")
	}
	expectedMantissa := (tileElements*mantissaBits + 7) / 8
	declaredPayloadLen, err := metaU64(m, "payload_len")
	if err != nil {
		return FloatTiledMeta{}, err
	}
	if declaredPayloadLen != payloadLen {
		return FloatTiledMeta{}, werrors.Malformedf(
			"float tiled payload is %d bytes but metadata declares %d",
			payloadLen, declaredPayloadLen)
	}
	tilesValue, ok := m["tiles"]
	if !ok || tilesValue.Kind != wcb.Array {
		return FloatTiledMeta{}, werrors.NewMalformed(
			"float tiled \"tiles\" must be an array")
	}
	if uint64(len(tilesValue.Array)) != tileCount {
		return FloatTiledMeta{}, werrors.Malformedf(
			"float tiled has %d tile entries but the grid requires %d",
			len(tilesValue.Array), tileCount)
	}
	tiles := make([]FloatTiledEntry, 0, len(tilesValue.Array))
	var expectedOffset uint64
	for position, item := range tilesValue.Array {
		if item.Kind != wcb.Map {
			return FloatTiledMeta{}, werrors.NewMalformed(
				"float tile entry must be a WCB map")
		}
		entry := item.Map
		if err := checkExactKeys(entry, []string{
			"index", "dims", "offset", "compressed_len",
			"exponent_plane_len", "mantissa_plane_len", "element_len", "tile_root",
		}, "float tile"); err != nil {
			return FloatTiledMeta{}, err
		}
		index, err := metaU64(entry, "index")
		if err != nil {
			return FloatTiledMeta{}, err
		}
		if index != uint64(position) {
			return FloatTiledMeta{}, werrors.Malformedf(
				"float tiled entry %d declares index %d; entries must be in canonical order",
				position, index)
		}
		dims, err := metaShapeKey(entry, "dims")
		if err != nil {
			return FloatTiledMeta{}, err
		}
		if !u64SlicesEqual(dims, tileShape) {
			return FloatTiledMeta{}, werrors.Malformedf(
				"float tile %d dims %v do not equal tile_shape %v; partial tiles are not defined in v1",
				position, dims, tileShape)
		}
		declaredElementLen, err := metaU64(entry, "element_len")
		if err != nil {
			return FloatTiledMeta{}, err
		}
		if declaredElementLen != tileElementLen {
			return FloatTiledMeta{}, werrors.Malformedf(
				"float tile %d element_len %d does not match %d",
				position, declaredElementLen, tileElementLen)
		}
		offset, err := metaU64(entry, "offset")
		if err != nil {
			return FloatTiledMeta{}, err
		}
		if offset != expectedOffset {
			return FloatTiledMeta{}, werrors.Malformedf(
				"float tile %d offset %d is not the contiguous offset %d",
				position, offset, expectedOffset)
		}
		compressedLen, err := metaU64(entry, "compressed_len")
		if err != nil {
			return FloatTiledMeta{}, err
		}
		exponentPlaneLen, err := metaU64(entry, "exponent_plane_len")
		if err != nil {
			return FloatTiledMeta{}, err
		}
		mantissaPlaneLen, err := metaU64(entry, "mantissa_plane_len")
		if err != nil {
			return FloatTiledMeta{}, err
		}
		if exponentPlaneLen < huffmanTableBytes {
			return FloatTiledMeta{}, werrors.Malformedf(
				"float tile %d exponent plane is shorter than its code-length table", position)
		}
		if mantissaPlaneLen != expectedMantissa {
			return FloatTiledMeta{}, werrors.Malformedf(
				"float tile %d mantissa_plane_len %d does not match %d",
				position, mantissaPlaneLen, expectedMantissa)
		}
		if compressedLen != exponentPlaneLen+mantissaPlaneLen {
			return FloatTiledMeta{}, werrors.Malformedf(
				"float tile %d compressed_len %d does not equal its plane lengths",
				position, compressedLen)
		}
		if compressedLen == 0 || expectedOffset > math.MaxUint64-compressedLen {
			return FloatTiledMeta{}, werrors.NewLimitExceeded(
				"float tile offset overflows u64")
		}
		expectedOffset += compressedLen
		rootBytes, err := metaBytes(entry, "tile_root")
		if err != nil {
			return FloatTiledMeta{}, err
		}
		if len(rootBytes) != 32 {
			return FloatTiledMeta{}, werrors.NewMalformed(
				"float tile_root must be 32 bytes")
		}
		var root hash.Hash32
		copy(root[:], rootBytes)
		tiles = append(tiles, FloatTiledEntry{
			Index:            index,
			Dims:             dims,
			Offset:           offset,
			CompressedLen:    compressedLen,
			ExponentPlaneLen: exponentPlaneLen,
			MantissaPlaneLen: mantissaPlaneLen,
			ElementLen:       declaredElementLen,
			TileRoot:         root,
		})
	}
	if expectedOffset != declaredPayloadLen {
		return FloatTiledMeta{}, werrors.Malformedf(
			"float tile ranges cover %d bytes but payload_len is %d",
			expectedOffset, declaredPayloadLen)
	}
	logicalBytes, err := metaBytes(m, "logical_root")
	if err != nil {
		return FloatTiledMeta{}, err
	}
	if len(logicalBytes) != 32 {
		return FloatTiledMeta{}, werrors.NewMalformed(
			"float tiled logical_root must be 32 bytes")
	}
	var logical hash.Hash32
	copy(logical[:], logicalBytes)
	return FloatTiledMeta{
		Dtype:        dtype,
		Shape:        shape,
		ElementCount: elementCount,
		TileShape:    tileShape,
		TileCounts:   counts,
		Tiles:        tiles,
		PayloadLen:   declaredPayloadLen,
		LogicalRoot:  logical,
	}, nil
}

// DecodeFloatTile decodes one stored float tile and verifies its tile root.
func DecodeFloatTile(dtype FloatDtype, entry FloatTiledEntry, stored []byte, lim limits.Limits) ([]byte, error) {
	if uint64(len(stored)) != entry.CompressedLen {
		return nil, werrors.Corruptf(
			"float tile %d stored range is %d bytes but the descriptor records %d",
			entry.Index, len(stored), entry.CompressedLen)
	}
	if entry.ExponentPlaneLen > uint64(len(stored)) {
		return nil, werrors.NewCorrupt("float tile exponent plane is truncated")
	}
	if entry.ExponentPlaneLen+entry.MantissaPlaneLen != entry.CompressedLen {
		return nil, werrors.NewMalformed(
			"float tile plane lengths disagree with compressed_len")
	}
	exponentStored := stored[:entry.ExponentPlaneLen]
	mantissaPlane := stored[entry.ExponentPlaneLen:]
	elementSize := dtype.ElementSize()
	if elementSize == 0 || entry.ElementLen%elementSize != 0 {
		return nil, werrors.NewMalformed(
			"float tile element length is not a whole number of elements")
	}
	elements := entry.ElementLen / elementSize
	exponentPlane, _, err := HuffmanDecodePrefix(exponentStored, int(elements))
	if err != nil {
		return nil, err
	}
	tile, err := JoinFloatTensorPayload(exponentPlane, mantissaPlane, dtype, elements, lim)
	if err != nil {
		return nil, err
	}
	root, err := FloatTileRoot(
		dtype.Name, entry.Dims, tile, entry.ExponentPlaneLen, entry.MantissaPlaneLen, lim)
	if err != nil {
		return nil, err
	}
	if root != entry.TileRoot {
		return nil, werrors.Corruptf(
			"float tile %d root %s does not match the descriptor's %s",
			entry.Index, root, entry.TileRoot)
	}
	return tile, nil
}

// DecodeFloatTiled decodes a mode-17 frame into exact row-major bytes.
func DecodeFloatTiled(f *frame.Frame, lim limits.Limits) ([]byte, error) {
	descriptor, err := ParseFloatTiled(f, lim)
	if err != nil {
		return nil, err
	}
	meta := descriptor.Meta
	tensorDtypeName, ok := floatTiledTensorDtype(meta.Dtype)
	if !ok {
		return nil, werrors.UnsupportedFeaturef(
			"float-tiled v1 requires a tensor identity layout; dtype %q has none", meta.Dtype.Name)
	}
	elementSize := meta.Dtype.ElementSize()
	if meta.ElementCount != 0 && elementSize > math.MaxUint64/meta.ElementCount {
		return nil, werrors.NewLimitExceeded("float tiled decoded bytes overflow u64")
	}
	total := meta.ElementCount * elementSize
	if uint64(int(total)) != total {
		return nil, werrors.NewLimitExceeded("float tiled decoded bytes do not fit int")
	}
	out := make([]byte, int(total))
	var tilesBytes uint64
	dstExtent := make([]uint64, len(meta.Shape))
	copy(dstExtent, meta.Shape)
	for _, entry := range meta.Tiles {
		if entry.Offset > uint64(len(descriptor.Payload)) ||
			entry.CompressedLen > uint64(len(descriptor.Payload))-entry.Offset {
			return nil, werrors.NewCorrupt("float tile range is outside the payload")
		}
		stored := descriptor.Payload[entry.Offset : entry.Offset+entry.CompressedLen]
		tile, err := DecodeFloatTile(meta.Dtype, entry, stored, lim)
		if err != nil {
			return nil, err
		}
		tilesBytes += uint64(len(tile))
		coords := tileCoords(meta.TileCounts, entry.Index)
		dstStart := make([]uint64, len(coords))
		for axis := range coords {
			dstStart[axis] = coords[axis] * meta.TileShape[axis]
		}
		if err := copyBoxInto(out, dstExtent, dstStart, tile, entry.Dims, elementSize); err != nil {
			return nil, err
		}
	}
	if tilesBytes != total {
		return nil, werrors.Corruptf(
			"float tiled decoded to %d bytes but the descriptor requires %d",
			tilesBytes, total)
	}
	logical, err := tensor.LogicalRoot(tensorDtypeName, meta.Shape, out, lim)
	if err != nil {
		return nil, err
	}
	if logical != meta.LogicalRoot {
		return nil, werrors.Corruptf(
			"float tiled logical root %s does not match the descriptor's %s",
			logical, meta.LogicalRoot)
	}
	return out, nil
}

// ReadFloatRegion reads one logical region, decoding only the float tiles it
// intersects. It returns the region bytes and the number of tiles decoded.
func ReadFloatRegion(descriptor FloatTiledDescriptor, region tensor.Region, lim limits.Limits) ([]byte, uint64, error) {
	meta := descriptor.Meta
	if err := region.Validate(meta.Shape); err != nil {
		return nil, 0, err
	}
	extent := region.Extent()
	if region.IsEmpty() {
		return []byte{}, 0, nil
	}
	elementSize := meta.Dtype.ElementSize()
	count := uint64(1)
	for _, dim := range extent {
		if count > math.MaxUint64/dim {
			return nil, 0, werrors.NewLimitExceeded("region element count overflows u64")
		}
		count *= dim
	}
	if count != 0 && elementSize > math.MaxUint64/count {
		return nil, 0, werrors.NewLimitExceeded("region payload length overflows u64")
	}
	byteLen := count * elementSize
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
				return nil, 0, werrors.NewLimitExceeded("float tile index overflows u64")
			}
			index = index*meta.TileCounts[axis] + coords[axis]
		}
		if index >= uint64(len(meta.Tiles)) {
			return nil, 0, werrors.NewCorrupt("float tile index is outside the descriptor")
		}
		entry := meta.Tiles[index]
		if entry.Offset > uint64(len(descriptor.Payload)) ||
			entry.CompressedLen > uint64(len(descriptor.Payload))-entry.Offset {
			return nil, 0, werrors.NewCorrupt("float tile range is outside the payload")
		}
		tile, err := DecodeFloatTile(
			meta.Dtype, entry, descriptor.Payload[entry.Offset:entry.Offset+entry.CompressedLen], lim)
		if err != nil {
			return nil, 0, err
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
