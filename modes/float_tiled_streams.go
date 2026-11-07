package modes

// FLOAT_TILED_STREAMS (mode 18, WEFT-AI-CODEC §19).
//
// The tile grid, plane split, offsets and roots are exactly mode 17's
// discipline. Within one tile the exponent plane is split round-robin into
// `n_streams` independently coded `huffman-canonical-v1` streams, and the raw
// sign+mantissa plane is split round-robin into `n_streams` raw streams. With
// `n_streams == 1` the stored tile bytes are byte-identical to mode 17.

import (
	"math"
	"sync"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// FloatTiledStreamsSchema is the canonical mode-18 descriptor schema.
const FloatTiledStreamsSchema = "weft.ai.tensor_tiled_streams.v1"

const (
	// FloatTiledStreamsMin is the smallest declared stream count.
	FloatTiledStreamsMin = 1
	// FloatTiledStreamsMax is the largest declared stream count.
	FloatTiledStreamsMax = 16
	// FloatTiledStreamsDefault is the stream count the store planner offers.
	FloatTiledStreamsDefault = 4
)

// FloatTiledStreamsEntry is one independently decodable multi-stream float
// tile.
type FloatTiledStreamsEntry struct {
	Index            uint64
	Dims             []uint64
	Offset           uint64
	CompressedLen    uint64
	ExponentPlaneLen uint64
	MantissaPlaneLen uint64
	ElementLen       uint64
	// StreamOffsets are the starts of the 2*n_streams streams relative to
	// the tile start: the first n_streams are exponent streams and the next
	// n_streams the raw mantissa streams.
	StreamOffsets []uint64
	TileRoot      hash.Hash32
}

// FloatTiledStreamsMeta is a parsed mode-18 descriptor.
type FloatTiledStreamsMeta struct {
	Dtype        FloatDtype
	Shape        []uint64
	ElementCount uint64
	TileShape    []uint64
	TileCounts   []uint64
	NStreams     uint64
	Tiles        []FloatTiledStreamsEntry
	PayloadLen   uint64
	LogicalRoot  hash.Hash32
}

// FloatTiledStreamsDescriptor is a parsed mode-18 frame with its payload.
type FloatTiledStreamsDescriptor struct {
	Meta    FloatTiledStreamsMeta
	Payload []byte
}

// FloatTileStreamsRoot is
// H("weft/tensor/float_tile_streams/v1",
//
//	WCB({dtype, dims, element_len, exponent_plane_len, mantissa_plane_len,
//	      n_streams, stream_offsets, values_hash})).
//
// The record covers n_streams and the whole offset table, so changing either
// changes the tile root.
func FloatTileStreamsRoot(
	dtype string, dims []uint64, tileBytes []byte,
	exponentPlaneLen, mantissaPlaneLen, nStreams uint64, streamOffsets []uint64,
	lim limits.Limits,
) (hash.Hash32, error) {
	valuesHash := hash.Domain(tensor.DomainTensorValues, tileBytes)
	meta := wcb.MapValue(map[string]wcb.Value{
		"dtype":              wcb.TextValue(dtype),
		"dims":               wcb.ArrayValue(u64Values(dims)),
		"element_len":        wcb.U64Value(uint64(len(tileBytes))),
		"exponent_plane_len": wcb.U64Value(exponentPlaneLen),
		"mantissa_plane_len": wcb.U64Value(mantissaPlaneLen),
		"n_streams":          wcb.U64Value(nStreams),
		"stream_offsets":     wcb.ArrayValue(u64Values(streamOffsets)),
		"values_hash":        wcb.BytesValue(valuesHash.Bytes()),
	})
	encoded, err := wcb.Encode(meta, lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(hash.DomainFloatTileStreams, encoded), nil
}

// InterleavedStreamLen is the number of bytes stream `stream` holds when a
// plane of `total` bytes is split round-robin over `nStreams`.
func InterleavedStreamLen(total, stream, nStreams uint64) uint64 {
	if nStreams == 0 || total <= stream {
		return 0
	}
	return (total - stream + nStreams - 1) / nStreams
}

func interleavePlane(plane []byte, nStreams int) [][]byte {
	streams := make([][]byte, nStreams)
	for index := range streams {
		streams[index] = make([]byte, 0, len(plane)/nStreams+1)
	}
	for index, value := range plane {
		streams[index%nStreams] = append(streams[index%nStreams], value)
	}
	return streams
}

func deinterleavePlanes(streams [][]byte, total int, what string) ([]byte, error) {
	if len(streams) == 0 {
		return nil, werrors.Malformedf("%s declares zero streams", what)
	}
	out := make([]byte, total)
	for index := 0; index < total; index++ {
		stream := streams[index%len(streams)]
		position := index / len(streams)
		if position >= len(stream) {
			return nil, werrors.Corruptf(
				"%s stream %d ended before its interleaved plane", what, index%len(streams))
		}
		out[index] = stream[position]
	}
	return out, nil
}

// decodeStreamPlanes decodes nStreams independent byte planes, in parallel on
// a bounded pool when threads > 1. Stream errors are returned in global stream
// order on both paths.
func decodeMultiStreamPlanes(encoded [][]byte, expected []int, threads int) ([][]byte, error) {
	if threads <= 1 || len(encoded) <= 1 {
		out := make([][]byte, 0, len(encoded))
		for index, bytes := range encoded {
			decoded, err := HuffmanDecode(bytes, expected[index])
			if err != nil {
				return nil, err
			}
			out = append(out, decoded)
		}
		return out, nil
	}
	workers := threads
	if workers > len(encoded) {
		workers = len(encoded)
	}
	if workers > 64 {
		workers = 64
	}
	if workers < 1 {
		workers = 1
	}
	chunk := (len(encoded) + workers - 1) / workers
	type result struct {
		parts [][]byte
		err   error
	}
	chunks := (len(encoded) + chunk - 1) / chunk
	var wg sync.WaitGroup
	results := make([]result, chunks)
	for ordinal := 0; ordinal < chunks; ordinal++ {
		start := ordinal * chunk
		end := start + chunk
		if end > len(encoded) {
			end = len(encoded)
		}
		wg.Add(1)
		go func(ordinal, start, end int) {
			defer wg.Done()
			parts := make([][]byte, 0, end-start)
			for index := start; index < end; index++ {
				decoded, err := HuffmanDecode(encoded[index], expected[index])
				if err != nil {
					results[ordinal] = result{err: err}
					return
				}
				parts = append(parts, decoded)
			}
			results[ordinal] = result{parts: parts}
		}(ordinal, start, end)
	}
	wg.Wait()
	out := make([][]byte, 0, len(encoded))
	for _, item := range results {
		if item.err != nil {
			return nil, item.err
		}
		out = append(out, item.parts...)
	}
	return out, nil
}

// EncodeFloatTiledStreams encodes a mode-18 FLOAT_TILED_STREAMS frame.
func EncodeFloatTiledStreams(
	payload []byte, dtype FloatDtype, shape, tileShape []uint64, nStreams uint64, lim limits.Limits,
) ([]byte, error) {
	if nStreams < FloatTiledStreamsMin || nStreams > FloatTiledStreamsMax {
		return nil, werrors.Malformedf(
			"n_streams %d is outside %d..=%d", nStreams, FloatTiledStreamsMin, FloatTiledStreamsMax)
	}
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
	n := int(nStreams)
	stored := make([]byte, 0, len(payload))
	tiles := make([]FloatTiledStreamsEntry, 0, int(tileCount))
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
		exponentPlane, mantissaPlane, err := splitFloatTensorPlanes(tile, dtype, tileElements)
		if err != nil {
			return nil, err
		}
		tileStored := make([]byte, 0, len(tile))
		streamOffsets := make([]uint64, 0, 2*n)
		for _, stream := range interleavePlane(exponentPlane, n) {
			streamOffsets = append(streamOffsets, uint64(len(tileStored)))
			tileStored = append(tileStored, HuffmanEncode(stream)...)
		}
		exponentPlaneLen := uint64(len(tileStored))
		for _, stream := range interleavePlane(mantissaPlane, n) {
			streamOffsets = append(streamOffsets, uint64(len(tileStored)))
			tileStored = append(tileStored, stream...)
		}
		compressedLen := uint64(len(tileStored))
		root, err := FloatTileStreamsRoot(
			dtype.Name, tileShape, tile,
			exponentPlaneLen, uint64(len(mantissaPlane)), nStreams, streamOffsets, lim)
		if err != nil {
			return nil, err
		}
		tiles = append(tiles, FloatTiledStreamsEntry{
			Index:            index,
			Dims:             append([]uint64(nil), tileShape...),
			Offset:           uint64(len(stored)),
			CompressedLen:    compressedLen,
			ExponentPlaneLen: exponentPlaneLen,
			MantissaPlaneLen: uint64(len(mantissaPlane)),
			ElementLen:       tileElementLen,
			StreamOffsets:    streamOffsets,
			TileRoot:         root,
		})
		stored = append(stored, tileStored...)
	}
	logicalRoot, err := tensor.LogicalRoot(tensorDtypeName, shape, payload, lim)
	if err != nil {
		return nil, err
	}
	meta := FloatTiledStreamsMeta{
		Dtype:        dtype,
		Shape:        append([]uint64(nil), shape...),
		ElementCount: elementCount,
		TileShape:    append([]uint64(nil), tileShape...),
		TileCounts:   counts,
		NStreams:     nStreams,
		Tiles:        tiles,
		PayloadLen:   uint64(len(stored)),
		LogicalRoot:  logicalRoot,
	}
	body, err := buildAIBody(floatTiledStreamsMetaValue(meta), stored, lim)
	if err != nil {
		return nil, err
	}
	return frame.Encode(frame.PayloadParts(
		frame.ModeFloatTiledStreams, 0, 0, decodedBytes,
		hash.Domain(hash.DomainChunk, payload), body), lim)
}

func floatTiledStreamsMetaValue(meta FloatTiledStreamsMeta) wcb.Value {
	tiles := make([]wcb.Value, 0, len(meta.Tiles))
	for _, entry := range meta.Tiles {
		tiles = append(tiles, wcb.MapValue(map[string]wcb.Value{
			"index":              wcb.U64Value(entry.Index),
			"dims":               wcb.ArrayValue(u64Values(entry.Dims)),
			"offset":             wcb.U64Value(entry.Offset),
			"compressed_len":     wcb.U64Value(entry.CompressedLen),
			"exponent_plane_len": wcb.U64Value(entry.ExponentPlaneLen),
			"mantissa_plane_len": wcb.U64Value(entry.MantissaPlaneLen),
			"element_len":        wcb.U64Value(entry.ElementLen),
			"stream_offsets":     wcb.ArrayValue(u64Values(entry.StreamOffsets)),
			"tile_root":          wcb.BytesValue(entry.TileRoot.Bytes()),
		}))
	}
	return wcb.MapValue(map[string]wcb.Value{
		"schema":         wcb.TextValue(FloatTiledStreamsSchema),
		"dtype":          wcb.TextValue(meta.Dtype.Name),
		"shape":          wcb.ArrayValue(u64Values(meta.Shape)),
		"element_count":  wcb.U64Value(meta.ElementCount),
		"layout":         wcb.TextValue(FloatTileLayout),
		"tile_shape":     wcb.ArrayValue(u64Values(meta.TileShape)),
		"tile_counts":    wcb.ArrayValue(u64Values(meta.TileCounts)),
		"exponent_coder": wcb.TextValue(HuffmanCoderID),
		"n_streams":      wcb.U64Value(meta.NStreams),
		"tiles":          wcb.ArrayValue(tiles),
		"payload_len":    wcb.U64Value(meta.PayloadLen),
		"logical_root":   wcb.BytesValue(meta.LogicalRoot.Bytes()),
	})
}

// ParseFloatTiledStreamsDescriptor parses a mode-18 descriptor from canonical
// WCB bytes.
func ParseFloatTiledStreamsDescriptor(
	descriptorBytes []byte, decodedLen, payloadLen uint64, lim limits.Limits,
) (FloatTiledStreamsMeta, error) {
	value, err := wcb.Decode(descriptorBytes, lim)
	if err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	if value.Kind != wcb.Map {
		return FloatTiledStreamsMeta{}, werrors.NewMalformed(
			"float tiled streams metadata must be a WCB map")
	}
	m := value.Map
	if err := checkExactKeys(m, []string{
		"schema", "dtype", "shape", "element_count", "layout", "tile_shape",
		"tile_counts", "exponent_coder", "n_streams", "tiles", "payload_len",
		"logical_root",
	}, "float tiled streams"); err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	schema, err := metaText(m, "schema")
	if err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	if schema != FloatTiledStreamsSchema {
		return FloatTiledStreamsMeta{}, werrors.Malformedf(
			"unsupported float tiled streams schema %q", schema)
	}
	dtypeName, err := metaText(m, "dtype")
	if err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	dtype, err := ParseFloatDtype(dtypeName)
	if err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	if _, ok := floatTiledTensorDtype(dtype); !ok {
		return FloatTiledStreamsMeta{}, werrors.UnsupportedFeaturef(
			"float-tiled v1 requires a tensor identity layout; dtype %q has none", dtype.Name)
	}
	layout, err := metaText(m, "layout")
	if err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	if layout != FloatTileLayout {
		return FloatTiledStreamsMeta{}, werrors.Malformedf(
			"float tiled streams layout %q is specified but not implemented in v1 (only %s)",
			layout, FloatTileLayout)
	}
	coder, err := metaText(m, "exponent_coder")
	if err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	if coder != HuffmanCoderID {
		return FloatTiledStreamsMeta{}, werrors.Malformedf(
			"unsupported float tile exponent coder %q (v1 allows %s)", coder, HuffmanCoderID)
	}
	nStreams, err := metaU64(m, "n_streams")
	if err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	if nStreams < FloatTiledStreamsMin || nStreams > FloatTiledStreamsMax {
		return FloatTiledStreamsMeta{}, werrors.Malformedf(
			"n_streams %d is outside %d..=%d", nStreams, FloatTiledStreamsMin, FloatTiledStreamsMax)
	}
	n := int(nStreams)
	shape, err := metaShape(m)
	if err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	elementCount, err := floatTensorProduct(shape, lim)
	if err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	declaredCount, err := metaU64(m, "element_count")
	if err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	if declaredCount != elementCount {
		return FloatTiledStreamsMeta{}, werrors.Malformedf(
			"float tiled element_count %d does not match shape product %d",
			declaredCount, elementCount)
	}
	elementSize := dtype.ElementSize()
	if elementCount != 0 && elementSize > math.MaxUint64/elementCount {
		return FloatTiledStreamsMeta{}, werrors.NewLimitExceeded(
			"float tiled decoded bytes overflow u64")
	}
	decodedBytes := elementCount * elementSize
	if err := limits.CheckLen(decodedBytes, lim.MaxDecodedBytes, "float tiled decoded bytes"); err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	if decodedLen != decodedBytes {
		return FloatTiledStreamsMeta{}, werrors.Corruptf(
			"float tiled decoded_len is %d but the descriptor requires %d",
			decodedLen, decodedBytes)
	}
	tileShape, err := metaShapeKey(m, "tile_shape")
	if err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	counts, err := tileCounts(shape, tileShape)
	if err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	declaredCounts, err := metaShapeKey(m, "tile_counts")
	if err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	if !u64SlicesEqual(declaredCounts, counts) {
		return FloatTiledStreamsMeta{}, werrors.Malformedf(
			"float tiled tile_counts %v do not match shape/tile_shape %v",
			declaredCounts, counts)
	}
	tileElements, err := tiledProduct(tileShape, lim, "float tile")
	if err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	if tileElements != 0 && elementSize > math.MaxUint64/tileElements {
		return FloatTiledStreamsMeta{}, werrors.NewLimitExceeded(
			"float tile byte length overflows u64")
	}
	tileElementLen := tileElements * elementSize
	tileCount, err := tileCountOf(counts, lim)
	if err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	mantissaBits := dtype.SignBits + dtype.MantissaBits
	if tileElements != 0 && mantissaBits > math.MaxUint64/tileElements {
		return FloatTiledStreamsMeta{}, werrors.NewLimitExceeded(
			"float tile mantissa bits overflow u64")
	}
	expectedMantissa := (tileElements*mantissaBits + 7) / 8
	declaredPayloadLen, err := metaU64(m, "payload_len")
	if err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	if declaredPayloadLen != payloadLen {
		return FloatTiledStreamsMeta{}, werrors.Malformedf(
			"float tiled payload is %d bytes but metadata declares %d",
			payloadLen, declaredPayloadLen)
	}
	tilesValue, ok := m["tiles"]
	if !ok || tilesValue.Kind != wcb.Array {
		return FloatTiledStreamsMeta{}, werrors.NewMalformed(
			"float tiled streams \"tiles\" must be an array")
	}
	if uint64(len(tilesValue.Array)) != tileCount {
		return FloatTiledStreamsMeta{}, werrors.Malformedf(
			"float tiled streams has %d tile entries but the grid requires %d",
			len(tilesValue.Array), tileCount)
	}
	tiles := make([]FloatTiledStreamsEntry, 0, len(tilesValue.Array))
	var expectedOffset uint64
	for position, item := range tilesValue.Array {
		if item.Kind != wcb.Map {
			return FloatTiledStreamsMeta{}, werrors.NewMalformed(
				"float tile streams entry must be a WCB map")
		}
		entry := item.Map
		if err := checkExactKeys(entry, []string{
			"index", "dims", "offset", "compressed_len", "exponent_plane_len",
			"mantissa_plane_len", "element_len", "stream_offsets", "tile_root",
		}, "float tile streams tile"); err != nil {
			return FloatTiledStreamsMeta{}, err
		}
		index, err := metaU64(entry, "index")
		if err != nil {
			return FloatTiledStreamsMeta{}, err
		}
		if index != uint64(position) {
			return FloatTiledStreamsMeta{}, werrors.Malformedf(
				"float tiled streams entry %d declares index %d; entries must be in canonical order",
				position, index)
		}
		dims, err := metaShapeKey(entry, "dims")
		if err != nil {
			return FloatTiledStreamsMeta{}, err
		}
		if !u64SlicesEqual(dims, tileShape) {
			return FloatTiledStreamsMeta{}, werrors.Malformedf(
				"float tile %d dims %v do not equal tile_shape %v; partial tiles are not defined in v1",
				position, dims, tileShape)
		}
		declaredElementLen, err := metaU64(entry, "element_len")
		if err != nil {
			return FloatTiledStreamsMeta{}, err
		}
		if declaredElementLen != tileElementLen {
			return FloatTiledStreamsMeta{}, werrors.Malformedf(
				"float tile %d element_len %d does not match %d",
				position, declaredElementLen, tileElementLen)
		}
		offset, err := metaU64(entry, "offset")
		if err != nil {
			return FloatTiledStreamsMeta{}, err
		}
		if offset != expectedOffset {
			return FloatTiledStreamsMeta{}, werrors.Malformedf(
				"float tile %d offset %d is not the contiguous offset %d",
				position, offset, expectedOffset)
		}
		compressedLen, err := metaU64(entry, "compressed_len")
		if err != nil {
			return FloatTiledStreamsMeta{}, err
		}
		exponentPlaneLen, err := metaU64(entry, "exponent_plane_len")
		if err != nil {
			return FloatTiledStreamsMeta{}, err
		}
		mantissaPlaneLen, err := metaU64(entry, "mantissa_plane_len")
		if err != nil {
			return FloatTiledStreamsMeta{}, err
		}
		if mantissaPlaneLen != expectedMantissa {
			return FloatTiledStreamsMeta{}, werrors.Malformedf(
				"float tile %d mantissa_plane_len %d does not match %d",
				position, mantissaPlaneLen, expectedMantissa)
		}
		if exponentPlaneLen > compressedLen || compressedLen-exponentPlaneLen != mantissaPlaneLen {
			return FloatTiledStreamsMeta{}, werrors.Malformedf(
				"float tile %d compressed_len %d does not equal its plane lengths",
				position, compressedLen)
		}
		if compressedLen == 0 || expectedOffset > math.MaxUint64-compressedLen {
			return FloatTiledStreamsMeta{}, werrors.NewLimitExceeded(
				"float tile offset overflows u64")
		}
		expectedOffset += compressedLen

		streamOffsets, err := metaShapeKey(entry, "stream_offsets")
		if err != nil {
			return FloatTiledStreamsMeta{}, err
		}
		if len(streamOffsets) != 2*n {
			return FloatTiledStreamsMeta{}, werrors.Malformedf(
				"float tile %d has %d stream offsets but n_streams %d requires %d",
				position, len(streamOffsets), nStreams, 2*nStreams)
		}
		if streamOffsets[0] != 0 {
			return FloatTiledStreamsMeta{}, werrors.Malformedf(
				"float tile %d stream_offsets[0] is %d not 0", position, streamOffsets[0])
		}
		for i := 1; i < len(streamOffsets); i++ {
			if streamOffsets[i] < streamOffsets[i-1] {
				return FloatTiledStreamsMeta{}, werrors.Malformedf(
					"float tile %d stream_offsets are not monotonic: %d then %d",
					position, streamOffsets[i-1], streamOffsets[i])
			}
		}
		if streamOffsets[2*n-1] > compressedLen {
			return FloatTiledStreamsMeta{}, werrors.Malformedf(
				"float tile %d stream offset %d is past the %d-byte tile payload",
				position, streamOffsets[2*n-1], compressedLen)
		}
		if streamOffsets[n] != exponentPlaneLen {
			return FloatTiledStreamsMeta{}, werrors.Malformedf(
				"float tile %d stream_offsets[%d] is %d but exponent_plane_len is %d",
				position, n, streamOffsets[n], exponentPlaneLen)
		}
		for stream := 0; stream < n; stream++ {
			end := exponentPlaneLen
			if stream+1 < n {
				end = streamOffsets[stream+1]
			}
			if end-streamOffsets[stream] < huffmanTableBytes {
				return FloatTiledStreamsMeta{}, werrors.Malformedf(
					"float tile %d exponent stream %d is %d bytes, shorter than its %d-byte code-length table",
					position, stream, end-streamOffsets[stream], huffmanTableBytes)
			}
		}
		for stream := 0; stream < n; stream++ {
			start := streamOffsets[n+stream] - exponentPlaneLen
			end := mantissaPlaneLen
			if stream+1 < n {
				end = streamOffsets[n+stream+1] - exponentPlaneLen
			}
			if end < start {
				return FloatTiledStreamsMeta{}, werrors.Malformedf(
					"float tile %d mantissa stream %d offsets are inverted", position, stream)
			}
			expected := InterleavedStreamLen(mantissaPlaneLen, uint64(stream), nStreams)
			if end-start != expected {
				return FloatTiledStreamsMeta{}, werrors.Malformedf(
					"float tile %d mantissa stream %d is %d bytes but the interleave requires %d",
					position, stream, end-start, expected)
			}
		}
		rootBytes, err := metaBytes(entry, "tile_root")
		if err != nil {
			return FloatTiledStreamsMeta{}, err
		}
		if len(rootBytes) != 32 {
			return FloatTiledStreamsMeta{}, werrors.NewMalformed(
				"float tile_root must be 32 bytes")
		}
		var root hash.Hash32
		copy(root[:], rootBytes)
		tiles = append(tiles, FloatTiledStreamsEntry{
			Index:            index,
			Dims:             dims,
			Offset:           offset,
			CompressedLen:    compressedLen,
			ExponentPlaneLen: exponentPlaneLen,
			MantissaPlaneLen: mantissaPlaneLen,
			ElementLen:       declaredElementLen,
			StreamOffsets:    streamOffsets,
			TileRoot:         root,
		})
	}
	if expectedOffset != declaredPayloadLen {
		return FloatTiledStreamsMeta{}, werrors.Malformedf(
			"float tile ranges cover %d bytes but payload_len is %d",
			expectedOffset, declaredPayloadLen)
	}
	logicalBytes, err := metaBytes(m, "logical_root")
	if err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	if len(logicalBytes) != 32 {
		return FloatTiledStreamsMeta{}, werrors.NewMalformed(
			"float tiled logical_root must be 32 bytes")
	}
	var logical hash.Hash32
	copy(logical[:], logicalBytes)
	return FloatTiledStreamsMeta{
		Dtype:        dtype,
		Shape:        shape,
		ElementCount: elementCount,
		TileShape:    tileShape,
		TileCounts:   counts,
		NStreams:     nStreams,
		Tiles:        tiles,
		PayloadLen:   declaredPayloadLen,
		LogicalRoot:  logical,
	}, nil
}

// ParseFloatTiledStreams parses a mode-18 frame with its payload slice.
func ParseFloatTiledStreams(f *frame.Frame, lim limits.Limits) (FloatTiledStreamsDescriptor, error) {
	if f.Header.Mode != frame.ModeFloatTiledStreams {
		return FloatTiledStreamsDescriptor{}, werrors.Malformedf(
			"frame mode %s is not float_tiled_streams", f.Header.Mode)
	}
	descriptorBytes, payload, err := splitAIBodyPrefix(f.Body)
	if err != nil {
		return FloatTiledStreamsDescriptor{}, err
	}
	meta, err := ParseFloatTiledStreamsDescriptor(
		descriptorBytes, f.Header.DecodedLen, uint64(len(payload)), lim)
	if err != nil {
		return FloatTiledStreamsDescriptor{}, err
	}
	return FloatTiledStreamsDescriptor{Meta: meta, Payload: payload}, nil
}

// FloatTiledStreamsMetadata parses a mode-18 frame and returns its descriptor.
func FloatTiledStreamsMetadata(f *frame.Frame, lim limits.Limits) (FloatTiledStreamsMeta, error) {
	descriptor, err := ParseFloatTiledStreams(f, lim)
	if err != nil {
		return FloatTiledStreamsMeta{}, err
	}
	return descriptor.Meta, nil
}

// DecodeFloatTileStreams decodes one stored multi-stream float tile and
// verifies its tile root.
func DecodeFloatTileStreams(
	dtype FloatDtype, entry FloatTiledStreamsEntry, nStreams uint64,
	stored []byte, lim limits.Limits, threads int,
) ([]byte, error) {
	if nStreams < FloatTiledStreamsMin || nStreams > FloatTiledStreamsMax {
		return nil, werrors.Malformedf(
			"float tile %d declares n_streams %d outside %d..=%d",
			entry.Index, nStreams, FloatTiledStreamsMin, FloatTiledStreamsMax)
	}
	if uint64(len(entry.StreamOffsets)) != 2*nStreams {
		return nil, werrors.Malformedf(
			"float tile %d has %d stream offsets but n_streams %d requires %d",
			entry.Index, len(entry.StreamOffsets), nStreams, 2*nStreams)
	}
	if uint64(len(stored)) != entry.CompressedLen {
		return nil, werrors.Corruptf(
			"float tile %d stored range is %d bytes but the descriptor records %d",
			entry.Index, len(stored), entry.CompressedLen)
	}
	if entry.ExponentPlaneLen > uint64(len(stored)) {
		return nil, werrors.NewCorrupt("float tile exponent plane is truncated")
	}
	elementSize := dtype.ElementSize()
	if elementSize == 0 || entry.ElementLen%elementSize != 0 {
		return nil, werrors.NewMalformed(
			"float tile element length is not a whole number of elements")
	}
	elements := entry.ElementLen / elementSize
	exponentStored := stored[:entry.ExponentPlaneLen]
	mantissaStored := stored[entry.ExponentPlaneLen:]
	n := int(nStreams)

	encoded := make([][]byte, 0, n)
	expected := make([]int, 0, n)
	for stream := 0; stream < n; stream++ {
		start := entry.StreamOffsets[stream]
		end := entry.ExponentPlaneLen
		if stream+1 < n {
			end = entry.StreamOffsets[stream+1]
		}
		if end > uint64(len(exponentStored)) || start > end {
			return nil, werrors.NewCorrupt("float tile exponent stream is outside the tile")
		}
		expectedLen, err := limits.ToInt(
			InterleavedStreamLen(elements, uint64(stream), nStreams), "stream output length")
		if err != nil {
			return nil, err
		}
		encoded = append(encoded, exponentStored[start:end])
		expected = append(expected, expectedLen)
	}
	parts, err := decodeMultiStreamPlanes(encoded, expected, threads)
	if err != nil {
		return nil, err
	}
	exponentPlane, err := deinterleavePlanes(parts, int(elements), "exponent plane")
	if err != nil {
		return nil, err
	}

	mantissaParts := make([][]byte, 0, n)
	for stream := 0; stream < n; stream++ {
		start := entry.StreamOffsets[n+stream] - entry.ExponentPlaneLen
		end := entry.MantissaPlaneLen
		if stream+1 < n {
			end = entry.StreamOffsets[n+stream+1] - entry.ExponentPlaneLen
		}
		if end > uint64(len(mantissaStored)) || start > end {
			return nil, werrors.NewCorrupt("float tile mantissa stream is outside the tile")
		}
		expectedLen := InterleavedStreamLen(entry.MantissaPlaneLen, uint64(stream), nStreams)
		if end-start != expectedLen {
			return nil, werrors.Corruptf(
				"float tile %d mantissa stream %d is %d bytes but the interleave requires %d",
				entry.Index, stream, end-start, expectedLen)
		}
		mantissaParts = append(mantissaParts, mantissaStored[start:end])
	}
	mantissaPlane, err := deinterleavePlanes(
		mantissaParts, int(entry.MantissaPlaneLen), "mantissa plane")
	if err != nil {
		return nil, err
	}
	tile, err := JoinFloatTensorPayload(exponentPlane, mantissaPlane, dtype, elements, lim)
	if err != nil {
		return nil, err
	}
	root, err := FloatTileStreamsRoot(
		dtype.Name, entry.Dims, tile, entry.ExponentPlaneLen, entry.MantissaPlaneLen,
		nStreams, entry.StreamOffsets, lim)
	if err != nil {
		return nil, err
	}
	if root != entry.TileRoot {
		return nil, werrors.Corruptf(
			"float tile %d stream root %s does not match the descriptor's %s",
			entry.Index, root, entry.TileRoot)
	}
	return tile, nil
}

// DecodeFloatTiledStreams decodes a mode-18 frame into exact row-major bytes.
// threads bounds the per-tile stream workers (`1` decodes sequentially).
func DecodeFloatTiledStreams(f *frame.Frame, lim limits.Limits, threads int) ([]byte, error) {
	descriptor, err := ParseFloatTiledStreams(f, lim)
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
	for _, entry := range meta.Tiles {
		if entry.Offset > uint64(len(descriptor.Payload)) ||
			entry.CompressedLen > uint64(len(descriptor.Payload))-entry.Offset {
			return nil, werrors.NewCorrupt("float tile range is outside the payload")
		}
		stored := descriptor.Payload[entry.Offset : entry.Offset+entry.CompressedLen]
		tile, err := DecodeFloatTileStreams(
			meta.Dtype, entry, meta.NStreams, stored, lim, threads)
		if err != nil {
			return nil, err
		}
		tilesBytes += uint64(len(tile))
		coords := tileCoords(meta.TileCounts, entry.Index)
		dstStart := make([]uint64, len(coords))
		for axis := range coords {
			dstStart[axis] = coords[axis] * meta.TileShape[axis]
		}
		if err := copyBoxInto(out, meta.Shape, dstStart, tile, entry.Dims, elementSize); err != nil {
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

// ReadFloatTiledStreamsRegion reads one logical region, decoding only the
// multi-stream tiles it intersects. Every touched tile fetches all of its
// streams, so streamsTouched == tilesTouched * 2 * n_streams.
func ReadFloatTiledStreamsRegion(
	descriptor FloatTiledStreamsDescriptor, region tensor.Region, lim limits.Limits, threads int,
) ([]byte, uint64, uint64, error) {
	meta := descriptor.Meta
	if err := region.Validate(meta.Shape); err != nil {
		return nil, 0, 0, err
	}
	extent := region.Extent()
	if region.IsEmpty() {
		return []byte{}, 0, 0, nil
	}
	elementSize := meta.Dtype.ElementSize()
	count := uint64(1)
	for _, dim := range extent {
		if count > math.MaxUint64/dim {
			return nil, 0, 0, werrors.NewLimitExceeded("region element count overflows u64")
		}
		count *= dim
	}
	if count != 0 && elementSize > math.MaxUint64/count {
		return nil, 0, 0, werrors.NewLimitExceeded("region payload length overflows u64")
	}
	byteLen := count * elementSize
	if err := limits.CheckLen(byteLen, lim.MaxDecodedBytes, "region payload"); err != nil {
		return nil, 0, 0, err
	}
	if uint64(int(byteLen)) != byteLen {
		return nil, 0, 0, werrors.NewLimitExceeded("region payload does not fit int")
	}
	out := make([]byte, int(byteLen))
	list, err := floatTiledStreamsRegionIndices(meta, region)
	if err != nil {
		return nil, 0, 0, err
	}
	var tilesTouched uint64
	for _, index := range list {
		entry := meta.Tiles[index]
		if entry.Offset > uint64(len(descriptor.Payload)) ||
			entry.CompressedLen > uint64(len(descriptor.Payload))-entry.Offset {
			return nil, 0, 0, werrors.NewCorrupt("float tile range is outside the payload")
		}
		tile, err := DecodeFloatTileStreams(
			meta.Dtype, entry, meta.NStreams,
			descriptor.Payload[entry.Offset:entry.Offset+entry.CompressedLen], lim, threads)
		if err != nil {
			return nil, 0, 0, err
		}
		tilesTouched++
		if err := placeFloatTileRegion(out, region, meta, entry, tile); err != nil {
			return nil, 0, 0, err
		}
	}
	streamsTouched := tilesTouched * 2 * meta.NStreams
	return out, tilesTouched, streamsTouched, nil
}

// floatTiledStreamsRegionIndices returns the indices of tiles intersecting
// region.
func floatTiledStreamsRegionIndices(
	meta FloatTiledStreamsMeta, region tensor.Region,
) ([]uint64, error) {
	if err := region.Validate(meta.Shape); err != nil {
		return nil, err
	}
	if region.IsEmpty() {
		return nil, nil
	}
	first := make([]uint64, len(meta.Shape))
	last := make([]uint64, len(meta.Shape))
	for axis := range meta.Shape {
		tile := meta.TileShape[axis]
		first[axis] = region.Start[axis] / tile
		last[axis] = (region.End[axis] + tile - 1) / tile
	}
	var indices []uint64
	coords := append([]uint64(nil), first...)
	for {
		index := uint64(0)
		for axis := range meta.TileCounts {
			if index > math.MaxUint64/meta.TileCounts[axis] {
				return nil, werrors.NewLimitExceeded("float tile index overflows u64")
			}
			index = index*meta.TileCounts[axis] + coords[axis]
		}
		if index >= uint64(len(meta.Tiles)) {
			return nil, werrors.NewCorrupt("float tile index is outside the descriptor")
		}
		indices = append(indices, index)
		axis := len(first)
		for {
			if axis == 0 {
				return indices, nil
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

// placeFloatTileRegion places the intersection of region and a decoded tile
// into out.
func placeFloatTileRegion(
	out []byte, region tensor.Region, meta FloatTiledStreamsMeta,
	entry FloatTiledStreamsEntry, tile []byte,
) error {
	coords := tileCoords(meta.TileCounts, entry.Index)
	elementSize := meta.Dtype.ElementSize()
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
		return err
	}
	return copyBoxInto(out, region.Extent(), dstStart, boxed, localExtent, elementSize)
}
