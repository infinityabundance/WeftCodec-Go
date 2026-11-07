// Package frame implements the WFT1 frame layout
// (WEFTCODEC-FORMAT §2-§3, §5-§6).
//
//	header (96) || dependency table (48/entry) || chunk table (48/entry) || body || trailer (4)
//
// Parsing uses checked arithmetic and never panics on untrusted bytes. The
// header CRC is verified before any length field is trusted, and the trailer
// CRC covers every byte before it.
package frame

import (
	"encoding/binary"
	"fmt"
	"io"
	"strings"
	"sync"

	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

const (
	// Magic is the frame magic.
	Magic = "WFT1"
	// HeaderLen is the fixed header length.
	HeaderLen = 96
	// TrailerLen is the trailer length.
	TrailerLen = 4
	// DepEntryLen is the dependency table entry length.
	DepEntryLen = 48
	// ChunkEntryLen is the chunk table entry length.
	ChunkEntryLen = 48
	// FormatMajor is the supported major version.
	FormatMajor = 1
	// FormatMinor is the supported minor version.
	FormatMinor = 0
)

// Header domains.
const (
	DomainPayload    uint8 = 0
	DomainStructured uint8 = 1
)

// Dependency domains.
const (
	DepDomainPayload    uint8 = 0
	DepDomainStructured uint8 = 1
	DepDomainDictionary uint8 = 2
)

// DepFlagRequired marks a dependency as required for decoding.
const DepFlagRequired uint8 = 1 << 0

// Header flag bits.
const (
	FlagChunked       uint16 = 1 << 0
	FlagSelfContained uint16 = 1 << 1
	FlagDictRaw       uint16 = 1 << 2
	FlagOriginalHash  uint16 = 1 << 3
)

// Header feature bits.
const (
	FeatureDependencies uint16 = 1 << 0
	FeatureChunkHashes  uint16 = 1 << 1
)

// Mode is a payload encoding mode.
type Mode uint8

const (
	ModeRaw        Mode = 0
	ModeZstd       Mode = 1
	ModeReference  Mode = 2
	ModeDelta      Mode = 3
	ModeDictionary Mode = 4
	ModeStructured Mode = 5
	// ModeTensor is the canonical tensor mode (WEFT-AI-CODEC §2).
	ModeTensor Mode = 6
	// ModeTokenSequence is the token-id sequence mode (WEFT-AI-CODEC §3).
	ModeTokenSequence Mode = 7
	// ModeQuantizedTensor is the affine quantized tensor mode (WEFT-AI-CODEC §4).
	ModeQuantizedTensor Mode = 8
	// ModeFloatTensor is the split float tensor mode (WEFT-AI-CODEC §5).
	ModeFloatTensor Mode = 9
	// ModeTensorSharded is the sharded quantized tensor mode (WEFT-AI-CODEC §11).
	ModeTensorSharded Mode = 10
	// ModeWeightDelta is the exact tensor delta mode (WEFT-AI-CODEC §12).
	ModeWeightDelta Mode = 11
	// ModeKvCache is the paged KV cache page mode (WEFT-AI-CODEC §13).
	ModeKvCache Mode = 12
	// ModeActivationStream is the activation/gradient stream chunk mode
	// (WEFT-AI-CODEC §14).
	ModeActivationStream Mode = 13
	// ModeSemanticDelta is the semantically matched exact tensor delta mode
	// (WEFT-AI-CODEC §15).
	ModeSemanticDelta Mode = 14
	// ModeTensorTiled is the independently decodable tensor tile mode
	// (WEFT-AI-CODEC §16).
	ModeTensorTiled Mode = 15
	// ModeSparseDelta is the zero-run coded exact tensor delta mode
	// (WEFT-AI-CODEC §17).
	ModeSparseDelta Mode = 16
	// ModeFloatTiled is the independently decodable float-tile mode
	// (WEFT-AI-CODEC §18).
	ModeFloatTiled Mode = 17
	// ModeFloatTiledStreams is the independently decodable multi-stream
	// float-tile mode (WEFT-AI-CODEC §19).
	ModeFloatTiledStreams Mode = 18
)

// String returns the lowercase mode name.
func (m Mode) String() string {
	switch m {
	case ModeRaw:
		return "raw"
	case ModeZstd:
		return "zstd"
	case ModeReference:
		return "reference"
	case ModeDelta:
		return "delta"
	case ModeDictionary:
		return "dictionary"
	case ModeStructured:
		return "structured"
	case ModeTensor:
		return "tensor"
	case ModeTokenSequence:
		return "tokens"
	case ModeQuantizedTensor:
		return "quantized_tensor"
	case ModeFloatTensor:
		return "float_tensor"
	case ModeTensorSharded:
		return "tensor_sharded"
	case ModeWeightDelta:
		return "weight_delta"
	case ModeKvCache:
		return "kv_cache"
	case ModeActivationStream:
		return "activation_stream"
	case ModeSemanticDelta:
		return "semantic_delta"
	case ModeTensorTiled:
		return "tensor_tiled"
	case ModeSparseDelta:
		return "sparse_delta"
	case ModeFloatTiled:
		return "float_tiled"
	case ModeFloatTiledStreams:
		return "float_tiled_streams"
	default:
		return fmt.Sprintf("mode(%d)", uint8(m))
	}
}

// ModeFromU8 validates a wire mode value.
func ModeFromU8(v uint8) (Mode, error) {
	switch {
	case v <= 18:
		return Mode(v), nil
	case v <= 127:
		return 0, werrors.UnsupportedModef("reserved mode %d is not defined in v1", v)
	default:
		return 0, werrors.UnsupportedModef("experimental mode %d must not be persisted", v)
	}
}

// ModeFromName parses a CLI mode name (case-insensitive).
func ModeFromName(name string) (Mode, error) {
	switch strings.ToLower(name) {
	case "raw":
		return ModeRaw, nil
	case "zstd":
		return ModeZstd, nil
	case "reference":
		return ModeReference, nil
	case "delta":
		return ModeDelta, nil
	case "dictionary":
		return ModeDictionary, nil
	case "structured":
		return ModeStructured, nil
	case "tensor":
		return ModeTensor, nil
	case "tokens", "token_sequence":
		return ModeTokenSequence, nil
	case "quantized", "quantized_tensor":
		return ModeQuantizedTensor, nil
	case "float_tensor", "float-tensor":
		return ModeFloatTensor, nil
	case "tensor_sharded", "tensor-sharded", "sharded":
		return ModeTensorSharded, nil
	case "weight_delta", "weight-delta", "weightdelta":
		return ModeWeightDelta, nil
	case "kv_cache", "kv-cache", "kvcache":
		return ModeKvCache, nil
	case "activation_stream", "activation-stream", "activation":
		return ModeActivationStream, nil
	case "semantic_delta", "semantic-delta", "semanticdelta":
		return ModeSemanticDelta, nil
	case "tensor_tiled", "tensor-tiled", "tiled":
		return ModeTensorTiled, nil
	case "sparse_delta", "sparse-delta", "sparsedelta":
		return ModeSparseDelta, nil
	case "float_tiled", "float-tiled", "float_tile", "ftiled":
		return ModeFloatTiled, nil
	case "float_tiled_streams", "float-tiled-streams", "float_tile_streams", "ftiled_streams", "tiled_streams":
		return ModeFloatTiledStreams, nil
	default:
		return 0, werrors.UnsupportedModef("unknown mode %q", name)
	}
}

// Dependency is a parsed dependency table entry.
type Dependency struct {
	Hash       hash.Hash32
	DecodedLen uint64
	DepDomain  uint8
	DepFlags   uint8
	CodecHint  uint32
}

// NewDependency builds a v1 dependency entry with REQUIRED set.
func NewDependency(h hash.Hash32, decodedLen uint64, depDomain uint8, codecHint uint32) Dependency {
	return Dependency{
		Hash:       h,
		DecodedLen: decodedLen,
		DepDomain:  depDomain,
		DepFlags:   DepFlagRequired,
		CodecHint:  codecHint,
	}
}

// ChunkEntry is a parsed chunk table entry.
type ChunkEntry struct {
	EncodedOff uint64
	EncodedLen uint32
	DecodedLen uint32
	Hash       hash.Hash32
}

// Header is a parsed frame header.
type Header struct {
	FormatMajor uint8
	FormatMinor uint8
	Domain      uint8
	Mode        Mode
	Flags       uint16
	Level       uint16
	DecodedLen  uint64
	BodyLen     uint64
	DepCount    uint32
	ChunkCount  uint32
	Features    uint16
	ContentHash hash.Hash32
	TableLen    uint64
}

// FlagNames lists the set flag names.
func (h Header) FlagNames() []string {
	names := make([]string, 0, 4)
	if h.Flags&FlagChunked != 0 {
		names = append(names, "CHUNKED")
	}
	if h.Flags&FlagSelfContained != 0 {
		names = append(names, "SELF_CONTAINED")
	}
	if h.Flags&FlagDictRaw != 0 {
		names = append(names, "DICT_RAW")
	}
	if h.Flags&FlagOriginalHash != 0 {
		names = append(names, "ORIGINAL_HASH")
	}
	return names
}

// FeatureNames lists the set feature names.
func (h Header) FeatureNames() []string {
	names := make([]string, 0, 2)
	if h.Features&FeatureDependencies != 0 {
		names = append(names, "DEPENDENCIES")
	}
	if h.Features&FeatureChunkHashes != 0 {
		names = append(names, "CHUNK_HASHES")
	}
	return names
}

// IsChunked reports whether the CHUNKED flag is set.
func (h Header) IsChunked() bool { return h.Flags&FlagChunked != 0 }

// Frame is a parsed frame. Body and Bytes alias the parsed input.
type Frame struct {
	Header       Header
	Dependencies []Dependency
	Chunks       []ChunkEntry
	Body         []byte
	Bytes        []byte
}

// Parts is the input to Encode.
type Parts struct {
	FormatMajor  uint8
	FormatMinor  uint8
	Domain       uint8
	Mode         Mode
	Flags        uint16
	Level        uint16
	DecodedLen   uint64
	ContentHash  hash.Hash32
	Dependencies []Dependency
	Chunks       []ChunkEntry
	Body         []byte
}

// PayloadParts builds payload-domain parts with no dependencies or chunks.
func PayloadParts(mode Mode, flags uint16, level uint16, decodedLen uint64, contentHash hash.Hash32, body []byte) Parts {
	return Parts{
		FormatMajor: FormatMajor,
		FormatMinor: FormatMinor,
		Domain:      DomainPayload,
		Mode:        mode,
		Flags:       flags,
		Level:       level,
		DecodedLen:  decodedLen,
		ContentHash: contentHash,
		Body:        body,
	}
}

// Encode builds a frame. Features are derived from the tables, and the result
// is re-parsed with lim before being returned.
func Encode(parts Parts, lim limits.Limits) ([]byte, error) {
	if uint64(len(parts.Dependencies)) > uint64(^uint32(0)) {
		return nil, werrors.NewLimitExceeded("dependency count does not fit u32")
	}
	if uint64(len(parts.Chunks)) > uint64(^uint32(0)) {
		return nil, werrors.NewLimitExceeded("chunk count does not fit u32")
	}
	depCount := uint32(len(parts.Dependencies))
	chunkCount := uint32(len(parts.Chunks))
	depTable, err := limits.MulChecked(uint64(depCount), DepEntryLen, "dependency table")
	if err != nil {
		return nil, err
	}
	chunkTable, err := limits.MulChecked(uint64(chunkCount), ChunkEntryLen, "chunk table")
	if err != nil {
		return nil, err
	}
	tableLen, err := limits.AddChecked(depTable, chunkTable, "table")
	if err != nil {
		return nil, err
	}
	bodyLen := uint64(len(parts.Body))

	var features uint16
	if len(parts.Dependencies) != 0 {
		features |= FeatureDependencies
	}
	if len(parts.Chunks) != 0 {
		features |= FeatureChunkHashes
	}

	total, err := limits.AddChecked(HeaderLen, tableLen, "frame")
	if err != nil {
		return nil, err
	}
	total, err = limits.AddChecked(total, bodyLen, "frame")
	if err != nil {
		return nil, err
	}
	total, err = limits.AddChecked(total, TrailerLen, "frame")
	if err != nil {
		return nil, err
	}
	if err := limits.CheckLen(total, lim.MaxFrameBytes, "frame"); err != nil {
		return nil, err
	}
	if err := limits.CheckLen(parts.DecodedLen, lim.MaxDecodedBytes, "decoded data"); err != nil {
		return nil, err
	}
	totalInt, err := limits.ToInt(total, "frame")
	if err != nil {
		return nil, err
	}

	header := make([]byte, HeaderLen)
	copy(header[0:4], Magic)
	header[4] = parts.FormatMajor
	header[5] = parts.FormatMinor
	header[6] = parts.Domain
	header[7] = uint8(parts.Mode)
	binary.LittleEndian.PutUint16(header[8:10], parts.Flags)
	binary.LittleEndian.PutUint16(header[10:12], parts.Level)
	binary.LittleEndian.PutUint64(header[12:20], parts.DecodedLen)
	binary.LittleEndian.PutUint64(header[20:28], bodyLen)
	binary.LittleEndian.PutUint32(header[28:32], depCount)
	binary.LittleEndian.PutUint32(header[32:36], chunkCount)
	binary.LittleEndian.PutUint16(header[36:38], features)
	binary.LittleEndian.PutUint64(header[76:84], tableLen)
	copy(header[40:72], parts.ContentHash[:])

	headerCRC := crc32c(header[0:72], header[76:96])
	binary.LittleEndian.PutUint32(header[72:76], headerCRC)

	out := make([]byte, 0, totalInt)
	out = append(out, header...)
	for _, dep := range parts.Dependencies {
		entry := make([]byte, DepEntryLen)
		copy(entry[0:32], dep.Hash[:])
		binary.LittleEndian.PutUint64(entry[32:40], dep.DecodedLen)
		entry[40] = dep.DepDomain
		entry[41] = dep.DepFlags
		binary.LittleEndian.PutUint32(entry[44:48], dep.CodecHint)
		out = append(out, entry...)
	}
	for _, chunk := range parts.Chunks {
		entry := make([]byte, ChunkEntryLen)
		binary.LittleEndian.PutUint64(entry[0:8], chunk.EncodedOff)
		binary.LittleEndian.PutUint32(entry[8:12], chunk.EncodedLen)
		binary.LittleEndian.PutUint32(entry[12:16], chunk.DecodedLen)
		copy(entry[16:48], chunk.Hash[:])
		out = append(out, entry...)
	}
	out = append(out, parts.Body...)
	trailer := crc32c(out)
	var trailerBuf [4]byte
	binary.LittleEndian.PutUint32(trailerBuf[:], trailer)
	out = append(out, trailerBuf[:]...)

	if _, err := Parse(out, lim); err != nil {
		return nil, err
	}
	return out, nil
}

// ParseDefault parses with default limits.
func ParseDefault(b []byte) (*Frame, error) { return Parse(b, limits.Default()) }

// Parse parses and structurally validates a frame.
func Parse(b []byte, lim limits.Limits) (*Frame, error) {
	if len(b) < HeaderLen+TrailerLen {
		return nil, werrors.Malformedf(
			"frame length %d is below the minimum %d", len(b), HeaderLen+TrailerLen)
	}
	if err := limits.CheckLen(uint64(len(b)), lim.MaxFrameBytes, "frame"); err != nil {
		return nil, err
	}
	if string(b[0:4]) != Magic {
		return nil, werrors.NewMalformed("bad frame magic (expected WFT1)")
	}
	formatMajor := b[4]
	formatMinor := b[5]
	if formatMajor != FormatMajor {
		return nil, werrors.UnsupportedVersionf("unsupported format major %d", formatMajor)
	}
	if formatMinor != FormatMinor {
		return nil, werrors.UnsupportedVersionf("unsupported format minor %d", formatMinor)
	}

	storedHeaderCRC := binary.LittleEndian.Uint32(b[72:76])
	if crc32c(b[0:72], b[76:96]) != storedHeaderCRC {
		return nil, werrors.NewCorrupt("header CRC32C mismatch")
	}

	domain := b[6]
	mode, err := ModeFromU8(b[7])
	if err != nil {
		return nil, err
	}
	flags := binary.LittleEndian.Uint16(b[8:10])
	level := binary.LittleEndian.Uint16(b[10:12])
	decodedLen := binary.LittleEndian.Uint64(b[12:20])
	bodyLen := binary.LittleEndian.Uint64(b[20:28])
	depCount := binary.LittleEndian.Uint32(b[28:32])
	chunkCount := binary.LittleEndian.Uint32(b[32:36])
	features := binary.LittleEndian.Uint16(b[36:38])
	reserved0 := binary.LittleEndian.Uint16(b[38:40])
	contentHash, err := hash.FromSlice(b[40:72])
	if err != nil {
		return nil, err
	}
	tableLen := binary.LittleEndian.Uint64(b[76:84])
	reserved1 := binary.LittleEndian.Uint32(b[84:88])
	reserved2 := binary.LittleEndian.Uint64(b[88:96])

	if reserved0 != 0 || reserved1 != 0 || reserved2 != 0 {
		return nil, werrors.NewMalformed("reserved header fields must be zero")
	}
	if domain > DomainStructured {
		return nil, werrors.Malformedf("invalid domain value %d", domain)
	}
	const knownFlags = FlagChunked | FlagSelfContained | FlagDictRaw | FlagOriginalHash
	if flags&^uint16(knownFlags) != 0 {
		return nil, werrors.UnsupportedFeaturef(
			"unknown critical flag bits 0x%04x", flags&^uint16(0x000f))
	}
	if flags&FlagOriginalHash != 0 {
		return nil, werrors.NewUnsupportedFeature("ORIGINAL_HASH flag is reserved in v1")
	}
	if flags&FlagSelfContained != 0 {
		return nil, werrors.NewUnsupportedFeature("SELF_CONTAINED is reserved and not produced in v1")
	}
	const knownFeatures = FeatureDependencies | FeatureChunkHashes
	if features&^uint16(knownFeatures) != 0 {
		return nil, werrors.UnsupportedFeaturef(
			"unknown critical feature bits 0x%04x", features&^uint16(0x0003))
	}

	if err := limits.CheckLen(decodedLen, lim.MaxDecodedBytes, "decoded data"); err != nil {
		return nil, err
	}
	if err := limits.CheckLen(uint64(depCount), lim.MaxDependencyCount, "dependency count"); err != nil {
		return nil, err
	}
	if err := limits.CheckLen(uint64(chunkCount), lim.MaxChunks, "chunk count"); err != nil {
		return nil, err
	}

	expectedDepTable, err := limits.MulChecked(uint64(depCount), DepEntryLen, "dependency table")
	if err != nil {
		return nil, werrors.NewMalformed(err.Error())
	}
	expectedChunkTable, err := limits.MulChecked(uint64(chunkCount), ChunkEntryLen, "chunk table")
	if err != nil {
		return nil, werrors.NewMalformed(err.Error())
	}
	expectedTableLen, err := limits.AddChecked(expectedDepTable, expectedChunkTable, "table")
	if err != nil {
		return nil, werrors.NewMalformed(err.Error())
	}
	if tableLen != expectedTableLen {
		return nil, werrors.Malformedf(
			"table_len %d does not match %d dependencies and %d chunks",
			tableLen, depCount, chunkCount)
	}

	expectedTotal, err := limits.AddChecked(HeaderLen, tableLen, "frame")
	if err != nil {
		return nil, werrors.NewMalformed(err.Error())
	}
	expectedTotal, err = limits.AddChecked(expectedTotal, bodyLen, "frame")
	if err != nil {
		return nil, werrors.NewMalformed(err.Error())
	}
	expectedTotal, err = limits.AddChecked(expectedTotal, TrailerLen, "frame")
	if err != nil {
		return nil, werrors.NewMalformed(err.Error())
	}
	if expectedTotal != uint64(len(b)) {
		return nil, werrors.Malformedf(
			"frame length %d does not match header-derived length %d", len(b), expectedTotal)
	}

	chunked := flags&FlagChunked != 0
	if chunked {
		if chunkCount == 0 {
			return nil, werrors.NewMalformed("CHUNKED set with an empty chunk table")
		}
		if features&FeatureChunkHashes == 0 {
			return nil, werrors.NewMalformed("CHUNKED set without the CHUNK_HASHES feature")
		}
	} else {
		if chunkCount != 0 {
			return nil, werrors.NewMalformed("chunk table present without the CHUNKED flag")
		}
		if features&FeatureChunkHashes != 0 {
			return nil, werrors.NewMalformed("CHUNK_HASHES set without the CHUNKED flag")
		}
	}
	if depCount > 0 && features&FeatureDependencies == 0 {
		return nil, werrors.NewMalformed("dependency table present without the DEPENDENCIES feature")
	}

	bodyStart, err := limits.ToInt(uint64(HeaderLen)+tableLen, "body offset")
	if err != nil {
		return nil, werrors.NewMalformed(err.Error())
	}
	bodyLenInt, err := limits.ToInt(bodyLen, "body")
	if err != nil {
		return nil, werrors.NewMalformed(err.Error())
	}
	bodyEnd, err := limits.AddChecked(uint64(bodyStart), uint64(bodyLenInt), "body end")
	if err != nil {
		return nil, werrors.NewMalformed(err.Error())
	}
	if bodyEnd != uint64(len(b))-TrailerLen {
		return nil, werrors.NewMalformed("body extent does not end at the trailer")
	}
	body := b[bodyStart : bodyStart+bodyLenInt]

	trailerAt := len(b) - TrailerLen
	storedTrailer := binary.LittleEndian.Uint32(b[trailerAt:])
	if crc32c(b[:trailerAt]) != storedTrailer {
		return nil, werrors.NewCorrupt("frame trailer CRC32C mismatch")
	}

	dependencies := make([]Dependency, 0, minInt(int(depCount), 256))
	for i := 0; i < int(depCount); i++ {
		off := HeaderLen + i*DepEntryLen
		h, err := hash.FromSlice(b[off : off+32])
		if err != nil {
			return nil, err
		}
		depLen := binary.LittleEndian.Uint64(b[off+32 : off+40])
		depDomain := b[off+40]
		depFlags := b[off+41]
		reserved := binary.LittleEndian.Uint16(b[off+42 : off+44])
		codecHint := binary.LittleEndian.Uint32(b[off+44 : off+48])
		if depFlags&^uint8(DepFlagRequired) != 0 {
			return nil, werrors.Malformedf("dependency %d has unknown flag bits", i)
		}
		if depFlags&DepFlagRequired == 0 {
			return nil, werrors.Malformedf("dependency %d does not set REQUIRED", i)
		}
		if reserved != 0 {
			return nil, werrors.Malformedf("dependency %d reserved field must be zero", i)
		}
		if depDomain > DepDomainDictionary {
			return nil, werrors.Malformedf("dependency %d has invalid domain %d", i, depDomain)
		}
		if err := limits.CheckLen(depLen, lim.MaxDecodedBytes, "dependency decoded data"); err != nil {
			return nil, err
		}
		dependencies = append(dependencies, Dependency{
			Hash:       h,
			DecodedLen: depLen,
			DepDomain:  depDomain,
			DepFlags:   depFlags,
			CodecHint:  codecHint,
		})
	}

	chunkTableStart := HeaderLen + int(depCount)*DepEntryLen
	chunks := make([]ChunkEntry, 0, minInt(int(chunkCount), 256))
	for i := 0; i < int(chunkCount); i++ {
		off := chunkTableStart + i*ChunkEntryLen
		h, err := hash.FromSlice(b[off+16 : off+48])
		if err != nil {
			return nil, err
		}
		chunks = append(chunks, ChunkEntry{
			EncodedOff: binary.LittleEndian.Uint64(b[off : off+8]),
			EncodedLen: binary.LittleEndian.Uint32(b[off+8 : off+12]),
			DecodedLen: binary.LittleEndian.Uint32(b[off+12 : off+16]),
			Hash:       h,
		})
	}

	if chunked {
		var nextOff uint64
		var decodedSum uint64
		for i, chunk := range chunks {
			if chunk.EncodedOff != nextOff {
				return nil, werrors.Malformedf(
					"chunk %d starts at %d but expected %d (gap or overlap)",
					i, chunk.EncodedOff, nextOff)
			}
			if chunk.EncodedLen == 0 {
				return nil, werrors.Malformedf("chunk %d has zero encoded length", i)
			}
			nextOff, err = limits.AddChecked(nextOff, uint64(chunk.EncodedLen), "chunk extent")
			if err != nil {
				return nil, werrors.NewMalformed(err.Error())
			}
			decodedSum, err = limits.AddChecked(decodedSum, uint64(chunk.DecodedLen), "chunk decoded length sum")
			if err != nil {
				return nil, werrors.NewMalformed(err.Error())
			}
		}
		if nextOff != bodyLen {
			return nil, werrors.Malformedf(
				"chunk extents end at %d but body_len is %d", nextOff, bodyLen)
		}
		if decodedSum != decodedLen {
			return nil, werrors.Malformedf(
				"chunk decoded lengths sum to %d but decoded_len is %d", decodedSum, decodedLen)
		}
	}

	header := Header{
		FormatMajor: formatMajor,
		FormatMinor: formatMinor,
		Domain:      domain,
		Mode:        mode,
		Flags:       flags,
		Level:       level,
		DecodedLen:  decodedLen,
		BodyLen:     bodyLen,
		DepCount:    depCount,
		ChunkCount:  chunkCount,
		Features:    features,
		ContentHash: contentHash,
		TableLen:    tableLen,
	}
	if err := validateModeRules(header, dependencies, lim); err != nil {
		return nil, err
	}

	return &Frame{
		Header:       header,
		Dependencies: dependencies,
		Chunks:       chunks,
		Body:         body,
		Bytes:        b,
	}, nil
}

func validateModeRules(h Header, deps []Dependency, lim limits.Limits) error {
	dictFlag := h.Flags&FlagDictRaw != 0
	switch h.Mode {
	case ModeRaw:
		if h.Domain != DomainPayload {
			return werrors.NewMalformed("RAW frame must use the payload domain")
		}
		if dictFlag {
			return werrors.NewMalformed("DICT_RAW is only valid for DICTIONARY")
		}
		if h.DepCount != 0 {
			return werrors.NewMalformed("RAW frame must not have dependencies")
		}
		if h.BodyLen != h.DecodedLen {
			return werrors.NewMalformed("RAW body_len must equal decoded_len")
		}
	case ModeZstd:
		if h.Domain != DomainPayload {
			return werrors.NewMalformed("ZSTD frame must use the payload domain")
		}
		if dictFlag {
			return werrors.NewMalformed("DICT_RAW is only valid for DICTIONARY")
		}
		if h.DepCount != 0 {
			return werrors.NewMalformed("ZSTD frame must not have dependencies in v1")
		}
	case ModeReference:
		if h.Domain != DomainPayload {
			return werrors.NewMalformed("REFERENCE frame must use the payload domain")
		}
		if dictFlag {
			return werrors.NewMalformed("DICT_RAW is only valid for DICTIONARY")
		}
		if h.BodyLen != 0 {
			return werrors.NewMalformed("REFERENCE body must be empty")
		}
		if h.DepCount != 1 {
			return werrors.NewMalformed("REFERENCE requires exactly one dependency")
		}
		if h.ChunkCount != 0 {
			return werrors.NewMalformed("REFERENCE must not be chunked")
		}
		if h.DecodedLen != deps[0].DecodedLen {
			return werrors.NewMalformed("REFERENCE decoded_len must equal dependency decoded_len")
		}
	case ModeDelta:
		if h.Domain != DomainPayload {
			return werrors.NewMalformed("DELTA frame must use the payload domain")
		}
		if dictFlag {
			return werrors.NewMalformed("DICT_RAW is only valid for DICTIONARY")
		}
		if h.DepCount != 1 {
			return werrors.NewMalformed("DELTA requires exactly one dependency")
		}
		if h.ChunkCount != 0 {
			return werrors.NewMalformed("DELTA must not be chunked")
		}
	case ModeTensor, ModeTokenSequence, ModeQuantizedTensor, ModeFloatTensor, ModeTensorSharded, ModeKvCache, ModeActivationStream, ModeTensorTiled, ModeFloatTiled, ModeFloatTiledStreams:
		if h.Domain != DomainPayload {
			return werrors.NewMalformed(
				"AI-mode frames must use the payload domain")
		}
		if dictFlag {
			return werrors.NewMalformed("DICT_RAW is only valid for DICTIONARY")
		}
		if h.DepCount != 0 {
			return werrors.NewMalformed(
				"AI-mode frames must not have dependencies in v1")
		}
		if h.ChunkCount != 0 {
			return werrors.NewMalformed(
				"AI-mode frames must not be chunked")
		}
		if h.BodyLen < 4 {
			return werrors.NewMalformed(
				"AI-mode body is shorter than its 4-byte metadata length")
		}
	case ModeWeightDelta, ModeSemanticDelta, ModeSparseDelta:
		modeName := "WEIGHT_DELTA"
		switch h.Mode {
		case ModeSemanticDelta:
			modeName = "SEMANTIC_DELTA"
		case ModeSparseDelta:
			modeName = "SPARSE_DELTA"
		}
		if h.Domain != DomainPayload {
			return werrors.Malformedf(
				"%s frame must use the payload domain", modeName)
		}
		if dictFlag {
			return werrors.NewMalformed("DICT_RAW is only valid for DICTIONARY")
		}
		if h.DepCount != 1 {
			return werrors.Malformedf(
				"%s requires exactly one dependency", modeName)
		}
		if h.ChunkCount != 0 {
			return werrors.Malformedf(
				"%s must not be chunked", modeName)
		}
		if h.BodyLen < 4 {
			return werrors.Malformedf(
				"%s body is shorter than its 4-byte metadata length", modeName)
		}
		if deps[0].DepDomain != DepDomainPayload {
			return werrors.Malformedf(
				"%s dependency must use dep_domain 0 (payload)", modeName)
		}
		if h.DecodedLen != deps[0].DecodedLen {
			return werrors.Malformedf(
				"%s decoded_len must equal the base decoded_len", modeName)
		}
	case ModeDictionary:
		if h.Domain != DomainPayload {
			return werrors.NewMalformed("DICTIONARY frame must use the payload domain")
		}
		if !dictFlag {
			return werrors.NewMalformed("DICTIONARY requires the DICT_RAW flag")
		}
		if h.DepCount != 1 {
			return werrors.NewMalformed("DICTIONARY requires exactly one dependency")
		}
		if deps[0].DepDomain != DepDomainDictionary {
			return werrors.NewMalformed(
				"DICTIONARY dependency must use dep_domain 2 (dictionary bytes)")
		}
		if h.ChunkCount != 0 {
			return werrors.NewMalformed("DICTIONARY must not be chunked")
		}
		if err := limits.CheckLen(deps[0].DecodedLen, lim.MaxDictionaryBytes, "dictionary"); err != nil {
			return err
		}
	case ModeStructured:
		if h.Domain != DomainStructured {
			return werrors.NewMalformed("STRUCTURED frame must use the structured domain")
		}
		if dictFlag {
			return werrors.NewMalformed("DICT_RAW is only valid for DICTIONARY")
		}
		if h.DepCount != 0 {
			return werrors.NewMalformed("STRUCTURED must not have dependencies in v1")
		}
		if h.ChunkCount != 0 {
			return werrors.NewMalformed("STRUCTURED must not be chunked")
		}
		if h.BodyLen != h.DecodedLen {
			return werrors.NewMalformed("STRUCTURED body_len must equal decoded_len")
		}
	}
	return nil
}

var readBufferPool = sync.Pool{
	New: func() any {
		b := make([]byte, 64<<10)
		return &b
	},
}

// ReadFrame reads at most MaxFrameBytes from r and returns the frame bytes.
// It is the io.Reader-friendly entry point; excess input is LimitExceeded.
func ReadFrame(r io.Reader, lim limits.Limits) ([]byte, error) {
	max := lim.MaxFrameBytes
	// io.LimitReader takes an int64. max_frame_bytes is far below MaxInt64 in
	// practice, but never overflow if a caller lowers/raises it.
	limit := int64(max)
	if max >= uint64(^uint64(0)>>1) {
		limit = int64(^uint64(0) >> 1)
	}
	lr := io.LimitReader(r, limit+1)
	bufPtr := readBufferPool.Get().(*[]byte)
	buf := *bufPtr
	out := make([]byte, 0, 64<<10)
	for {
		n, err := lr.Read(buf)
		if n > 0 {
			if uint64(len(out))+uint64(n) > max {
				readBufferPool.Put(bufPtr)
				return nil, werrors.LimitExceededf("frame exceeds limit %d bytes", max)
			}
			out = append(out, buf[:n]...)
		}
		if err == io.EOF {
			readBufferPool.Put(bufPtr)
			return out, nil
		}
		if err != nil {
			readBufferPool.Put(bufPtr)
			return nil, werrors.NewIo(fmt.Sprintf("cannot read frame: %v", err))
		}
	}
}

// WriteFrame writes frame bytes to w.
func WriteFrame(w io.Writer, b []byte) error {
	if _, err := w.Write(b); err != nil {
		return werrors.NewIo(fmt.Sprintf("cannot write frame: %v", err))
	}
	return nil
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
