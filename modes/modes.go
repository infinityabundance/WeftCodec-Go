// Package modes implements the v1 payload modes (WEFTCODEC-FORMAT §3-§6):
// RAW, ZSTD, REFERENCE, DELTA, DICTIONARY, and STRUCTURED.
//
// Every encoder computes complete frame bytes including header, tables, and
// trailer. Every decoder verifies the header/trailer, every chunk hash, the
// dependency hash and length at the materialization boundary, and the whole
// content hash.
package modes

import (
	"context"
	"fmt"
	"math"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// EncodeRaw encodes data as RAW. chunkSize == 0 means no chunking; a positive
// value requests independent literal chunks of that size.
func EncodeRaw(data []byte, chunkSize int, lim limits.Limits) ([]byte, error) {
	if err := limits.CheckLen(uint64(len(data)), lim.MaxDecodedBytes, "decoded data"); err != nil {
		return nil, err
	}
	if err := validateChunkSize(chunkSize); err != nil {
		return nil, err
	}
	contentHash := hash.Domain(hash.DomainChunk, data)
	body, chunks, chunked, err := maybeChunked(data, chunkSize, lim, func(chunk []byte) ([]byte, error) {
		out := make([]byte, len(chunk))
		copy(out, chunk)
		return out, nil
	})
	if err != nil {
		return nil, err
	}
	flags := uint16(0)
	if chunked {
		flags = frame.FlagChunked
	} else {
		body = append([]byte(nil), data...)
	}
	parts := frame.PayloadParts(frame.ModeRaw, flags, 0, uint64(len(data)), contentHash, body)
	parts.Chunks = chunks
	return frame.Encode(parts, lim)
}

// EncodeZstd encodes data as one or more zstd frames at level.
func EncodeZstd(data []byte, level int, chunkSize int, lim limits.Limits) ([]byte, error) {
	if err := limits.CheckLen(uint64(len(data)), lim.MaxDecodedBytes, "decoded data"); err != nil {
		return nil, err
	}
	if err := validateChunkSize(chunkSize); err != nil {
		return nil, err
	}
	headerLevel, err := levelToU16(level)
	if err != nil {
		return nil, err
	}
	contentHash := hash.Domain(hash.DomainChunk, data)
	body, chunks, chunked, err := maybeChunked(data, chunkSize, lim, func(chunk []byte) ([]byte, error) {
		return compressZstd(chunk, level)
	})
	if err != nil {
		return nil, err
	}
	flags := uint16(0)
	if chunked {
		flags = frame.FlagChunked
	} else {
		body, err = compressZstd(data, level)
		if err != nil {
			return nil, err
		}
	}
	parts := frame.PayloadParts(frame.ModeZstd, flags, headerLevel, uint64(len(data)), contentHash, body)
	parts.Chunks = chunks
	return frame.Encode(parts, lim)
}

// EncodeReference encodes a frame whose plaintext equals base byte for byte.
func EncodeReference(data, base []byte, lim limits.Limits) ([]byte, error) {
	if !bytesEqual(data, base) {
		return nil, werrors.NewMalformed(
			"REFERENCE encoding requires the input to equal the supplied base")
	}
	if err := limits.CheckLen(uint64(len(data)), lim.MaxDecodedBytes, "decoded data"); err != nil {
		return nil, err
	}
	dep := dependencyFor(base, frame.DepDomainPayload, 0)
	parts := frame.PayloadParts(
		frame.ModeReference, 0, 0, uint64(len(data)),
		hash.Domain(hash.DomainChunk, data), nil)
	parts.Dependencies = []frame.Dependency{dep}
	return frame.Encode(parts, lim)
}

// EncodeDelta encodes data as a COPY/INSERT stream over base.
func EncodeDelta(data, base []byte, lim limits.Limits) ([]byte, error) {
	if err := limits.CheckLen(uint64(len(data)), lim.MaxDecodedBytes, "decoded data"); err != nil {
		return nil, err
	}
	if err := limits.CheckLen(uint64(len(base)), lim.MaxDecodedBytes, "delta base"); err != nil {
		return nil, err
	}
	body, err := EncodeDeltaOps(data, base, lim)
	if err != nil {
		return nil, err
	}
	dep := dependencyFor(base, frame.DepDomainPayload, 0)
	parts := frame.PayloadParts(
		frame.ModeDelta, 0, 0, uint64(len(data)),
		hash.Domain(hash.DomainChunk, data), body)
	parts.Dependencies = []frame.Dependency{dep}
	return frame.Encode(parts, lim)
}

// EncodeDictionary encodes data with dict as a raw-content zstd dictionary.
func EncodeDictionary(data, dict []byte, level int, lim limits.Limits) ([]byte, error) {
	if err := limits.CheckLen(uint64(len(data)), lim.MaxDecodedBytes, "decoded data"); err != nil {
		return nil, err
	}
	if err := limits.CheckLen(uint64(len(dict)), lim.MaxDictionaryBytes, "dictionary"); err != nil {
		return nil, err
	}
	headerLevel, err := levelToU16(level)
	if err != nil {
		return nil, err
	}
	body, err := compressZstdWithDictionary(data, dict, level)
	if err != nil {
		return nil, err
	}
	dep := dependencyFor(dict, frame.DepDomainDictionary, 0)
	parts := frame.PayloadParts(
		frame.ModeDictionary, frame.FlagDictRaw, headerLevel, uint64(len(data)),
		hash.Domain(hash.DomainChunk, data), body)
	parts.Dependencies = []frame.Dependency{dep}
	return frame.Encode(parts, lim)
}

// EncodeStructured encodes one canonical WCB value.
func EncodeStructured(v wcb.Value, lim limits.Limits) ([]byte, error) {
	body, err := wcb.Encode(v, lim)
	if err != nil {
		return nil, err
	}
	if err := limits.CheckLen(uint64(len(body)), lim.MaxDecodedBytes, "structured body"); err != nil {
		return nil, err
	}
	parts := frame.Parts{
		FormatMajor: frame.FormatMajor,
		FormatMinor: frame.FormatMinor,
		Domain:      frame.DomainStructured,
		Mode:        frame.ModeStructured,
		Flags:       0,
		Level:       0,
		DecodedLen:  uint64(len(body)),
		ContentHash: hash.Domain(hash.DomainChunk, body),
		Body:        body,
	}
	return frame.Encode(parts, lim)
}

// DecodeBytes decodes a complete frame, resolving dependencies through resolver.
// The returned bytes satisfy len(bytes) == decoded_len and
// H("weft/chunk/v1", bytes) == header.content_hash.
func DecodeBytes(ctx context.Context, frameBytes []byte, resolver dependency.Resolver, lim limits.Limits) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, werrors.NewIo(fmt.Sprintf("decode canceled: %v", err))
	}
	f, err := frame.Parse(frameBytes, lim)
	if err != nil {
		return nil, err
	}

	var plaintext []byte
	switch f.Header.Mode {
	case frame.ModeRaw:
		if f.Header.IsChunked() {
			plaintext, err = decodeChunks(f, lim)
		} else {
			plaintext = append([]byte(nil), f.Body...)
		}
	case frame.ModeZstd:
		if f.Header.IsChunked() {
			plaintext, err = decodeChunks(f, lim)
		} else {
			plaintext, err = decompressZstd(f.Body, f.Header.DecodedLen)
		}
	case frame.ModeReference:
		plaintext, err = resolveDependency(ctx, f.Dependencies[0], resolver)
	case frame.ModeDelta:
		var dep []byte
		dep, err = resolveDependency(ctx, f.Dependencies[0], resolver)
		if err == nil {
			plaintext, err = DecodeDeltaOps(f.Body, dep, f.Header.DecodedLen, lim)
		}
	case frame.ModeDictionary:
		var dict []byte
		dict, err = resolveDependency(ctx, f.Dependencies[0], resolver)
		if err == nil {
			plaintext, err = decompressZstdWithDictionary(f.Body, f.Header.DecodedLen, dict)
		}
	case frame.ModeStructured:
		// Validate canonical WCB before materializing the bytes.
		if _, err = wcb.Decode(f.Body, lim); err == nil {
			plaintext = append([]byte(nil), f.Body...)
		}
	case frame.ModeTensor:
		plaintext, err = DecodeTensorFrame(f, lim)
	case frame.ModeTokenSequence:
		plaintext, err = DecodeTokensFrame(f, lim)
	case frame.ModeQuantizedTensor:
		plaintext, err = DecodeQuantizedTensorFrame(f, lim)
	case frame.ModeFloatTensor:
		plaintext, err = DecodeFloatTensorFrame(f, lim)
	case frame.ModeTensorSharded:
		plaintext, err = DecodeShardedPlanes(f, lim)
	case frame.ModeWeightDelta:
		plaintext, err = DecodeWeightDeltaFrame(ctx, f, resolver, lim)
	case frame.ModeSemanticDelta:
		plaintext, err = DecodeSemanticDeltaFrame(ctx, f, resolver, lim)
	case frame.ModeSparseDelta:
		plaintext, err = DecodeSparseDeltaFrame(ctx, f, resolver, lim)
	case frame.ModeKvCache:
		var page DecodedKvPage
		page, err = DecodeKvPage(f, lim)
		if err == nil {
			plaintext = page.Bytes
		}
	case frame.ModeKvCacheProv:
		var page DecodedKvProvPage
		page, err = DecodeKvProvPage(f, lim)
		if err == nil {
			plaintext = page.Bytes
		}
	case frame.ModeActivationStream:
		var chunk DecodedActivationChunk
		chunk, err = DecodeActivationChunk(f, lim)
		if err == nil {
			plaintext = chunk.Bytes
		}
	case frame.ModeTensorTiled:
		plaintext, err = DecodeTiledTensor(f, lim)
	case frame.ModeFloatTiled:
		plaintext, err = DecodeFloatTiled(f, lim)
	case frame.ModeFloatTiledStreams:
		plaintext, err = DecodeFloatTiledStreams(f, lim, 1)
	case frame.ModeEngramShard:
		plaintext, err = DecodeEngramFrame(ctx, f, resolver, lim)
	default:
		err = werrors.UnsupportedModef("mode %d is not decodable", f.Header.Mode)
	}
	if err != nil {
		return nil, err
	}

	if uint64(len(plaintext)) != f.Header.DecodedLen {
		return nil, werrors.Corruptf(
			"decoded length %d differs from declared %d", len(plaintext), f.Header.DecodedLen)
	}
	if hash.Domain(hash.DomainChunk, plaintext) != f.Header.ContentHash {
		return nil, werrors.NewCorrupt("content hash mismatch")
	}
	return plaintext, nil
}

func decodeChunks(f *frame.Frame, lim limits.Limits) ([]byte, error) {
	capacity, err := limits.ToInt(f.Header.DecodedLen, "decoded length")
	if err != nil {
		return nil, err
	}
	if capacity > 1<<20 {
		capacity = 1 << 20
	}
	out := make([]byte, 0, capacity)
	for i := range f.Chunks {
		decoded, err := DecodeChunk(f, i)
		if err != nil {
			return nil, err
		}
		if err := limits.CheckLen(uint64(len(out)), lim.MaxDecodedBytes, "decoded data"); err != nil {
			return nil, err
		}
		out = append(out, decoded...)
	}
	return out, nil
}

// DecodeChunk decodes and hash-verifies one chunk independently, as if it were
// the body of a frame with the same mode. It is used by range reads so chunks
// outside the requested range are never decoded.
func DecodeChunk(f *frame.Frame, index int) ([]byte, error) {
	if index < 0 || index >= len(f.Chunks) {
		return nil, werrors.Malformedf("chunk index %d out of range", index)
	}
	chunk := f.Chunks[index]
	start, err := limits.ToInt(chunk.EncodedOff, "chunk offset")
	if err != nil {
		return nil, werrors.NewMalformed(err.Error())
	}
	end, err := limits.AddChecked(chunk.EncodedOff, uint64(chunk.EncodedLen), "chunk extent")
	if err != nil {
		return nil, werrors.NewMalformed(err.Error())
	}
	endInt, err := limits.ToInt(end, "chunk extent")
	if err != nil {
		return nil, werrors.NewMalformed(err.Error())
	}
	if start > len(f.Body) || endInt > len(f.Body) {
		return nil, werrors.Malformedf("chunk %d extends past body", index)
	}
	encoded := f.Body[start:endInt]

	var decoded []byte
	switch f.Header.Mode {
	case frame.ModeRaw:
		if uint64(len(encoded)) != uint64(chunk.DecodedLen) {
			return nil, werrors.Malformedf(
				"RAW chunk %d encoded_len does not equal decoded_len", index)
		}
		decoded = append([]byte(nil), encoded...)
	case frame.ModeZstd:
		decoded, err = decompressZstd(encoded, uint64(chunk.DecodedLen))
		if err != nil {
			return nil, err
		}
	default:
		return nil, werrors.NewMalformed("chunk table is only valid for RAW and ZSTD frames")
	}
	if uint64(len(decoded)) != uint64(chunk.DecodedLen) {
		return nil, werrors.Corruptf(
			"chunk %d decoded length %d differs from declared %d",
			index, len(decoded), chunk.DecodedLen)
	}
	if hash.Domain(hash.DomainChunk, decoded) != chunk.Hash {
		return nil, werrors.Corruptf("chunk %d hash mismatch", index)
	}
	return decoded, nil
}

// ResolveDependency resolves and verifies dependency bytes.
func ResolveDependency(ctx context.Context, dep frame.Dependency, resolver dependency.Resolver) ([]byte, error) {
	if resolver == nil {
		return nil, werrors.NewDependencyMissing("no resolver supplied")
	}
	if err := ctx.Err(); err != nil {
		return nil, werrors.NewIo(fmt.Sprintf("dependency resolution canceled: %v", err))
	}
	b, err := resolver.Resolve(ctx, dep.Hash, dep.DecodedLen, dep.DepDomain)
	if err != nil {
		return nil, err
	}
	if err := dependency.Verify(dep.Hash, dep.DecodedLen, b); err != nil {
		return nil, err
	}
	return b, nil
}

func resolveDependency(ctx context.Context, dep frame.Dependency, resolver dependency.Resolver) ([]byte, error) {
	return ResolveDependency(ctx, dep, resolver)
}

func dependencyFor(b []byte, depDomain uint8, codecHint uint32) frame.Dependency {
	return frame.NewDependency(
		hash.Domain(hash.DomainChunk, b), uint64(len(b)), depDomain, codecHint)
}

func validateChunkSize(chunkSize int) error {
	if chunkSize < 0 {
		return werrors.NewMalformed("chunk size must be greater than zero")
	}
	if chunkSize == 0 {
		return nil
	}
	if uint64(chunkSize) > math.MaxUint32 {
		return werrors.NewLimitExceeded(
			"chunk size does not fit the 32-bit encoded_len field")
	}
	return nil
}

func maybeChunked(
	data []byte,
	chunkSize int,
	lim limits.Limits,
	encodeChunk func([]byte) ([]byte, error),
) (body []byte, chunks []frame.ChunkEntry, chunked bool, err error) {
	if chunkSize <= 0 || len(data) == 0 {
		return nil, nil, false, nil
	}
	count := (len(data)-1)/chunkSize + 1
	if err := limits.CheckLen(uint64(count), lim.MaxChunks, "chunk count"); err != nil {
		return nil, nil, false, err
	}
	chunks = make([]frame.ChunkEntry, 0, min(count, 1024))
	var offset uint64
	for start := 0; start < len(data); start += chunkSize {
		end := start + chunkSize
		if end > len(data) {
			end = len(data)
		}
		chunk := data[start:end]
		encoded, err := encodeChunk(chunk)
		if err != nil {
			return nil, nil, false, err
		}
		if uint64(len(encoded)) > math.MaxUint32 {
			return nil, nil, false, werrors.NewLimitExceeded(
				"encoded chunk does not fit u32")
		}
		chunks = append(chunks, frame.ChunkEntry{
			EncodedOff: offset,
			EncodedLen: uint32(len(encoded)),
			DecodedLen: uint32(len(chunk)),
			Hash:       hash.Domain(hash.DomainChunk, chunk),
		})
		offset, err = limits.AddChecked(offset, uint64(len(encoded)), "chunk offset")
		if err != nil {
			return nil, nil, false, err
		}
		body = append(body, encoded...)
	}
	return body, chunks, true, nil
}

func levelToU16(level int) (uint16, error) {
	if level < 0 || level > math.MaxUint16 {
		return 0, werrors.NewLimitExceeded(
			"zstd level is informational and must fit the u16 header field")
	}
	return uint16(level), nil
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
