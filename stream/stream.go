// Package stream provides chunked encoding and random-range reads.
//
// Chunking is meaningful only for RAW and ZSTD in v1. ReadRange decodes only
// the chunks intersecting the requested range and verifies each such chunk's
// hash. Unchunked frames are decoded fully (which verifies their whole-content
// hash) and then sliced.
package stream

import (
	"context"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// EncodeChunked encodes with independent chunks. mode must be RAW or ZSTD.
// Empty input is returned unchunked (a CHUNKED frame requires a non-empty
// chunk table).
func EncodeChunked(data []byte, mode frame.Mode, chunkSize, level int, lim limits.Limits) ([]byte, error) {
	if chunkSize <= 0 {
		return nil, werrors.NewMalformed("chunk size must be greater than zero")
	}
	switch mode {
	case frame.ModeRaw:
		return modes.EncodeRaw(data, chunkSize, lim)
	case frame.ModeZstd:
		return modes.EncodeZstd(data, level, chunkSize, lim)
	default:
		return nil, werrors.UnsupportedModef("mode %s cannot be chunked in v1", mode)
	}
}

// DecodeChunked decodes a chunked or unchunked frame.
func DecodeChunked(ctx context.Context, frameBytes []byte, resolver dependency.Resolver, lim limits.Limits) ([]byte, error) {
	return modes.DecodeBytes(ctx, frameBytes, resolver, lim)
}

// ReadRange reads length bytes starting at plaintext offset.
//
// The range is clamped to the end of the plaintext. For chunked frames only
// intersecting chunks are decoded and hash-verified; the whole-content hash
// cannot be checked without decoding every chunk and is therefore not checked
// here.
func ReadRange(ctx context.Context, frameBytes []byte, resolver dependency.Resolver, lim limits.Limits, offset, length uint64) ([]byte, error) {
	f, err := frame.Parse(frameBytes, lim)
	if err != nil {
		return nil, err
	}
	decodedLen := f.Header.DecodedLen
	if offset >= decodedLen {
		return []byte{}, nil
	}
	available := decodedLen - offset
	wanted := available
	if length < wanted {
		wanted = length
	}
	if wanted == 0 {
		return []byte{}, nil
	}
	end, err := limits.AddChecked(offset, wanted, "range end")
	if err != nil {
		return nil, werrors.NewMalformed("range end overflow")
	}
	startInt, err := limits.ToInt(offset, "range offset")
	if err != nil {
		return nil, err
	}
	endInt, err := limits.ToInt(end, "range end")
	if err != nil {
		return nil, err
	}

	if !f.Header.IsChunked() {
		plaintext, err := modes.DecodeBytes(ctx, frameBytes, resolver, lim)
		if err != nil {
			return nil, err
		}
		if endInt > len(plaintext) || startInt > endInt {
			return nil, werrors.NewCorrupt("decoded frame is shorter than its header claims")
		}
		out := make([]byte, endInt-startInt)
		copy(out, plaintext[startInt:endInt])
		return out, nil
	}

	capacity, err := limits.ToInt(wanted, "range length")
	if err != nil {
		return nil, err
	}
	if capacity > 1<<20 {
		capacity = 1 << 20
	}
	out := make([]byte, 0, capacity)
	var chunkStart uint64
	for i := range f.Chunks {
		chunkLen := uint64(f.Chunks[i].DecodedLen)
		chunkEnd, err := limits.AddChecked(chunkStart, chunkLen, "chunk decoded extent")
		if err != nil {
			return nil, werrors.NewMalformed(err.Error())
		}
		if chunkEnd <= offset {
			chunkStart = chunkEnd
			continue
		}
		if chunkStart >= end {
			break
		}
		decoded, err := modes.DecodeChunk(f, i)
		if err != nil {
			return nil, err
		}
		from := maxU64(offset, chunkStart) - chunkStart
		to := minU64(end, chunkEnd) - chunkStart
		fromInt, err := limits.ToInt(from, "range slice")
		if err != nil {
			return nil, err
		}
		toInt, err := limits.ToInt(to, "range slice")
		if err != nil {
			return nil, err
		}
		if toInt > len(decoded) || fromInt > toInt {
			return nil, werrors.NewCorrupt("chunk decoded extent is inconsistent")
		}
		out = append(out, decoded[fromInt:toInt]...)
		chunkStart = chunkEnd
	}
	if uint64(len(out)) != wanted {
		return nil, werrors.NewCorrupt("random-range read produced an unexpected length")
	}
	return out, nil
}

func maxU64(a, b uint64) uint64 {
	if a > b {
		return a
	}
	return b
}

func minU64(a, b uint64) uint64 {
	if a < b {
		return a
	}
	return b
}
