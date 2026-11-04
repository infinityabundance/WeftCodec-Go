package modes

import (
	"bytes"
	"io"
	"math"
	"sync"

	"github.com/klauspost/compress/zstd"

	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// maxZstdWindow is the v1 window-log bound (2^27, WEFTCODEC-FORMAT §3).
const maxZstdWindow = 1 << 27

// plainDecoderPool reuses stateless zstd decoders. Each decoder is configured
// with the v1 window bound and single-threaded decoding; Reset(nil) releases
// references before the decoder returns to the pool.
var plainDecoderPool = sync.Pool{
	New: func() any {
		d, err := zstd.NewReader(nil,
			zstd.WithDecoderMaxWindow(maxZstdWindow),
			zstd.WithDecoderConcurrency(1),
		)
		if err != nil {
			return (*zstd.Decoder)(nil)
		}
		return d
	},
}

func getPlainDecoder() *zstd.Decoder {
	v := plainDecoderPool.Get()
	if v == nil {
		return nil
	}
	d, _ := v.(*zstd.Decoder)
	return d
}

func putPlainDecoder(d *zstd.Decoder) {
	if d == nil {
		return
	}
	_ = d.Reset(nil)
	plainDecoderPool.Put(d)
}

func compressZstd(data []byte, level int) ([]byte, error) {
	encoder, err := zstd.NewWriter(nil,
		zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(level)),
		zstd.WithEncoderConcurrency(1),
		// Empty input MUST still produce a valid minimal zstd frame; without
		// this option klauspost returns no bytes at all.
		zstd.WithZeroFrames(true),
	)
	if err != nil {
		return nil, werrors.Malformedf("zstd compressor init failed: %v", err)
	}
	defer encoder.Close()
	return encoder.EncodeAll(data, nil), nil
}

func compressZstdWithDictionary(data, dict []byte, level int) ([]byte, error) {
	encoder, err := zstd.NewWriter(nil,
		zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(level)),
		zstd.WithEncoderConcurrency(1),
		zstd.WithEncoderDictRaw(0, dict),
		// Empty input MUST still produce a valid minimal zstd frame; without
		// this option klauspost returns no bytes at all.
		zstd.WithZeroFrames(true),
	)
	if err != nil {
		return nil, werrors.Malformedf("zstd dictionary init failed: %v", err)
	}
	defer encoder.Close()
	return encoder.EncodeAll(data, nil), nil
}

// checkZstdPreamble validates the zstd frame preamble shared by ZSTD,
// DICTIONARY, and chunked ZSTD payloads before decompression.
//
// An empty payload is structural (Malformed): a zstd frame is never zero
// bytes, even for empty plaintext. If the frame header declares a content
// size, it MUST equal the Weft decoded_len (Corrupt on disagreement), so a
// stream cannot declare a different length than the frame embedding it.
func checkZstdPreamble(body []byte, declaredLen uint64) error {
	if len(body) == 0 {
		return werrors.Malformedf("zstd payload is empty; a zstd frame is never zero bytes")
	}
	var header zstd.Header
	if err := header.Decode(body); err == nil && header.HasFCS &&
		header.FrameContentSize != declaredLen {
		return werrors.Corruptf(
			"zstd frame declares content size %d, Weft header declares %d",
			header.FrameContentSize, declaredLen)
	}
	return nil
}

// decompressZstd decompresses exactly one zstd frame, capping the output at
// declaredLen and failing closed on a length mismatch.
func decompressZstd(body []byte, declaredLen uint64) ([]byte, error) {
	if err := checkZstdPreamble(body, declaredLen); err != nil {
		return nil, err
	}
	if declaredLen > math.MaxInt64-1 {
		return nil, werrors.LimitExceededf(
			"declared zstd output length %d does not fit int64", declaredLen)
	}
	d := getPlainDecoder()
	if d == nil {
		return nil, werrors.NewIo("zstd decompressor init failed")
	}
	defer putPlainDecoder(d)
	if err := d.Reset(bytes.NewReader(body)); err != nil {
		return nil, werrors.Corruptf("zstd decompression failed: %v", err)
	}
	return readExactZstd(d, declaredLen)
}

// decompressZstdWithDictionary decompresses a zstd frame using a raw-content
// dictionary with an exact output bound.
func decompressZstdWithDictionary(body []byte, declaredLen uint64, dict []byte) ([]byte, error) {
	if err := checkZstdPreamble(body, declaredLen); err != nil {
		return nil, err
	}
	if declaredLen > math.MaxInt64-1 {
		return nil, werrors.LimitExceededf(
			"declared zstd output length %d does not fit int64", declaredLen)
	}
	d, err := zstd.NewReader(bytes.NewReader(body),
		zstd.WithDecoderMaxWindow(maxZstdWindow),
		zstd.WithDecoderConcurrency(1),
		zstd.WithDecoderDictRaw(0, dict),
	)
	if err != nil {
		return nil, werrors.Corruptf("zstd dictionary init failed: %v", err)
	}
	defer d.Close()
	return readExactZstd(d, declaredLen)
}

// readExactZstd reads at most declaredLen+1 bytes and requires exactly
// declaredLen bytes of actual output.
func readExactZstd(d *zstd.Decoder, declaredLen uint64) ([]byte, error) {
	out, err := io.ReadAll(io.LimitReader(d, int64(declaredLen)+1))
	if err != nil {
		return nil, werrors.Corruptf("zstd decompression failed: %v", err)
	}
	if uint64(len(out)) != declaredLen {
		return nil, werrors.Corruptf(
			"zstd output length %d differs from declared %d", len(out), declaredLen)
	}
	return out, nil
}
