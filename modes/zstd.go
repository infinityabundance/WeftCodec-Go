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
	)
	if err != nil {
		return nil, werrors.Malformedf("zstd dictionary init failed: %v", err)
	}
	defer encoder.Close()
	return encoder.EncodeAll(data, nil), nil
}

// decompressZstd decompresses exactly one zstd frame, capping the output at
// declaredLen and failing closed on a length mismatch.
func decompressZstd(body []byte, declaredLen uint64) ([]byte, error) {
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
