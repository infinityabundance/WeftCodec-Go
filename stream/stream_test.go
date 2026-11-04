package stream

import (
	"bytes"
	"context"
	"encoding/binary"
	"hash/crc32"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

var testCastagnoli = crc32.MakeTable(crc32.Castagnoli)

func refixTrailer(f []byte) {
	at := len(f) - 4
	crc := crc32.Checksum(f[:at], testCastagnoli)
	binary.LittleEndian.PutUint32(f[at:], crc)
}

func pattern(n int, state uint64) []byte {
	out := make([]byte, n)
	for i := 0; i < n; i++ {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		out[i] = byte(state)
	}
	return out
}

func TestChunkedRoundTrip(t *testing.T) {
	lim := limits.Default()
	data := pattern(5000, 99)
	encoded, err := EncodeChunked(data, frame.ModeRaw, 1024, 3, lim)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeChunked(context.Background(), encoded, nil, lim)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(decoded, data) {
		t.Fatal("chunked round-trip mismatch")
	}
}

func TestReadRangeAcrossChunks(t *testing.T) {
	lim := limits.Default()
	data := pattern(5000, 7)
	encoded, err := EncodeChunked(data, frame.ModeRaw, 1024, 3, lim)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	got, err := ReadRange(ctx, encoded, nil, lim, 1000, 200)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, data[1000:1200]) {
		t.Fatal("range across chunks mismatch")
	}
	got, err = ReadRange(ctx, encoded, nil, lim, 4900, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, data[4900:]) {
		t.Fatal("clamped range mismatch")
	}
	got, err = ReadRange(ctx, encoded, nil, lim, 6000, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatal("range past EOF should be empty")
	}
}

// TestRangeDecodesOnlyIntersectingChunks corrupts the last chunk's encoded
// bytes while keeping the frame trailer valid. A full decode fails with
// Corrupt, but a range that does not intersect the last chunk succeeds, which
// proves non-intersecting chunks are never decoded.
func TestRangeDecodesOnlyIntersectingChunks(t *testing.T) {
	lim := limits.Default()
	data := pattern(5000, 11)
	encoded, err := EncodeChunked(data, frame.ModeRaw, 1024, 3, lim)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := frame.Parse(encoded, lim)
	if err != nil {
		t.Fatal(err)
	}
	bodyStart := 96 + int(parsed.Header.TableLen)
	corrupt := append([]byte(nil), encoded...)
	corrupt[bodyStart+len(data)-1] ^= 0xff
	refixTrailer(corrupt)

	ctx := context.Background()
	if _, err := DecodeChunked(ctx, corrupt, nil, lim); werrors.ClassOf(err) != werrors.Corrupt {
		t.Fatalf("full decode: got %v, want Corrupt", err)
	}
	got, err := ReadRange(ctx, corrupt, nil, lim, 0, 10)
	if err != nil {
		t.Fatalf("range before corrupt chunk: %v", err)
	}
	if !bytes.Equal(got, data[:10]) {
		t.Fatal("range before corrupt chunk mismatch")
	}
	if _, err := ReadRange(ctx, corrupt, nil, lim, 4900, 100); werrors.ClassOf(err) != werrors.Corrupt {
		t.Fatalf("range intersecting corrupt chunk: got %v, want Corrupt", err)
	}
}

func TestReadRangeUnchunked(t *testing.T) {
	lim := limits.Default()
	data := pattern(2048, 5)
	encoded, err := modes.EncodeRaw(data, 0, lim)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ReadRange(context.Background(), encoded, nil, lim, 100, 50)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, data[100:150]) {
		t.Fatal("unchunked range mismatch")
	}
}
