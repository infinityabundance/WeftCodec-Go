package modes

import (
	"bytes"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func tiledPattern(n int, state uint64) []byte {
	out := make([]byte, n)
	for i := range out {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		out[i] = byte(state)
	}
	return out
}

func tiledRoundtrip(t *testing.T, dtype string, shape, tileShape []uint64, seed uint64) {
	t.Helper()
	lim := limits.Default()
	size, err := tensorElementSize(dtype)
	if err != nil {
		t.Fatalf("element size: %v", err)
	}
	elements := uint64(1)
	for _, dim := range shape {
		elements *= dim
	}
	payload := tiledPattern(int(elements*size), seed)
	encoded, err := EncodeTiledTensor(payload, dtype, shape, tileShape, lim)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	parsed, err := frame.Parse(encoded, lim)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if parsed.Header.Mode != frame.ModeTensorTiled {
		t.Fatalf("mode = %s, want tensor_tiled", parsed.Header.Mode)
	}
	descriptor, err := ParseTiledTensor(parsed, lim)
	if err != nil {
		t.Fatalf("descriptor: %v", err)
	}
	count := uint64(1)
	for _, c := range descriptor.Meta.TileCounts {
		count *= c
	}
	if uint64(len(descriptor.Meta.Tiles)) != count {
		t.Fatalf("tiles = %d, want %d", len(descriptor.Meta.Tiles), count)
	}
	decoded, err := DecodeTiledTensor(parsed, lim)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(decoded, payload) {
		t.Fatal("mode 15 decode is not exact")
	}
	got, touched, err := ReadRegion(descriptor, tensor.FullRegion(shape), lim)
	if err != nil {
		t.Fatalf("region: %v", err)
	}
	if !bytes.Equal(got, payload) || touched != count {
		t.Fatalf("full region = %d bytes/%d tiles, want %d/%d",
			len(got), touched, len(payload), count)
	}
}

func TestTiledRoundtripBf16Square(t *testing.T) {
	tiledRoundtrip(t, "bf16", []uint64{4, 4}, []uint64{2, 2}, 0x51)
}

func TestTiledRoundtripFp32RankThree(t *testing.T) {
	tiledRoundtrip(t, "f32", []uint64{3, 4, 2}, []uint64{1, 2, 2}, 0x77)
}

func TestTiledRoundtripSingleTile(t *testing.T) {
	tiledRoundtrip(t, "i8", []uint64{3, 5}, []uint64{3, 5}, 0x33)
}

func TestTiledRegionTouchesOnlyIntersectingTiles(t *testing.T) {
	lim := limits.Default()
	shape := []uint64{8, 8}
	payload := tiledPattern(64*2, 0x91)
	encoded, err := EncodeTiledTensor(payload, "bf16", shape, []uint64{2, 2}, lim)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	parsed, err := frame.Parse(encoded, lim)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	descriptor, err := ParseTiledTensor(parsed, lim)
	if err != nil {
		t.Fatalf("descriptor: %v", err)
	}

	// One row intersects four tiles.
	region := tensor.NewRegion([]uint64{3, 0}, []uint64{4, 8})
	got, touched, err := ReadRegion(descriptor, region, lim)
	if err != nil {
		t.Fatalf("region: %v", err)
	}
	want := make([]byte, 0, 8*2)
	for col := uint64(0); col < 8; col++ {
		at := (3*8 + col) * 2
		want = append(want, payload[at:at+2]...)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("one-row region bytes differ")
	}
	if touched != 4 {
		t.Fatalf("one-row tiles = %d, want 4", touched)
	}

	// One element intersects exactly one tile.
	region = tensor.NewRegion([]uint64{5, 5}, []uint64{6, 6})
	got, touched, err = ReadRegion(descriptor, region, lim)
	if err != nil {
		t.Fatalf("region: %v", err)
	}
	if !bytes.Equal(got, payload[(5*8+5)*2:(5*8+6)*2]) {
		t.Fatal("single element differs")
	}
	if touched != 1 {
		t.Fatalf("single element tiles = %d, want 1", touched)
	}
}

func TestTiledRejectsNonDividingTileShape(t *testing.T) {
	lim := limits.Default()
	_, err := EncodeTiledTensor(tiledPattern(32, 1), "bf16", []uint64{4, 4}, []uint64{3, 2}, lim)
	if err == nil {
		t.Fatal("expected rejection")
	}
	if got := string(werrors.ClassOf(err)); got != "Malformed" {
		t.Fatalf("class = %s, want Malformed", got)
	}
}

func TestTiledRejectsRankMismatch(t *testing.T) {
	lim := limits.Default()
	_, err := EncodeTiledTensor(tiledPattern(32, 1), "bf16", []uint64{4, 4}, []uint64{2}, lim)
	if err == nil {
		t.Fatal("expected rejection")
	}
}
