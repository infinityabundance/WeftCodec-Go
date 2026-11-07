package modes

import (
	"bytes"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func ftdPattern(length int, seed uint64) []byte {
	state := seed | 1
	out := make([]byte, length)
	for index := range out {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		out[index] = byte(state & 0xff)
	}
	return out
}

func TestFloatTiledRoundTripAndRegion(t *testing.T) {
	lim := limits.Default()
	dtype, err := ParseFloatDtype("bf16")
	if err != nil {
		t.Fatal(err)
	}
	bytesIn := ftdPattern(8*8*2, 0x51)
	encoded, err := EncodeFloatTiled(bytesIn, dtype, []uint64{8, 8}, []uint64{2, 4}, lim)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	parsed, err := frame.Parse(encoded, lim)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if parsed.Header.Mode != frame.ModeFloatTiled {
		t.Fatalf("mode %s", parsed.Header.Mode)
	}
	descriptor, err := ParseFloatTiled(parsed, lim)
	if err != nil {
		t.Fatalf("descriptor: %v", err)
	}
	if len(descriptor.Meta.Tiles) != 8 {
		t.Fatalf("tiles %d", len(descriptor.Meta.Tiles))
	}
	decoded, err := DecodeFloatTiled(parsed, lim)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(decoded, bytesIn) {
		t.Fatalf("decoded bytes differ")
	}

	whole := tensor.Region{Start: []uint64{0, 0}, End: []uint64{8, 8}}
	all, touched, err := ReadFloatRegion(descriptor, whole, lim)
	if err != nil {
		t.Fatalf("whole region: %v", err)
	}
	if touched != 8 || !bytes.Equal(all, bytesIn) {
		t.Fatalf("whole region: touched=%d equal=%v", touched, bytes.Equal(all, bytesIn))
	}

	row := tensor.Region{Start: []uint64{1, 0}, End: []uint64{2, 8}}
	rowBytes, touched, err := ReadFloatRegion(descriptor, row, lim)
	if err != nil {
		t.Fatalf("row region: %v", err)
	}
	if touched != 2 || !bytes.Equal(rowBytes, bytesIn[16:32]) {
		t.Fatalf("row region: touched=%d bytes=%x", touched, rowBytes)
	}

	one := tensor.Region{Start: []uint64{5, 7}, End: []uint64{6, 8}}
	element, touched, err := ReadFloatRegion(descriptor, one, lim)
	if err != nil {
		t.Fatalf("element region: %v", err)
	}
	if touched != 1 || !bytes.Equal(element, bytesIn[94:96]) {
		t.Fatalf("element region: touched=%d bytes=%x", touched, element)
	}
}

func TestFloatTiledIsDeterministic(t *testing.T) {
	lim := limits.Default()
	dtype, _ := ParseFloatDtype("fp16")
	payload := ftdPattern(16*2, 0x33)
	first, err := EncodeFloatTiled(payload, dtype, []uint64{4, 4}, []uint64{1, 2}, lim)
	if err != nil {
		t.Fatal(err)
	}
	second, err := EncodeFloatTiled(payload, dtype, []uint64{4, 4}, []uint64{1, 2}, lim)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("encoding is not deterministic")
	}
}

func TestFloatTiledRejectsNonDividingShape(t *testing.T) {
	lim := limits.Default()
	dtype, _ := ParseFloatDtype("bf16")
	payload := ftdPattern(8*8*2, 0x11)
	_, err := EncodeFloatTiled(payload, dtype, []uint64{8, 8}, []uint64{3, 4}, lim)
	if err == nil || werrors.ClassOf(err) != werrors.Malformed {
		t.Fatalf("want Malformed, got %v", err)
	}
}

func TestFloatTiledRejectsFp8(t *testing.T) {
	lim := limits.Default()
	dtype, _ := ParseFloatDtype("fp8-e4m3")
	payload := ftdPattern(4, 0x22)
	_, err := EncodeFloatTiled(payload, dtype, []uint64{4}, []uint64{2}, lim)
	if err == nil || werrors.ClassOf(err) != werrors.UnsupportedFeature {
		t.Fatalf("want UnsupportedFeature, got %v", err)
	}
}
