package varint

import (
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func TestRoundTripBoundaries(t *testing.T) {
	values := []uint64{
		0, 1, 127, 128, 129, 255, 256, 16384,
		1<<32 - 1, 1<<63 - 1, ^uint64(0) - 1, ^uint64(0),
	}
	for _, v := range values {
		got := Append(nil, v)
		if len(got) != EncodedLen(v) {
			t.Fatalf("%d: encoded length %d, want %d", v, len(got), EncodedLen(v))
		}
		decoded, next, err := DecodeAt(got, 0)
		if err != nil {
			t.Fatalf("%d: %v", v, err)
		}
		if decoded != v || next != len(got) {
			t.Fatalf("%d: got %d next %d", v, decoded, next)
		}
		buf := make([]byte, MaxLen)
		n := EncodeTo(buf, v)
		if n != len(got) {
			t.Fatalf("%d: EncodeTo wrote %d, want %d", v, n, len(got))
		}
	}
}

func TestRejectsNonMinimal(t *testing.T) {
	for _, input := range [][]byte{{0x80, 0x00}, {0x81, 0x00}, {0xFF, 0x00}} {
		_, _, err := DecodeAt(input, 0)
		if err == nil || werrors.ClassOf(err) != werrors.NonCanonical {
			t.Fatalf("%x: got %v, want NonCanonical", input, err)
		}
	}
}

func TestRejectsOverlongAndTruncated(t *testing.T) {
	cases := [][]byte{
		{0x80},
		{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01},
		{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x02},
	}
	for _, input := range cases {
		_, _, err := DecodeAt(input, 0)
		if err == nil || werrors.ClassOf(err) != werrors.Malformed {
			t.Fatalf("%x: got %v, want Malformed", input, err)
		}
	}
	max := []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01}
	v, _, err := DecodeAt(max, 0)
	if err != nil || v != ^uint64(0) {
		t.Fatalf("u64 max: got %d, %v", v, err)
	}
}

func TestZigzag(t *testing.T) {
	values := []int64{0, 1, -1, 2, -2, 1<<63 - 1, -1 << 63, -123456789}
	for _, v := range values {
		if got := ZigzagDecode(ZigzagEncode(v)); got != v {
			t.Fatalf("zigzag %d -> %d", v, got)
		}
	}
	if ZigzagEncode(-1) != 1 || ZigzagEncode(1) != 2 || ZigzagEncode(0) != 0 {
		t.Fatal("zigzag reference values")
	}
}
