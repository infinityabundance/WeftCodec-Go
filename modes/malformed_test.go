package modes

import (
	"context"
	"encoding/binary"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func parseClass(t *testing.T, b []byte, lim limits.Limits) werrors.Class {
	t.Helper()
	_, err := frame.Parse(b, lim)
	if err == nil {
		t.Fatal("expected parse failure")
	}
	return werrors.ClassOf(err)
}

func decodeClass(t *testing.T, b []byte, resolver dependency.Resolver, lim limits.Limits) werrors.Class {
	t.Helper()
	_, err := DecodeBytes(context.Background(), b, resolver, lim)
	if err == nil {
		t.Fatal("expected decode failure")
	}
	return werrors.ClassOf(err)
}

func TestStructuralHeaderErrors(t *testing.T) {
	lim := limits.Default()
	good, err := EncodeRaw(pattern(4096, 1), 0, lim)
	if err != nil {
		t.Fatal(err)
	}

	mutate := func(edit func([]byte)) []byte {
		b := append([]byte(nil), good...)
		edit(b)
		refixHeaderAndTrailer(b)
		return b
	}

	cases := []struct {
		name  string
		frame []byte
		want  werrors.Class
	}{
		{"bad_magic", func() []byte { b := append([]byte(nil), good...); b[0] ^= 0xff; return b }(), werrors.Malformed},
		{"bad_header_crc", func() []byte { b := append([]byte(nil), good...); b[72] ^= 0xff; return b }(), werrors.Corrupt},
		{"bad_trailer_crc", func() []byte { b := append([]byte(nil), good...); b[len(b)-1] ^= 0xff; return b }(), werrors.Corrupt},
		{"truncated", func() []byte { b := append([]byte(nil), good...); return b[:len(b)-1] }(), werrors.Malformed},
		{"major", mutate(func(b []byte) { b[4] = 2 }), werrors.UnsupportedVersion},
		{"minor", mutate(func(b []byte) { b[5] = 1 }), werrors.UnsupportedVersion},
		{"mode", mutate(func(b []byte) { b[7] = 16 }), werrors.UnsupportedMode},
		{"feature", mutate(func(b []byte) {
			f := binary.LittleEndian.Uint16(b[36:38]) | 0x0004
			binary.LittleEndian.PutUint16(b[36:38], f)
		}), werrors.UnsupportedFeature},
		{"original_hash_flag", mutate(func(b []byte) {
			f := binary.LittleEndian.Uint16(b[8:10]) | 0x0008
			binary.LittleEndian.PutUint16(b[8:10], f)
		}), werrors.UnsupportedFeature},
		{"self_contained", mutate(func(b []byte) {
			f := binary.LittleEndian.Uint16(b[8:10]) | 0x0002
			binary.LittleEndian.PutUint16(b[8:10], f)
		}), werrors.UnsupportedFeature},
		{"reserved0", mutate(func(b []byte) { b[38] = 1 }), werrors.Malformed},
		{"table_len", mutate(func(b []byte) { b[76] = 1 }), werrors.Malformed},
		{"body_len", mutate(func(b []byte) {
			v := binary.LittleEndian.Uint64(b[20:28]) + 1
			binary.LittleEndian.PutUint64(b[20:28], v)
		}), werrors.Malformed},
		{"domain", mutate(func(b []byte) { b[6] = 2 }), werrors.Malformed},
		{"incomplete", append([]byte("WFT1"), make([]byte, 95)...), werrors.Malformed},
	}
	for _, c := range cases {
		if got := parseClass(t, c.frame, lim); got != c.want {
			t.Fatalf("%s: got %s, want %s", c.name, got, c.want)
		}
	}
}

func TestChunkTableErrors(t *testing.T) {
	lim := limits.Default()
	good, err := EncodeRaw(pattern(3000, 7), 1024, lim)
	if err != nil {
		t.Fatal(err)
	}
	secondOff := 96 + 48
	mutate := func(edit func([]byte)) []byte {
		b := append([]byte(nil), good...)
		edit(b)
		refixTrailer(b)
		return b
	}

	if got := parseClass(t, mutate(func(b []byte) {
		v := binary.LittleEndian.Uint64(b[secondOff:secondOff+8]) + 1
		binary.LittleEndian.PutUint64(b[secondOff:secondOff+8], v)
	}), lim); got != werrors.Malformed {
		t.Fatalf("chunk gap: got %s", got)
	}
	if got := parseClass(t, mutate(func(b []byte) {
		binary.LittleEndian.PutUint64(b[secondOff:secondOff+8], 96)
	}), lim); got != werrors.Malformed {
		t.Fatalf("chunk overlap: got %s", got)
	}
	if got := parseClass(t, mutate(func(b []byte) {
		binary.LittleEndian.PutUint32(b[96+8:96+12], 0)
	}), lim); got != werrors.Malformed {
		t.Fatalf("zero encoded length: got %s", got)
	}
	if got := parseClass(t, mutate(func(b []byte) {
		v := binary.LittleEndian.Uint32(b[96+12:96+16]) + 1
		binary.LittleEndian.PutUint32(b[96+12:96+16], v)
	}), lim); got != werrors.Malformed {
		t.Fatalf("decoded sum: got %s", got)
	}

	parsed, err := frame.Parse(good, lim)
	if err != nil {
		t.Fatal(err)
	}
	bodyStart := 96 + int(parsed.Header.TableLen)
	corrupt := mutate(func(b []byte) { b[bodyStart] ^= 0xff })
	if got := decodeClass(t, corrupt, resolverWith(), lim); got != werrors.Corrupt {
		t.Fatalf("chunk hash: got %s", got)
	}
}

func TestDependencyTableErrors(t *testing.T) {
	lim := limits.Default()
	base := pattern(1000, 3)
	target := pattern(1200, 4)
	encoded, err := EncodeDelta(target, base, lim)
	if err != nil {
		t.Fatal(err)
	}
	depAt := 96
	mutate := func(edit func([]byte)) []byte {
		b := append([]byte(nil), encoded...)
		edit(b)
		refixTrailer(b)
		return b
	}
	cases := []struct {
		name  string
		edit  func([]byte)
		class werrors.Class
	}{
		{"domain", func(b []byte) { b[depAt+40] = 3 }, werrors.Malformed},
		{"no_required", func(b []byte) { b[depAt+41] = 0 }, werrors.Malformed},
		{"unknown_flag", func(b []byte) { b[depAt+41] = 0x03 }, werrors.Malformed},
		{"reserved", func(b []byte) { b[depAt+42] = 1 }, werrors.Malformed},
	}
	for _, c := range cases {
		if got := parseClass(t, mutate(c.edit), lim); got != c.class {
			t.Fatalf("%s: got %s, want %s", c.name, got, c.class)
		}
	}

	if got := decodeClass(t, encoded, dependency.NewMapResolver(), lim); got != werrors.DependencyMissing {
		t.Fatalf("missing dependency: got %s", got)
	}

	badLen := mutate(func(b []byte) {
		v := binary.LittleEndian.Uint64(b[depAt+32:depAt+40]) - 1
		binary.LittleEndian.PutUint64(b[depAt+32:depAt+40], v)
	})
	if got := decodeClass(t, badLen, resolverWith(base), lim); got != werrors.DependencyMismatch {
		t.Fatalf("bad dependency length: got %s", got)
	}

	badHash := mutate(func(b []byte) { b[depAt] ^= 0xff })
	wrongResolver := dependency.NewMapResolver()
	wrongHash, err := hash.FromSlice(badHash[depAt : depAt+32])
	if err != nil {
		t.Fatal(err)
	}
	wrongResolver.Insert(wrongHash, base)
	if got := decodeClass(t, badHash, wrongResolver, lim); got != werrors.DependencyMismatch {
		t.Fatalf("bad dependency hash: got %s", got)
	}
}

func TestDeltaOpErrors(t *testing.T) {
	lim := limits.Default()
	base := pattern(1000, 11)
	dep := frame.NewDependency(depHash(base), uint64(len(base)), frame.DepDomainPayload, 0)
	build := func(body []byte, decodedLen uint64) []byte {
		parts := frame.PayloadParts(
			frame.ModeDelta, 0, 0, decodedLen,
			hash.Domain(hash.DomainChunk, make([]byte, decodedLen)), body)
		parts.Dependencies = []frame.Dependency{dep}
		encoded, err := frame.Encode(parts, lim)
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		return encoded
	}
	resolver := resolverWith(base)
	cases := []struct {
		name   string
		body   []byte
		decLen uint64
		class  werrors.Class
	}{
		{"copy_offset_beyond", []byte{0x00, 0xE8, 0x07, 0x01}, 1, werrors.Malformed},
		{"copy_zero_length", []byte{0x00, 0x00, 0x00}, 1, werrors.Malformed},
		{"insert_zero_length", []byte{0x01, 0x00}, 1, werrors.Malformed},
		{"insert_past_body", []byte{0x01, 0x05, 'a'}, 5, werrors.Malformed},
		{"unknown_opcode", []byte{0x02}, 0, werrors.Malformed},
		{"output_short", []byte{0x00, 0x00, 0x01}, 2, werrors.Corrupt},
		{"output_long", []byte{0x00, 0x00, 0x05}, 2, werrors.Corrupt},
		{"non_minimal_varint", []byte{0x00, 0x80, 0x00, 0x01}, 1, werrors.NonCanonical},
	}
	for _, c := range cases {
		if got := decodeClass(t, build(c.body, c.decLen), resolver, lim); got != c.class {
			t.Fatalf("%s: got %s, want %s", c.name, got, c.class)
		}
	}
}

func TestLimitEnforcement(t *testing.T) {
	lim := limits.Default()
	encoded, err := EncodeRaw(pattern(4096, 1), 0, lim)
	if err != nil {
		t.Fatal(err)
	}
	tight := lim
	tight.MaxDecodedBytes = 100
	if got := parseClass(t, encoded, tight); got != werrors.LimitExceeded {
		t.Fatalf("max decoded: got %s", got)
	}
	tight = lim
	tight.MaxFrameBytes = 50
	if got := parseClass(t, encoded, tight); got != werrors.LimitExceeded {
		t.Fatalf("max frame: got %s", got)
	}
	chunked, err := EncodeRaw(pattern(5000, 1), 1024, lim)
	if err != nil {
		t.Fatal(err)
	}
	tight = lim
	tight.MaxChunks = 2
	if got := parseClass(t, chunked, tight); got != werrors.LimitExceeded {
		t.Fatalf("max chunks: got %s", got)
	}
	tight = lim
	tight.MaxDependencyCount = 0
	base := pattern(100, 2)
	delta, err := EncodeDelta(pattern(120, 3), base, lim)
	if err != nil {
		t.Fatal(err)
	}
	if got := parseClass(t, delta, tight); got != werrors.LimitExceeded {
		t.Fatalf("max dependencies: got %s", got)
	}
}
