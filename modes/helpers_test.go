package modes

import (
	"encoding/binary"
	"hash/crc32"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

var testCastagnoli = crc32.MakeTable(crc32.Castagnoli)

func testCRC32(parts ...[]byte) uint32 {
	h := crc32.New(testCastagnoli)
	for _, p := range parts {
		_, _ = h.Write(p)
	}
	return h.Sum32()
}

func refixTrailer(f []byte) {
	at := len(f) - 4
	binary.LittleEndian.PutUint32(f[at:], testCRC32(f[:at]))
}

func refixHeaderAndTrailer(f []byte) {
	crc := testCRC32(f[0:72], f[76:96])
	binary.LittleEndian.PutUint32(f[72:76], crc)
	refixTrailer(f)
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

func repeat(text []byte, n int) []byte {
	out := make([]byte, 0, n)
	for len(out) < n {
		out = append(out, text...)
	}
	return out[:n]
}

func depHash(b []byte) hash.Hash32 { return hash.Domain(hash.DomainChunk, b) }

func resolverWith(b ...[]byte) *dependency.MapResolver {
	r := dependency.NewMapResolver()
	for _, payload := range b {
		r.Insert(depHash(payload), payload)
	}
	return r
}

func requireClass(t *testing.T, err error, want werrors.Class) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected %s, got nil", want)
	}
	if got := werrors.ClassOf(err); got != want {
		t.Fatalf("expected class %s, got %s (%v)", want, got, err)
	}
}
