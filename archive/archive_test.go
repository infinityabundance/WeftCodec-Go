package archive

import (
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func buildArchive(frames [][]byte) []byte {
	out := make([]byte, 0)
	out = append(out, Magic...)
	var u32 [4]byte
	binary.LittleEndian.PutUint32(u32[:], Version)
	out = append(out, u32[:]...)
	binary.LittleEndian.PutUint32(u32[:], uint32(len(frames)))
	out = append(out, u32[:]...)
	for _, f := range frames {
		var u64 [8]byte
		binary.LittleEndian.PutUint64(u64[:], uint64(len(f)))
		out = append(out, u64[:]...)
		out = append(out, f...)
	}
	var crc [4]byte
	binary.LittleEndian.PutUint32(crc[:], crc32c(out))
	out = append(out, crc[:]...)
	return out
}

func rawFrameWithHash(t *testing.T, body []byte, contentHash hash.Hash32, lim limits.Limits) []byte {
	t.Helper()
	parts := frame.PayloadParts(
		frame.ModeRaw, 0, 0, uint64(len(body)), contentHash, body)
	encoded, err := frame.Encode(parts, lim)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

func TestArchiveRoundTrip(t *testing.T) {
	lim := limits.Default()
	base := []byte("base payload for archive round trip")
	target := []byte("base payload for archive round trip plus more")
	baseFrame, err := modes.EncodeRaw(base, 0, lim)
	if err != nil {
		t.Fatal(err)
	}
	deltaFrame, err := modes.EncodeDelta(target, base, lim)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := Create([][]byte{deltaFrame, baseFrame}, lim)
	if err != nil {
		t.Fatal(err)
	}
	entries, err := Decode(encoded, lim)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatalf("got %d entries, want 2", len(entries))
	}
	if !bytes.Equal(entries[0].Bytes, base) || !bytes.Equal(entries[1].Bytes, target) {
		t.Fatal("archive plaintext mismatch")
	}
	if entries[1].ContentHash != hash.Domain(hash.DomainChunk, target) {
		t.Fatal("archive content hash mismatch")
	}
}

func TestArchiveMissingDependency(t *testing.T) {
	lim := limits.Default()
	base := []byte("base")
	target := []byte("base-plus")
	deltaFrame, err := modes.EncodeDelta(target, base, lim)
	if err != nil {
		t.Fatal(err)
	}
	_, err = Create([][]byte{deltaFrame}, lim)
	if got := werrors.ClassOf(err); got != werrors.DependencyMissing {
		t.Fatalf("got %s (%v), want DependencyMissing", got, err)
	}
}

func TestArchiveCycle(t *testing.T) {
	lim := limits.Default()
	contentA := hash.FromBytes([32]byte{1})
	contentB := hash.FromBytes([32]byte{2})
	makeRef := func(contentHash, dep hash.Hash32) []byte {
		parts := frame.PayloadParts(frame.ModeReference, 0, 0, 1, contentHash, nil)
		parts.Dependencies = []frame.Dependency{
			frame.NewDependency(dep, 1, frame.DepDomainPayload, 0),
		}
		encoded, err := frame.Encode(parts, lim)
		if err != nil {
			t.Fatal(err)
		}
		return encoded
	}
	_, err := Create([][]byte{makeRef(contentA, contentB), makeRef(contentB, contentA)}, lim)
	if got := werrors.ClassOf(err); got != werrors.DependencyCycle {
		t.Fatalf("got %s (%v), want DependencyCycle", got, err)
	}
}

func TestArchiveRejections(t *testing.T) {
	lim := limits.Default()
	base := []byte("base")
	target := []byte("base-plus")
	baseFrame, err := modes.EncodeRaw(base, 0, lim)
	if err != nil {
		t.Fatal(err)
	}
	deltaFrame, err := modes.EncodeDelta(target, base, lim)
	if err != nil {
		t.Fatal(err)
	}
	good, err := Create([][]byte{baseFrame, deltaFrame}, lim)
	if err != nil {
		t.Fatal(err)
	}

	badMagic := append([]byte(nil), good...)
	badMagic[0] ^= 0xff
	_, err = Parse(badMagic, lim)
	if got := werrors.ClassOf(err); got != werrors.Malformed {
		t.Fatalf("bad magic: got %s (%v)", got, err)
	}

	badCRC := append([]byte(nil), good...)
	badCRC[len(badCRC)-1] ^= 0xff
	_, err = Parse(badCRC, lim)
	if got := werrors.ClassOf(err); got != werrors.Corrupt {
		t.Fatalf("bad crc: got %s (%v)", got, err)
	}

	// Non-topological order: delta before its base, with a valid archive CRC.
	nonTopo := buildArchive([][]byte{deltaFrame, baseFrame})
	_, err = Parse(nonTopo, lim)
	if got := werrors.ClassOf(err); got != werrors.Malformed {
		t.Fatalf("non-topological: got %s (%v)", got, err)
	}

	// A frame_len that extends past the archive body, with a valid CRC.
	badLen := append([]byte(nil), good...)
	binary.LittleEndian.PutUint64(badLen[12:20], uint64(len(good)))
	binary.LittleEndian.PutUint32(badLen[len(badLen)-4:], crc32c(badLen[:len(badLen)-4]))
	_, err = Parse(badLen, lim)
	if got := werrors.ClassOf(err); got != werrors.Malformed {
		t.Fatalf("bad frame length: got %s (%v)", got, err)
	}

	// Entry count larger than the remaining bytes allow.
	badCount := append([]byte(nil), good...)
	binary.LittleEndian.PutUint32(badCount[8:12], 1_000_000)
	binary.LittleEndian.PutUint32(badCount[len(badCount)-4:], crc32c(badCount[:len(badCount)-4]))
	_, err = Parse(badCount, lim)
	if got := werrors.ClassOf(err); got != werrors.Malformed {
		t.Fatalf("bad entry count: got %s (%v)", got, err)
	}

	// Duplicate content hash with different frame bytes.
	sharedHash := hash.Domain(hash.DomainChunk, []byte("declared"))
	first := rawFrameWithHash(t, []byte("aaaa"), sharedHash, lim)
	second := rawFrameWithHash(t, []byte("bbbb"), sharedHash, lim)
	_, err = Create([][]byte{first, second}, lim)
	if got := werrors.ClassOf(err); got != werrors.Malformed {
		t.Fatalf("create duplicate: got %s (%v)", got, err)
	}
	_, err = Parse(buildArchive([][]byte{first, second}), lim)
	if got := werrors.ClassOf(err); got != werrors.Corrupt {
		t.Fatalf("parse duplicate: got %s (%v)", got, err)
	}

	// Limit on entry count.
	tight := lim
	tight.MaxArchiveEntries = 1
	_, err = Parse(good, tight)
	if got := werrors.ClassOf(err); got != werrors.LimitExceeded {
		t.Fatalf("entry limit: got %s (%v)", got, err)
	}
}

func TestArchiveCumulativeDecodedLimit(t *testing.T) {
	lim := limits.Default()
	base := []byte("0123456789")
	target := []byte("0123456789abcdef")
	baseFrame, err := modes.EncodeRaw(base, 0, lim)
	if err != nil {
		t.Fatal(err)
	}
	deltaFrame, err := modes.EncodeDelta(target, base, lim)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := Create([][]byte{baseFrame, deltaFrame}, lim)
	if err != nil {
		t.Fatal(err)
	}
	tight := lim
	tight.MaxDecodedBytes = uint64(len(target)) // each frame fits, the sum does not
	_, err = Decode(encoded, tight)
	if got := werrors.ClassOf(err); got != werrors.LimitExceeded {
		t.Fatalf("cumulative limit: got %s (%v)", got, err)
	}
}
