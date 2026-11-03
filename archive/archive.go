// Package archive implements the self-contained WFA1 transport archive
// (WEFTCODEC-FORMAT §7).
//
//	"WFA1" || u32le version=1 || u32le entry_count ||
//	  entry_count * ( u64le frame_len || frame_bytes ) ||
//	  u32le archive_crc32c
//
// Entries are topologically ordered; every dependency is an earlier entry.
package archive

import (
	"container/heap"
	"context"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"sort"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

const (
	// Magic is the archive magic.
	Magic = "WFA1"
	// Version is the archive version.
	Version uint32 = 1
	// HeaderLen is the fixed archive header length.
	HeaderLen = 12
	// TrailerLen is the archive CRC length.
	TrailerLen = 4
)

var castagnoli = crc32.MakeTable(crc32.Castagnoli)

func crc32c(b []byte) uint32 { return crc32.Checksum(b, castagnoli) }

// Entry is one parsed frame inside an archive.
type Entry struct {
	FrameBytes []byte
	Frame      *frame.Frame
}

// Archive is a parsed and structurally validated archive.
type Archive struct {
	Entries []Entry
}

// DecodedEntry is a successfully decoded archive entry.
type DecodedEntry struct {
	ContentHash hash.Hash32
	Bytes       []byte
}

type frameInfo struct {
	bytes []byte
	deps  []hash.Hash32
}

// Create builds a self-contained archive from complete frame bytes.
//
// Frames are deduplicated when byte-identical. Every dependency hash must
// correspond to another supplied frame. Output order is deterministic
// (Kahn's algorithm, smallest hash first).
func Create(frames [][]byte, lim limits.Limits) ([]byte, error) {
	if err := limits.CheckLen(uint64(len(frames)), lim.MaxArchiveEntries, "archive entry count"); err != nil {
		return nil, err
	}
	if uint64(len(frames)) > uint64(^uint32(0)) {
		return nil, werrors.NewLimitExceeded("archive entry count does not fit u32")
	}

	byHash := make(map[hash.Hash32]*frameInfo)
	for _, b := range frames {
		f, err := frame.Parse(b, lim)
		if err != nil {
			return nil, err
		}
		h := f.Header.ContentHash
		deps := make([]hash.Hash32, 0, len(f.Dependencies))
		for _, dep := range f.Dependencies {
			deps = append(deps, dep.Hash)
		}
		if existing, ok := byHash[h]; ok {
			if !bytesEqual(existing.bytes, b) {
				return nil, werrors.Malformedf(
					"duplicate content hash %s with different frame bytes", h)
			}
			continue
		}
		cp := make([]byte, len(b))
		copy(cp, b)
		byHash[h] = &frameInfo{bytes: cp, deps: deps}
	}

	indegree := make(map[hash.Hash32]int, len(byHash))
	dependents := make(map[hash.Hash32][]hash.Hash32)
	for h := range byHash {
		indegree[h] = 0
	}
	for h, info := range byHash {
		for _, dep := range info.deps {
			if _, ok := byHash[dep]; !ok {
				return nil, werrors.DependencyMissingf(
					"frame %s depends on %s, which is not in the archive", h, dep)
			}
			indegree[h]++
			dependents[dep] = append(dependents[dep], h)
		}
	}
	for dep := range dependents {
		list := dependents[dep]
		sort.Slice(list, func(i, j int) bool { return list[i].Less(list[j]) })
		dependents[dep] = list
	}

	ready := &hashHeap{}
	heap.Init(ready)
	for h, degree := range indegree {
		if degree == 0 {
			heap.Push(ready, h)
		}
	}
	order := make([]hash.Hash32, 0, len(byHash))
	for ready.Len() > 0 {
		h := heap.Pop(ready).(hash.Hash32)
		order = append(order, h)
		for _, child := range dependents[h] {
			degree, ok := indegree[child]
			if !ok {
				return nil, werrors.NewMalformed("archive graph inconsistency")
			}
			if degree == 0 {
				return nil, werrors.NewMalformed("archive indegree underflow")
			}
			degree--
			indegree[child] = degree
			if degree == 0 {
				heap.Push(ready, child)
			}
		}
	}
	if len(order) != len(byHash) {
		return nil, werrors.NewDependencyCycle("archive dependency graph contains a cycle")
	}

	total := uint64(HeaderLen)
	for _, h := range order {
		info := byHash[h]
		var err error
		total, err = limits.AddChecked(total, 8, "archive size")
		if err != nil {
			return nil, err
		}
		total, err = limits.AddChecked(total, uint64(len(info.bytes)), "archive size")
		if err != nil {
			return nil, err
		}
	}
	if err := limits.CheckLen(total, lim.MaxTotalFetchedBytes, "archive size"); err != nil {
		return nil, err
	}
	totalWithTrailer, err := limits.AddChecked(total, TrailerLen, "archive size")
	if err != nil {
		return nil, err
	}
	totalInt, err := limits.ToInt(totalWithTrailer, "archive size")
	if err != nil {
		return nil, err
	}

	out := make([]byte, 0, totalInt)
	out = append(out, Magic...)
	var u32 [4]byte
	binary.LittleEndian.PutUint32(u32[:], Version)
	out = append(out, u32[:]...)
	binary.LittleEndian.PutUint32(u32[:], uint32(len(order)))
	out = append(out, u32[:]...)
	for _, h := range order {
		info := byHash[h]
		var u64 [8]byte
		binary.LittleEndian.PutUint64(u64[:], uint64(len(info.bytes)))
		out = append(out, u64[:]...)
		out = append(out, info.bytes...)
	}
	var crc [4]byte
	binary.LittleEndian.PutUint32(crc[:], crc32c(out))
	out = append(out, crc[:]...)
	return out, nil
}

// Parse parses and structurally validates an archive without decoding frames.
func Parse(b []byte, lim limits.Limits) (*Archive, error) {
	if len(b) < HeaderLen+TrailerLen {
		return nil, werrors.NewMalformed("archive is shorter than its header and trailer")
	}
	if string(b[0:4]) != Magic {
		return nil, werrors.NewMalformed("bad archive magic (expected WFA1)")
	}
	version := binary.LittleEndian.Uint32(b[4:8])
	if version != Version {
		return nil, werrors.UnsupportedVersionf("unsupported archive version %d", version)
	}
	entryCount := binary.LittleEndian.Uint32(b[8:12])
	if err := limits.CheckLen(uint64(entryCount), lim.MaxArchiveEntries, "archive entry count"); err != nil {
		return nil, err
	}
	crcAt := len(b) - TrailerLen
	storedCRC := binary.LittleEndian.Uint32(b[crcAt:])
	if crc32c(b[:crcAt]) != storedCRC {
		return nil, werrors.NewCorrupt("archive CRC32C mismatch")
	}

	minEntries, err := limits.MulChecked(uint64(entryCount), 8, "archive entry count")
	if err != nil {
		return nil, werrors.NewMalformed(err.Error())
	}
	if minEntries > uint64(crcAt-HeaderLen) {
		return nil, werrors.NewMalformed("archive entry count cannot fit in the remaining bytes")
	}

	pos := HeaderLen
	entries := make([]Entry, 0, int(min(uint64(entryCount), 1024)))
	for i := 0; i < int(entryCount); i++ {
		if pos+8 > crcAt {
			return nil, werrors.Malformedf("archive entry %d is truncated", i)
		}
		frameLen := binary.LittleEndian.Uint64(b[pos : pos+8])
		pos += 8
		if frameLen < 100 {
			return nil, werrors.Malformedf(
				"archive entry %d frame_len %d is below 100", i, frameLen)
		}
		frameLenInt, err := limits.ToInt(frameLen, "archive frame")
		if err != nil {
			return nil, werrors.NewMalformed(err.Error())
		}
		if frameLenInt > crcAt-pos {
			return nil, werrors.Malformedf("archive entry %d extends past the archive body", i)
		}
		frameBytes := b[pos : pos+frameLenInt]
		f, err := frame.Parse(frameBytes, lim)
		if err != nil {
			return nil, err
		}
		entries = append(entries, Entry{FrameBytes: frameBytes, Frame: f})
		pos += frameLenInt
	}
	if pos != crcAt {
		return nil, werrors.NewMalformed(
			"archive has trailing bytes or entry_count undercounts the entries")
	}

	seen := make(map[hash.Hash32][]byte)
	for _, entry := range entries {
		h := entry.Frame.Header.ContentHash
		if existing, ok := seen[h]; ok {
			if !bytesEqual(existing, entry.FrameBytes) {
				return nil, werrors.Corruptf(
					"archive contains two different frames with content hash %s", h)
			}
		} else {
			seen[h] = entry.FrameBytes
		}
	}

	allHashes := make(map[hash.Hash32]bool, len(entries))
	for _, entry := range entries {
		allHashes[entry.Frame.Header.ContentHash] = true
	}
	earlier := make(map[hash.Hash32]bool, len(entries))
	for i, entry := range entries {
		for _, dep := range entry.Frame.Dependencies {
			if !earlier[dep.Hash] {
				if allHashes[dep.Hash] {
					return nil, werrors.Malformedf(
						"archive entry %d depends on a frame that appears later (not topological)", i)
				}
				return nil, werrors.DependencyMissingf(
					"archive entry %d depends on %s, which is not present", i, dep.Hash)
			}
		}
		earlier[entry.Frame.Header.ContentHash] = true
	}

	return &Archive{Entries: entries}, nil
}

// Decode decodes every archive entry in order using only in-archive plaintext.
// Duplicate byte-identical entries are decoded once. The result is in
// topological order.
func Decode(b []byte, lim limits.Limits) ([]DecodedEntry, error) {
	a, err := Parse(b, lim)
	if err != nil {
		return nil, err
	}
	ctx := context.Background()
	plaintext := make(map[hash.Hash32][]byte)
	storedFrames := make(map[hash.Hash32][]byte)
	decodedEntries := make([]DecodedEntry, 0, len(a.Entries))
	var cumulative uint64

	for _, entry := range a.Entries {
		h := entry.Frame.Header.ContentHash
		if previous, ok := storedFrames[h]; ok {
			if !bytesEqual(previous, entry.FrameBytes) {
				return nil, werrors.Corruptf(
					"archive has conflicting frames for content hash %s", h)
			}
			continue
		}
		resolver := &mapResolver{entries: plaintext}
		plain, err := modes.DecodeBytes(ctx, entry.FrameBytes, resolver, lim)
		if err != nil {
			return nil, err
		}
		cumulative, err = limits.AddChecked(cumulative, uint64(len(plain)), "archive decoded bytes")
		if err != nil {
			return nil, err
		}
		if err := limits.CheckLen(cumulative, lim.MaxDecodedBytes, "cumulative archive decoded data"); err != nil {
			return nil, err
		}
		storedFrames[h] = entry.FrameBytes
		plaintext[h] = plain
		decodedEntries = append(decodedEntries, DecodedEntry{ContentHash: h, Bytes: plain})
	}
	return decodedEntries, nil
}

type mapResolver struct {
	entries map[hash.Hash32][]byte
}

func (m *mapResolver) Resolve(ctx context.Context, h hash.Hash32, decodedLen uint64, _ uint8) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, werrors.NewIo(fmt.Sprintf("archive resolution canceled: %v", err))
	}
	b, ok := m.entries[h]
	if !ok {
		return nil, werrors.DependencyMissingf(
			"archive dependency %s has not been decoded yet", h)
	}
	if err := dependency.Verify(h, decodedLen, b); err != nil {
		return nil, err
	}
	out := make([]byte, len(b))
	copy(out, b)
	return out, nil
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

type hashHeap []hash.Hash32

func (h hashHeap) Len() int           { return len(h) }
func (h hashHeap) Less(i, j int) bool { return h[i].Less(h[j]) }
func (h hashHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *hashHeap) Push(x any)        { *h = append(*h, x.(hash.Hash32)) }
func (h *hashHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}
