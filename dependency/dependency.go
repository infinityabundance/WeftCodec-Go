// Package dependency holds the v1 dependency resolution contracts
// (WEFTCODEC-FORMAT §5).
//
// Resolvers are the only door through which external bytes enter a decode. A
// decoder verifies the returned length and domain-separated hash at every
// materialization boundary; a resolver that lies yields DependencyMismatch.
package dependency

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// Resolver returns dependency plaintext for a digest. Implementations must
// honour ctx cancellation. They should return DependencyMissing when bytes are
// unknown and DependencyDenied when policy refuses them.
type Resolver interface {
	Resolve(ctx context.Context, h hash.Hash32, decodedLen uint64, depDomain uint8) ([]byte, error)
}

// Verify checks a resolved dependency's length and chunk-domain hash.
func Verify(h hash.Hash32, decodedLen uint64, b []byte) error {
	if uint64(len(b)) != decodedLen {
		return werrors.DependencyMismatchf(
			"resolved dependency length %d differs from declared %d", len(b), decodedLen)
	}
	if hash.Domain(hash.DomainChunk, b) != h {
		return werrors.DependencyMismatchf("resolved dependency %s fails hash verification", h)
	}
	return nil
}

// MapResolver is an in-memory resolver keyed by digest.
type MapResolver struct {
	mu      sync.RWMutex
	entries map[hash.Hash32][]byte
}

// NewMapResolver returns an empty resolver.
func NewMapResolver() *MapResolver {
	return &MapResolver{entries: make(map[hash.Hash32][]byte)}
}

// Insert stores bytes under an explicit digest.
func (m *MapResolver) Insert(h hash.Hash32, b []byte) {
	cp := make([]byte, len(b))
	copy(cp, b)
	m.mu.Lock()
	m.entries[h] = cp
	m.mu.Unlock()
}

// InsertBytes stores plaintext under its chunk-domain digest and returns it.
func (m *MapResolver) InsertBytes(b []byte) hash.Hash32 {
	h := hash.Domain(hash.DomainChunk, b)
	m.Insert(h, b)
	return h
}

// Len reports the number of entries.
func (m *MapResolver) Len() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.entries)
}

// Resolve implements Resolver.
func (m *MapResolver) Resolve(ctx context.Context, h hash.Hash32, decodedLen uint64, _ uint8) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, werrors.NewIo(fmt.Sprintf("dependency resolution canceled: %v", err))
	}
	m.mu.RLock()
	b, ok := m.entries[h]
	m.mu.RUnlock()
	if !ok {
		return nil, werrors.DependencyMissingf("dependency %s not in map", h)
	}
	if err := Verify(h, decodedLen, b); err != nil {
		return nil, err
	}
	out := make([]byte, len(b))
	copy(out, b)
	return out, nil
}

// DirectoryResolver reads dependencies from files named by lowercase hex
// digest, each containing exactly the dependency plaintext.
type DirectoryResolver struct {
	Dir string
}

// NewDirectoryResolver returns a resolver rooted at dir.
func NewDirectoryResolver(dir string) *DirectoryResolver {
	return &DirectoryResolver{Dir: dir}
}

// Resolve implements Resolver.
func (d *DirectoryResolver) Resolve(ctx context.Context, h hash.Hash32, decodedLen uint64, _ uint8) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, werrors.NewIo(fmt.Sprintf("dependency resolution canceled: %v", err))
	}
	path := filepath.Join(d.Dir, h.Hex())
	info, err := os.Stat(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, werrors.DependencyMissingf("dependency file %s not found", path)
		}
		return nil, werrors.NewIo(fmt.Sprintf("cannot stat dependency %s: %v", path, err))
	}
	if !info.Mode().IsRegular() {
		return nil, werrors.DependencyMissingf("dependency path %s is not a regular file", path)
	}
	if uint64(info.Size()) != decodedLen {
		return nil, werrors.DependencyMismatchf(
			"dependency file %s has length %d but declared %d", path, info.Size(), decodedLen)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, werrors.NewIo(fmt.Sprintf("cannot read dependency %s: %v", path, err))
	}
	if err := Verify(h, decodedLen, b); err != nil {
		return nil, err
	}
	return b, nil
}

// FrameSource supplies complete dependency frames for DAG reconstruction.
type FrameSource interface {
	// GetFrame returns the frame whose content_hash is contentHash, or ok=false.
	GetFrame(ctx context.Context, contentHash hash.Hash32) ([]byte, bool, error)
}

// MapFrameSource is an in-memory frame source keyed by frame content hash.
type MapFrameSource struct {
	mu     sync.RWMutex
	frames map[hash.Hash32][]byte
}

// NewMapFrameSource returns an empty frame source.
func NewMapFrameSource() *MapFrameSource {
	return &MapFrameSource{frames: make(map[hash.Hash32][]byte)}
}

// InsertFrame parses a frame and stores it under its declared content hash.
func (m *MapFrameSource) InsertFrame(ctx context.Context, b []byte, lim limits.Limits) (hash.Hash32, error) {
	if err := ctx.Err(); err != nil {
		return hash.Hash32{}, werrors.NewIo(fmt.Sprintf("frame source canceled: %v", err))
	}
	f, err := frame.Parse(b, lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	h := f.Header.ContentHash
	cp := make([]byte, len(b))
	copy(cp, b)
	m.mu.Lock()
	defer m.mu.Unlock()
	if existing, ok := m.frames[h]; ok && !bytesEqual(existing, b) {
		return hash.Hash32{}, werrors.NewCorrupt(
			"frame source has two different frames with the same content hash")
	}
	m.frames[h] = cp
	return h, nil
}

// Len reports the number of stored frames.
func (m *MapFrameSource) Len() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.frames)
}

// DirectoryFrameSource reads complete frames from files named by lowercase
// content hash. Files are read lazily so the DAG fetch bound counts only frames
// actually traversed.
type DirectoryFrameSource struct {
	Dir           string
	MaxFrameBytes uint64
}

// NewDirectoryFrameSource returns a frame source rooted at dir.
func NewDirectoryFrameSource(dir string, maxFrameBytes uint64) *DirectoryFrameSource {
	return &DirectoryFrameSource{Dir: dir, MaxFrameBytes: maxFrameBytes}
}

// GetFrame implements FrameSource.
func (d *DirectoryFrameSource) GetFrame(ctx context.Context, contentHash hash.Hash32) ([]byte, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, werrors.NewIo("frame source canceled: " + err.Error())
	}
	path := filepath.Join(d.Dir, contentHash.Hex())
	info, err := os.Stat(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, false, nil
		}
		return nil, false, werrors.NewIo(fmt.Sprintf("cannot stat frame %s: %v", path, err))
	}
	if !info.Mode().IsRegular() {
		return nil, false, nil
	}
	if err := limits.CheckLen(uint64(info.Size()), d.MaxFrameBytes, "dependency frame"); err != nil {
		return nil, false, err
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, false, werrors.NewIo(fmt.Sprintf("cannot read frame %s: %v", path, err))
	}
	if err := limits.CheckLen(uint64(len(b)), d.MaxFrameBytes, "dependency frame"); err != nil {
		return nil, false, err
	}
	return b, true, nil
}

// GetFrame implements FrameSource.
func (m *MapFrameSource) GetFrame(ctx context.Context, contentHash hash.Hash32) ([]byte, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, werrors.NewIo(fmt.Sprintf("frame source canceled: %v", err))
	}
	m.mu.RLock()
	b, ok := m.frames[contentHash]
	m.mu.RUnlock()
	if !ok {
		return nil, false, nil
	}
	out := make([]byte, len(b))
	copy(out, b)
	return out, true, nil
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
