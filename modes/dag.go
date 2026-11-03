package modes

import (
	"context"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

type dagStats struct {
	fetched uint64
	decoded uint64
}

type dagResolver struct {
	ctx      context.Context
	source   dependency.FrameSource
	limits   limits.Limits
	visiting map[hash.Hash32]bool
	stats    *dagStats
	depth    uint64
}

// Resolve implements dependency.Resolver over frame sources.
func (d *dagResolver) Resolve(ctx context.Context, h hash.Hash32, decodedLen uint64, _ uint8) ([]byte, error) {
	if err := d.ctx.Err(); err != nil {
		return nil, werrors.NewIo("dependency resolution canceled: " + err.Error())
	}
	if d.depth == ^uint64(0) {
		return nil, werrors.NewLimitExceeded("dependency depth overflow")
	}
	depth := d.depth + 1
	if depth > d.limits.MaxDependencyDepth {
		return nil, werrors.LimitExceededf(
			"dependency depth %d exceeds limit %d", depth, d.limits.MaxDependencyDepth)
	}
	if d.visiting[h] {
		return nil, werrors.DependencyCyclef("dependency cycle re-entered frame %s", h)
	}
	frameBytes, ok, err := d.source.GetFrame(ctx, h)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, werrors.DependencyMissingf("frame source has no frame for %s", h)
	}
	newFetched, err := limits.AddChecked(d.stats.fetched, uint64(len(frameBytes)), "fetched bytes")
	if err != nil {
		return nil, err
	}
	d.stats.fetched = newFetched
	if err := limits.CheckLen(d.stats.fetched, d.limits.MaxTotalFetchedBytes, "total fetched bytes"); err != nil {
		return nil, err
	}
	f, err := frame.Parse(frameBytes, d.limits)
	if err != nil {
		return nil, err
	}
	if f.Header.ContentHash != h {
		return nil, werrors.DependencyMismatchf(
			"frame at %s declares content hash %s", h, f.Header.ContentHash)
	}

	d.visiting[h] = true
	child := &dagResolver{
		ctx:      d.ctx,
		source:   d.source,
		limits:   d.limits,
		visiting: d.visiting,
		stats:    d.stats,
		depth:    depth,
	}
	decoded, err := DecodeBytes(ctx, frameBytes, child, d.limits)
	delete(d.visiting, h)
	if err != nil {
		return nil, err
	}
	if uint64(len(decoded)) != decodedLen {
		return nil, werrors.DependencyMismatchf(
			"dependency frame %s decodes to %d bytes, declared %d", h, len(decoded), decodedLen)
	}
	if hash.Domain(hash.DomainChunk, decoded) != h {
		return nil, werrors.NewDependencyMismatch("dependency frame plaintext hash mismatch")
	}
	newDecoded, err := limits.AddChecked(d.stats.decoded, uint64(len(decoded)), "decoded bytes")
	if err != nil {
		return nil, err
	}
	d.stats.decoded = newDecoded
	if err := limits.CheckLen(d.stats.decoded, d.limits.MaxDecodedBytes, "cumulative decoded dependency bytes"); err != nil {
		return nil, err
	}
	return decoded, nil
}

// DecodeDAG recursively decodes a frame DAG.
//
// Dependency frames are fetched by content hash through source, decoded
// recursively with a visiting set (cycle detection), and verified at every
// boundary. Depth, fetched bytes, and cumulative decoded bytes are enforced.
func DecodeDAG(ctx context.Context, source dependency.FrameSource, rootFrame []byte, lim limits.Limits) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, werrors.NewIo("decode canceled: " + err.Error())
	}
	root, err := frame.Parse(rootFrame, lim)
	if err != nil {
		return nil, err
	}
	rootHash := root.Header.ContentHash
	stats := &dagStats{fetched: uint64(len(rootFrame))}
	if err := limits.CheckLen(stats.fetched, lim.MaxTotalFetchedBytes, "total fetched bytes"); err != nil {
		return nil, err
	}
	visiting := map[hash.Hash32]bool{rootHash: true}
	resolver := &dagResolver{
		ctx:      ctx,
		source:   source,
		limits:   lim,
		visiting: visiting,
		stats:    stats,
		depth:    0,
	}
	decoded, err := DecodeBytes(ctx, rootFrame, resolver, lim)
	delete(visiting, rootHash)
	return decoded, err
}
