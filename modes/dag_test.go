package modes

import (
	"bytes"
	"context"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func referenceFrame(t *testing.T, contentHash, depHash hash.Hash32, lim limits.Limits) []byte {
	t.Helper()
	parts := frame.PayloadParts(frame.ModeReference, 0, 0, 1, contentHash, nil)
	parts.Dependencies = []frame.Dependency{
		frame.NewDependency(depHash, 1, frame.DepDomainPayload, 0),
	}
	encoded, err := frame.Encode(parts, lim)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

func TestDecodeDAGSucceeds(t *testing.T) {
	ctx := context.Background()
	lim := limits.Default()
	base := []byte("base-bytes-for-the-dag")
	target := []byte("base-bytes-for-the-dag-plus")
	baseFrame, err := EncodeRaw(base, 0, lim)
	if err != nil {
		t.Fatal(err)
	}
	deltaFrame, err := EncodeDelta(target, base, lim)
	if err != nil {
		t.Fatal(err)
	}
	source := dependency.NewMapFrameSource()
	if _, err := source.InsertFrame(ctx, baseFrame, lim); err != nil {
		t.Fatal(err)
	}
	if _, err := source.InsertFrame(ctx, deltaFrame, lim); err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeDAG(ctx, source, deltaFrame, lim)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(decoded, target) {
		t.Fatal("DAG plaintext mismatch")
	}
}

func TestDecodeDAGCycle(t *testing.T) {
	ctx := context.Background()
	lim := limits.Default()
	contentA := hash.FromBytes([32]byte{1})
	contentB := hash.FromBytes([32]byte{2})
	frameA := referenceFrame(t, contentA, contentB, lim)
	frameB := referenceFrame(t, contentB, contentA, lim)
	source := dependency.NewMapFrameSource()
	if _, err := source.InsertFrame(ctx, frameA, lim); err != nil {
		t.Fatal(err)
	}
	if _, err := source.InsertFrame(ctx, frameB, lim); err != nil {
		t.Fatal(err)
	}
	_, err := DecodeDAG(ctx, source, frameA, lim)
	requireClass(t, err, werrors.DependencyCycle)
}

func TestDecodeDAGMissing(t *testing.T) {
	ctx := context.Background()
	lim := limits.Default()
	base := []byte("base")
	target := []byte("base-plus")
	deltaFrame, err := EncodeDelta(target, base, lim)
	if err != nil {
		t.Fatal(err)
	}
	_, err = DecodeDAG(ctx, dependency.NewMapFrameSource(), deltaFrame, lim)
	requireClass(t, err, werrors.DependencyMissing)
}

func TestDecodeDAGDepthLimit(t *testing.T) {
	ctx := context.Background()
	lim := limits.Default()
	base := []byte("base-value")
	mid := []byte("base-value-mid")
	top := []byte("base-value-mid-top")
	baseFrame, err := EncodeRaw(base, 0, lim)
	if err != nil {
		t.Fatal(err)
	}
	midFrame, err := EncodeDelta(mid, base, lim)
	if err != nil {
		t.Fatal(err)
	}
	topFrame, err := EncodeDelta(top, mid, lim)
	if err != nil {
		t.Fatal(err)
	}
	source := dependency.NewMapFrameSource()
	for _, b := range [][]byte{baseFrame, midFrame, topFrame} {
		if _, err := source.InsertFrame(ctx, b, lim); err != nil {
			t.Fatal(err)
		}
	}
	tight := lim
	tight.MaxDependencyDepth = 1
	_, err = DecodeDAG(ctx, source, topFrame, tight)
	requireClass(t, err, werrors.LimitExceeded)
}

func TestDecodeDAGCanceled(t *testing.T) {
	lim := limits.Default()
	baseFrame, err := EncodeRaw([]byte("x"), 0, lim)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = DecodeDAG(ctx, dependency.NewMapFrameSource(), baseFrame, lim)
	requireClass(t, err, werrors.Io)
}
