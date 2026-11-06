package conformance_test

import (
	"bytes"
	"context"
	"path/filepath"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

type sparseDeltaVector struct {
	Name           string   `json:"name"`
	Schema         string   `json:"schema"`
	Dtype          string   `json:"dtype"`
	Shape          []uint64 `json:"shape"`
	ElementCount   uint64   `json:"element_count"`
	StorageBytes   uint64   `json:"storage_bytes"`
	Op             string   `json:"op"`
	SparseCodec    string   `json:"sparse_codec"`
	Base           string   `json:"base"`
	DeltaRoot      string   `json:"delta_root"`
	RunCount       uint32   `json:"run_count"`
	PayloadLen     uint64   `json:"payload_len"`
	TargetHash     string   `json:"target_hash"`
	TargetHex      string   `json:"target_hex"`
	BaseFrameHex   string   `json:"base_frame_hex"`
	DeltaFrameHex  string   `json:"delta_frame_hex"`
	UseFrameSource bool     `json:"use_frame_source"`
	Limits         struct {
		MaxSparseRuns *uint64 `json:"max_sparse_runs"`
	} `json:"limits"`
	Expect struct {
		OK bool `json:"ok"`
	} `json:"expect"`
	ExpectError string `json:"expect_error"`
	Note        string `json:"note"`
}

func TestSparseDeltaVectors(t *testing.T) {
	files := jsonFiles(t, filepath.Join("v1", "sparse-delta"))
	if len(files) != 18 {
		t.Fatalf("expected 18 SPARSE_DELTA vectors, found %d", len(files))
	}
	lim := limits.Default()
	ctx := context.Background()
	for _, path := range files {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			var vector sparseDeltaVector
			loadJSON(t, path, &vector)
			deltaBytes := mustHex(t, "delta_frame_hex", vector.DeltaFrameHex)
			parsed, err := frame.Parse(deltaBytes, lim)
			if err != nil {
				t.Fatalf("parse delta frame: %v", err)
			}

			if vector.ExpectError != "" {
				vectorLimits := lim
				if vector.Limits.MaxSparseRuns != nil {
					vectorLimits.MaxSparseRuns = *vector.Limits.MaxSparseRuns
				}
				var decodeErr error
				if vector.UseFrameSource {
					source := dependency.NewMapFrameSource()
					if _, err := source.InsertFrame(ctx, deltaBytes, vectorLimits); err != nil {
						t.Fatalf("insert root frame: %v", err)
					}
					_, decodeErr = modes.DecodeDAG(ctx, source, deltaBytes, vectorLimits)
				} else if vector.BaseFrameHex != "" {
					baseBytes := mustHex(t, "base_frame_hex", vector.BaseFrameHex)
					_, decodeErr = modes.DecodeSparseDeltaFrameWithBase(
						ctx, parsed, baseBytes, dependency.NewMapResolver(), vectorLimits)
				} else {
					_, decodeErr = modes.DecodeSparseDeltaFrame(
						ctx, parsed, dependency.NewMapResolver(), vectorLimits)
				}
				if decodeErr == nil {
					t.Fatalf("expected %s, operation succeeded", vector.ExpectError)
				}
				if class := string(werrors.ClassOf(decodeErr)); class != vector.ExpectError {
					t.Fatalf("expected class %s, got %s (%v)",
						vector.ExpectError, class, decodeErr)
				}
				return
			}

			baseBytes := mustHex(t, "base_frame_hex", vector.BaseFrameHex)
			target := mustHex(t, "target_hex", vector.TargetHex)
			decoded, err := modes.DecodeSparseDeltaFrameWithBase(
				ctx, parsed, baseBytes, dependency.NewMapResolver(), lim)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if !bytes.Equal(decoded, target) {
				t.Fatalf("reconstructed tensor differs from the target")
			}
			if got := hash.Domain(hash.DomainChunk, decoded).Hex(); got != vector.TargetHash {
				t.Fatalf("target hash %s, want %s", got, vector.TargetHash)
			}

			meta, err := modes.SparseDeltaMetadata(parsed, lim)
			if err != nil {
				t.Fatalf("metadata: %v", err)
			}
			if meta.Dtype.Name != vector.Dtype ||
				meta.StorageBytes != vector.StorageBytes ||
				meta.Op != vector.Op ||
				meta.SparseCodec != vector.SparseCodec ||
				meta.ElementCount != vector.ElementCount ||
				meta.DeltaRoot.Hex() != vector.DeltaRoot ||
				meta.Base.Hex() != vector.Base {
				t.Fatalf("descriptor does not match the vector")
			}
			if !equalShape(meta.Shape, vector.Shape) {
				t.Fatalf("shape does not match the vector")
			}

			reencoded, err := modes.EncodeSparseDelta(
				ctx, target, meta.Dtype, meta.Shape, baseBytes, lim)
			if err != nil {
				t.Fatalf("re-encode: %v", err)
			}
			if !bytes.Equal(reencoded, deltaBytes) {
				t.Fatalf("Go re-encode is not byte-identical to the committed Rust frame")
			}
		})
	}
}
