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

type weightDeltaVector struct {
	Name           string   `json:"name"`
	Schema         string   `json:"schema"`
	Dtype          string   `json:"dtype"`
	Shape          []uint64 `json:"shape"`
	ElementCount   uint64   `json:"element_count"`
	WordBits       uint64   `json:"word_bits"`
	Base           string   `json:"base"`
	BaseDtype      string   `json:"base_dtype"`
	BaseShape      []uint64 `json:"base_shape"`
	Op             string   `json:"op"`
	Planes         uint64   `json:"planes"`
	PlaneCodec     string   `json:"plane_codec"`
	DeltaRoot      string   `json:"delta_root"`
	PayloadLen     uint64   `json:"payload_len"`
	TargetHash     string   `json:"target_hash"`
	TargetHex      string   `json:"target_hex"`
	BaseFrameHex   string   `json:"base_frame_hex"`
	DeltaFrameHex  string   `json:"delta_frame_hex"`
	UseFrameSource bool     `json:"use_frame_source"`
	Expect         struct {
		OK bool `json:"ok"`
	} `json:"expect"`
	ExpectError string `json:"expect_error"`
	Note        string `json:"note"`
}

func TestWeightDeltaVectors(t *testing.T) {
	files := jsonFiles(t, filepath.Join("v1", "weight-delta"))
	if len(files) != 11 {
		t.Fatalf("expected 11 WEIGHT_DELTA vectors, found %d", len(files))
	}
	lim := limits.Default()
	ctx := context.Background()
	for _, path := range files {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			var vector weightDeltaVector
			loadJSON(t, path, &vector)
			deltaBytes := mustHex(t, "delta_frame_hex", vector.DeltaFrameHex)
			parsed, err := frame.Parse(deltaBytes, lim)
			if err != nil {
				t.Fatalf("parse delta frame: %v", err)
			}

			if vector.ExpectError != "" {
				var decodeErr error
				if vector.UseFrameSource {
					source := dependency.NewMapFrameSource()
					if _, err := source.InsertFrame(ctx, deltaBytes, lim); err != nil {
						t.Fatalf("insert root frame: %v", err)
					}
					_, decodeErr = modes.DecodeDAG(ctx, source, deltaBytes, lim)
				} else if vector.BaseFrameHex != "" {
					baseBytes := mustHex(t, "base_frame_hex", vector.BaseFrameHex)
					_, decodeErr = modes.DecodeWeightDeltaFrameWithBase(
						ctx, parsed, baseBytes, dependency.NewMapResolver(), lim)
				} else {
					_, decodeErr = modes.DecodeWeightDeltaFrame(
						ctx, parsed, dependency.NewMapResolver(), lim)
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
			decoded, err := modes.DecodeWeightDeltaFrameWithBase(
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

			meta, err := modes.WeightDeltaMetadata(parsed, lim)
			if err != nil {
				t.Fatalf("metadata: %v", err)
			}
			if meta.Dtype.Name != vector.Dtype ||
				meta.WordBits != vector.WordBits ||
				meta.Planes != vector.Planes ||
				meta.Op != vector.Op ||
				meta.PlaneCodec != vector.PlaneCodec ||
				meta.ElementCount != vector.ElementCount ||
				meta.PayloadLen != vector.PayloadLen ||
				meta.DeltaRoot.Hex() != vector.DeltaRoot ||
				meta.Base.Hex() != vector.Base {
				t.Fatalf("descriptor does not match the vector")
			}
			if meta.BaseDtype.Name != vector.BaseDtype || !equalShape(meta.BaseShape, vector.BaseShape) {
				t.Fatalf("base descriptor does not match the vector")
			}

			reencoded, err := modes.EncodeWeightDelta(ctx, target, meta.Dtype, meta.Shape, baseBytes, lim)
			if err != nil {
				t.Fatalf("re-encode: %v", err)
			}
			if !bytes.Equal(reencoded, deltaBytes) {
				t.Fatalf("Go re-encode is not byte-identical to the committed Rust frame")
			}
		})
	}
}

func equalShape(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for index := range a {
		if a[index] != b[index] {
			return false
		}
	}
	return true
}
