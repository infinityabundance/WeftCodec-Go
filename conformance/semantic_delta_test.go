package conformance_test

import (
	"bytes"
	"context"
	"encoding/hex"
	"path/filepath"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

type semanticPolicyVector struct {
	Scheme    string `json:"scheme"`
	Bits      uint32 `json:"bits"`
	BlockSize uint64 `json:"block_size"`
	Levels    uint64 `json:"levels"`
}

func (p semanticPolicyVector) toPolicy() modes.SemanticPolicy {
	switch p.Scheme {
	case "none":
		return modes.NonePolicy()
	case "mantissa_truncate":
		return modes.MantissaTruncatePolicy(p.Bits)
	case "block_quantize":
		return modes.BlockQuantizePolicy(p.BlockSize, p.Levels)
	default:
		return modes.SemanticPolicy{Scheme: p.Scheme}
	}
}

type semanticRootVector struct {
	Name                string               `json:"name"`
	Kind                string               `json:"kind"`
	Schema              string               `json:"schema"`
	Dtype               string               `json:"dtype"`
	Shape               []uint64             `json:"shape"`
	ElementCount        uint64               `json:"element_count"`
	Policy              semanticPolicyVector `json:"policy"`
	ValuesHex           string               `json:"values_hex"`
	QuantizedValuesHash string               `json:"quantized_values_hash"`
	CanonicalHex        string               `json:"canonical_hex"`
	SemanticRoot        string               `json:"semantic_root"`
	RecordHex           string               `json:"record_hex"`
	ExpectError         string               `json:"expect_error"`
	Note                string               `json:"note"`
}

func TestSemanticRootVectors(t *testing.T) {
	files := jsonFiles(t, filepath.Join("v1", "semantic-root"))
	if len(files) != 9 {
		t.Fatalf("expected 9 semantic-root vectors, found %d", len(files))
	}
	lim := limits.Default()
	for _, path := range files {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			var vector semanticRootVector
			loadJSON(t, path, &vector)
			if vector.ExpectError != "" {
				record := mustHex(t, "record_hex", vector.RecordHex)
				_, err := modes.ParseSemanticRecord(record, lim)
				if err == nil {
					t.Fatalf("expected %s, operation succeeded", vector.ExpectError)
				}
				if class := string(werrors.ClassOf(err)); class != vector.ExpectError {
					t.Fatalf("expected class %s, got %s (%v)", vector.ExpectError, class, err)
				}
				return
			}
			dtype, err := modes.ParseWeightDtype(vector.Dtype)
			if err != nil {
				t.Fatalf("dtype: %v", err)
			}
			values := mustHex(t, "values_hex", vector.ValuesHex)
			record, err := modes.NewSemanticRecord(dtype, vector.Shape, values, vector.Policy.toPolicy(), lim)
			if err != nil {
				t.Fatalf("record: %v", err)
			}
			if record.QuantizedValuesHash.Hex() != vector.QuantizedValuesHash {
				t.Fatalf("values hash %s, want %s",
					record.QuantizedValuesHash.Hex(), vector.QuantizedValuesHash)
			}
			canonical, err := record.CanonicalBytes(lim)
			if err != nil {
				t.Fatalf("canonical: %v", err)
			}
			if got := hex.EncodeToString(canonical); got != vector.CanonicalHex {
				t.Fatalf("canonical bytes %s, want %s", got, vector.CanonicalHex)
			}
			root, err := record.Root(lim)
			if err != nil {
				t.Fatalf("root: %v", err)
			}
			if root.Hex() != vector.SemanticRoot {
				t.Fatalf("root %s, want %s", root.Hex(), vector.SemanticRoot)
			}
			decoded, err := modes.ParseSemanticRecord(canonical, lim)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			if !decoded.Equal(record) {
				t.Fatalf("record round trip differs")
			}
		})
	}
}

type semanticDeltaVector struct {
	Name             string               `json:"name"`
	Schema           string               `json:"schema"`
	Dtype            string               `json:"dtype"`
	Shape            []uint64             `json:"shape"`
	ElementCount     uint64               `json:"element_count"`
	WordBits         uint64               `json:"word_bits"`
	Base             string               `json:"base"`
	BaseDtype        string               `json:"base_dtype"`
	BaseShape        []uint64             `json:"base_shape"`
	BaseSemanticRoot string               `json:"base_semantic_root"`
	Policy           semanticPolicyVector `json:"policy"`
	Planes           uint64               `json:"planes"`
	PlaneCodec       string               `json:"plane_codec"`
	DeltaRoot        string               `json:"delta_root"`
	PayloadLen       uint64               `json:"payload_len"`
	TargetHash       string               `json:"target_hash"`
	TargetHex        string               `json:"target_hex"`
	BaseFrameHex     string               `json:"base_frame_hex"`
	DeltaFrameHex    string               `json:"delta_frame_hex"`
	UseFrameSource   bool                 `json:"use_frame_source"`
	Expect           struct {
		OK bool `json:"ok"`
	} `json:"expect"`
	ExpectError string `json:"expect_error"`
	Note        string `json:"note"`
}

func TestSemanticDeltaVectors(t *testing.T) {
	files := jsonFiles(t, filepath.Join("v1", "semantic-delta"))
	if len(files) != 9 {
		t.Fatalf("expected 9 SEMANTIC_DELTA vectors, found %d", len(files))
	}
	lim := limits.Default()
	ctx := context.Background()
	for _, path := range files {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			var vector semanticDeltaVector
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
					_, decodeErr = modes.DecodeSemanticDeltaFrameWithBase(
						ctx, parsed, baseBytes, dependency.NewMapResolver(), lim)
				} else {
					_, decodeErr = modes.DecodeSemanticDeltaFrame(
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
			decoded, err := modes.DecodeSemanticDeltaFrameWithBase(
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
			meta, err := modes.SemanticDeltaMetadata(parsed, lim)
			if err != nil {
				t.Fatalf("metadata: %v", err)
			}
			if meta.Dtype.Name != vector.Dtype ||
				meta.WordBits != vector.WordBits ||
				meta.ElementCount != vector.ElementCount ||
				meta.PayloadLen != vector.PayloadLen ||
				meta.PlaneCodec != vector.PlaneCodec ||
				meta.DeltaRoot.Hex() != vector.DeltaRoot ||
				meta.Base.Hex() != vector.Base ||
				meta.BaseSemanticRoot.Hex() != vector.BaseSemanticRoot {
				t.Fatalf("descriptor does not match the vector")
			}
			if meta.BaseDtype.Name != vector.BaseDtype ||
				!equalShape(meta.BaseShape, vector.BaseShape) ||
				!equalShape(meta.Shape, vector.Shape) {
				t.Fatalf("shape descriptor does not match the vector")
			}
			if meta.Policy != vector.Policy.toPolicy() {
				t.Fatalf("policy %v, want %v", meta.Policy, vector.Policy.toPolicy())
			}
			reencoded, err := modes.EncodeSemanticDelta(
				ctx, target, meta.Dtype, meta.Shape, baseBytes, meta.Policy, lim)
			if err != nil {
				t.Fatalf("re-encode: %v", err)
			}
			if !bytes.Equal(reencoded, deltaBytes) {
				t.Fatalf("Go re-encode is not byte-identical to the committed Rust frame")
			}
		})
	}
}
