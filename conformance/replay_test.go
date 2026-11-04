package conformance_test

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/selector"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func jsonFiles(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("cannot read %s: %v", dir, err)
	}
	var paths []string
	for _, entry := range entries {
		if !entry.IsDir() && filepath.Ext(entry.Name()) == ".json" {
			paths = append(paths, filepath.Join(dir, entry.Name()))
		}
	}
	sort.Strings(paths)
	return paths
}

func loadJSON(t *testing.T, path string, target any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("cannot read %s: %v", path, err)
	}
	if err := json.Unmarshal(b, target); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}

func mustHex(t *testing.T, what, s string) []byte {
	t.Helper()
	b, err := hex.DecodeString(s)
	if err != nil {
		t.Fatalf("%s: bad hex: %v", what, err)
	}
	return b
}

type depVector struct {
	Hash       string `json:"hash"`
	DecodedLen uint64 `json:"decoded_len"`
	DepDomain  uint8  `json:"dep_domain"`
	BytesHex   string `json:"bytes_hex"`
}

type expectation struct {
	OK    *bool  `json:"ok"`
	Error string `json:"error"`
}

type codecVector struct {
	Name          string            `json:"name"`
	Mode          string            `json:"mode"`
	ModeValue     *uint8            `json:"mode_value"`
	Flags         *uint16           `json:"flags"`
	Features      *uint16           `json:"features"`
	DecodedLen    *uint64           `json:"decoded_len"`
	BodyLen       *uint64           `json:"body_len"`
	ChunkCount    *uint64           `json:"chunk_count"`
	ContentHash   string            `json:"content_hash"`
	PlaintextHex  string            `json:"plaintext_hex"`
	PlaintextHash string            `json:"plaintext_hash"`
	FrameHex      string            `json:"frame_hex"`
	Dependencies  []depVector       `json:"dependencies"`
	Expect        expectation       `json:"expect"`
	Limits        map[string]uint64 `json:"limits"`
	SupersededBy  string            `json:"superseded_by"`
}

func limitsFrom(m map[string]uint64) limits.Limits {
	lim := limits.Default()
	for key, value := range m {
		switch key {
		case "max_frame_bytes":
			lim.MaxFrameBytes = value
		case "max_decoded_bytes":
			lim.MaxDecodedBytes = value
		case "max_dependency_depth":
			lim.MaxDependencyDepth = value
		case "max_dependency_count":
			lim.MaxDependencyCount = value
		case "max_total_fetched_bytes":
			lim.MaxTotalFetchedBytes = value
		case "max_chunks":
			lim.MaxChunks = value
		case "max_dictionary_bytes":
			lim.MaxDictionaryBytes = value
		case "max_structured_depth":
			lim.MaxStructuredDepth = value
		case "max_structured_items":
			lim.MaxStructuredItems = value
		case "max_string_bytes":
			lim.MaxStringBytes = value
		case "max_delta_ops":
			lim.MaxDeltaOps = value
		case "max_tensor_elements":
			lim.MaxTensorElements = value
		case "max_tensor_rank":
			lim.MaxTensorRank = value
		case "max_tokens":
			lim.MaxTokens = value
		case "max_selector_bytes":
			lim.MaxSelectorBytes = value
		case "max_selector_depth":
			lim.MaxSelectorDepth = value
		case "max_field_segments":
			lim.MaxFieldSegments = value
		case "max_selector_operands":
			lim.MaxSelectorOperands = value
		case "max_archive_entries":
			lim.MaxArchiveEntries = value
		}
	}
	return lim
}

func resolverFor(t *testing.T, deps []depVector) *dependency.MapResolver {
	t.Helper()
	resolver := dependency.NewMapResolver()
	for _, dep := range deps {
		h, err := hash.FromHex(dep.Hash)
		if err != nil {
			t.Fatalf("dependency hash %q: %v", dep.Hash, err)
		}
		b := mustHex(t, "dependency bytes", dep.BytesHex)
		if uint64(len(b)) != dep.DecodedLen {
			t.Fatalf("dependency vector declares %d bytes but has %d", dep.DecodedLen, len(b))
		}
		resolver.Insert(h, b)
	}
	return resolver
}

func TestCodecVectors(t *testing.T) {
	files := jsonFiles(t, filepath.Join("v1", "codec"))
	if len(files) < 20 {
		t.Fatalf("expected at least 20 codec vectors, found %d", len(files))
	}
	ctx := context.Background()
	for _, path := range files {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			var vector codecVector
			loadJSON(t, path, &vector)
			if vector.SupersededBy != "" {
				t.Skipf("superseded by %s", vector.SupersededBy)
			}
			frameBytes := mustHex(t, "frame", vector.FrameHex)
			lim := limitsFrom(vector.Limits)
			resolver := resolverFor(t, vector.Dependencies)

			if vector.Expect.Error != "" {
				_, err := modes.DecodeBytes(ctx, frameBytes, resolver, lim)
				if err == nil {
					t.Fatalf("expected %s, decode succeeded", vector.Expect.Error)
				}
				if got := string(werrors.ClassOf(err)); got != vector.Expect.Error {
					t.Fatalf("expected class %s, got %s (%v)", vector.Expect.Error, got, err)
				}
				return
			}

			decoded, err := modes.DecodeBytes(ctx, frameBytes, resolver, lim)
			if err != nil {
				t.Fatalf("decode failed: %v", err)
			}
			plaintext := mustHex(t, "plaintext", vector.PlaintextHex)
			if !bytes.Equal(decoded, plaintext) {
				t.Fatalf("plaintext mismatch: got %d bytes, want %d", len(decoded), len(plaintext))
			}
			if got := hash.Domain(hash.DomainChunk, decoded).Hex(); got != vector.PlaintextHash {
				t.Fatalf("plaintext hash %s, want %s", got, vector.PlaintextHash)
			}
			parsed, err := frame.Parse(frameBytes, lim)
			if err != nil {
				t.Fatalf("parse failed: %v", err)
			}
			if parsed.Header.Mode.String() != vector.Mode {
				t.Fatalf("mode %s, want %s", parsed.Header.Mode, vector.Mode)
			}
			if vector.ModeValue != nil && uint8(parsed.Header.Mode) != *vector.ModeValue {
				t.Fatalf("mode value %d, want %d", parsed.Header.Mode, *vector.ModeValue)
			}
			if vector.Flags != nil && parsed.Header.Flags != *vector.Flags {
				t.Fatalf("flags %d, want %d", parsed.Header.Flags, *vector.Flags)
			}
			if vector.Features != nil && parsed.Header.Features != *vector.Features {
				t.Fatalf("features %d, want %d", parsed.Header.Features, *vector.Features)
			}
			if vector.DecodedLen != nil && parsed.Header.DecodedLen != *vector.DecodedLen {
				t.Fatalf("decoded_len %d, want %d", parsed.Header.DecodedLen, *vector.DecodedLen)
			}
			if vector.BodyLen != nil && parsed.Header.BodyLen != *vector.BodyLen {
				t.Fatalf("body_len %d, want %d", parsed.Header.BodyLen, *vector.BodyLen)
			}
			if vector.ChunkCount != nil && uint64(parsed.Header.ChunkCount) != *vector.ChunkCount {
				t.Fatalf("chunk_count %d, want %d", parsed.Header.ChunkCount, *vector.ChunkCount)
			}
			if parsed.Header.ContentHash.Hex() != vector.ContentHash {
				t.Fatalf("content hash %s, want %s", parsed.Header.ContentHash, vector.ContentHash)
			}
		})
	}
}

type structuredVector struct {
	Name         string            `json:"name"`
	Description  string            `json:"description"`
	CanonicalHex string            `json:"canonical_hex"`
	Hex          string            `json:"hex"`
	Expect       string            `json:"expect"`
	ExpectError  string            `json:"expect_error"`
	Limits       map[string]uint64 `json:"limits"`
}

func TestStructuredVectors(t *testing.T) {
	files := jsonFiles(t, filepath.Join("v1", "structured"))
	if len(files) < 10 {
		t.Fatalf("expected at least 10 structured vectors, found %d", len(files))
	}
	for _, path := range files {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			var vector structuredVector
			loadJSON(t, path, &vector)
			lim := limitsFrom(vector.Limits)
			if vector.ExpectError != "" {
				b := mustHex(t, "value", vector.Hex)
				_, err := wcb.Decode(b, lim)
				if err == nil {
					t.Fatalf("expected %s, decode succeeded", vector.ExpectError)
				}
				if got := string(werrors.ClassOf(err)); got != vector.ExpectError {
					t.Fatalf("expected class %s, got %s (%v)", vector.ExpectError, got, err)
				}
				return
			}
			b := mustHex(t, "canonical", vector.CanonicalHex)
			value, err := wcb.Decode(b, lim)
			if err != nil {
				t.Fatalf("decode failed: %v", err)
			}
			reencoded, err := wcb.Encode(value, lim)
			if err != nil {
				t.Fatalf("encode failed: %v", err)
			}
			if !bytes.Equal(reencoded, b) {
				t.Fatalf("canonical bytes differ:\n got %x\nwant %x", reencoded, b)
			}
		})
	}
}

type selectorVector struct {
	Name         string `json:"name"`
	Source       string `json:"source"`
	Namespace    string `json:"namespace"`
	CanonicalHex string `json:"canonical_hex"`
	SelectorID   string `json:"selector_id"`
}

func TestSelectorVectors(t *testing.T) {
	files := jsonFiles(t, filepath.Join("v1", "selectors"))
	if len(files) < 10 {
		t.Fatalf("expected at least 10 selector vectors, found %d", len(files))
	}
	lim := limits.Default()
	for _, path := range files {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			var vector selectorVector
			loadJSON(t, path, &vector)
			namespace := vector.Namespace
			if namespace == "" {
				namespace = selector.DefaultNamespace
			}
			parsed, err := selector.ParseWith(vector.Source, namespace, lim)
			if err != nil {
				t.Fatalf("parse failed: %v", err)
			}
			canonical, err := parsed.CanonicalBytes(lim)
			if err != nil {
				t.Fatalf("canonical bytes failed: %v", err)
			}
			if got := hex.EncodeToString(canonical); got != vector.CanonicalHex {
				t.Fatalf("canonical hex\n got %s\nwant %s", got, vector.CanonicalHex)
			}
			id, err := parsed.SelectorID(lim)
			if err != nil {
				t.Fatalf("selector id failed: %v", err)
			}
			if id.Hex() != vector.SelectorID {
				t.Fatalf("selector id %s, want %s", id, vector.SelectorID)
			}
			decoded, err := selector.FromCanonicalBytes(canonical, lim)
			if err != nil {
				t.Fatalf("canonical decode failed: %v", err)
			}
			if !decoded.Equal(parsed) {
				t.Fatalf("AST round-trip mismatch")
			}
		})
	}
}

// TestMetadataByteEquality proves the Go encoder reproduces the frozen Rust
// frame bytes for the byte-deterministic modes: RAW (chunked and unchunked),
// REFERENCE, DELTA, and STRUCTURED.
func TestMetadataByteEquality(t *testing.T) {
	lim := limits.Default()
	cases := []string{
		"001_raw_empty",
		"002_raw_one_byte",
		"003_raw_4k",
		"005_raw_64k_repetitive",
		"006_raw_chunked_1024",
		"009_reference_equal",
		"010_delta_edits",
		"012_structured_map",
		"100_tensor_f32_2x3",
		"101_tensor_f16_scalar",
		"102_tensor_u8_zero_element",
		"103_tokens_delta_boundaries",
		"104_tokens_empty",
		"105_quantized_i8_2x3",
		"106_quantized_u8_zero_element",
		"107_quantized_i8_scalar",
	}
	for _, name := range cases {
		name := name
		t.Run(name, func(t *testing.T) {
			path := filepath.Join("v1", "codec", name+".json")
			var vector codecVector
			loadJSON(t, path, &vector)
			plaintext := mustHex(t, "plaintext", vector.PlaintextHex)
			want := mustHex(t, "frame", vector.FrameHex)
			var encoded []byte
			var err error
			switch vector.Mode {
			case "raw":
				chunkSize := 0
				if vector.ChunkCount != nil && *vector.ChunkCount > 0 {
					// The frozen chunked vector uses 1024-byte chunks.
					chunkSize = 1024
				}
				encoded, err = modes.EncodeRaw(plaintext, chunkSize, lim)
			case "reference":
				if len(vector.Dependencies) != 1 {
					t.Fatalf("reference vector has %d dependencies", len(vector.Dependencies))
				}
				base := mustHex(t, "base", vector.Dependencies[0].BytesHex)
				encoded, err = modes.EncodeReference(plaintext, base, lim)
			case "delta":
				if len(vector.Dependencies) != 1 {
					t.Fatalf("delta vector has %d dependencies", len(vector.Dependencies))
				}
				base := mustHex(t, "base", vector.Dependencies[0].BytesHex)
				encoded, err = modes.EncodeDelta(plaintext, base, lim)
			case "structured":
				value, decodeErr := wcb.Decode(plaintext, lim)
				if decodeErr != nil {
					t.Fatalf("structured body: %v", decodeErr)
				}
				encoded, err = modes.EncodeStructured(value, lim)
			case "tensor":
				parsed, parseErr := frame.Parse(want, lim)
				if parseErr != nil {
					t.Fatalf("parse: %v", parseErr)
				}
				meta, metaErr := modes.TensorMetadata(parsed, lim)
				if metaErr != nil {
					t.Fatalf("metadata: %v", metaErr)
				}
				encoded, err = modes.EncodeTensor(plaintext, meta.Dtype, meta.Shape, lim)
			case "tokens":
				parsed, parseErr := frame.Parse(want, lim)
				if parseErr != nil {
					t.Fatalf("parse: %v", parseErr)
				}
				meta, metaErr := modes.TokenMetadata(parsed, lim)
				if metaErr != nil {
					t.Fatalf("metadata: %v", metaErr)
				}
				ids, idsErr := modes.DecodeTokenPayload(plaintext, meta.TokenCount, lim)
				if idsErr != nil {
					t.Fatalf("token ids: %v", idsErr)
				}
				encoded, err = modes.EncodeTokens(ids, meta.VocabSchema, lim)
			case "quantized_tensor":
				parsed, parseErr := frame.Parse(want, lim)
				if parseErr != nil {
					t.Fatalf("parse: %v", parseErr)
				}
				meta, metaErr := modes.QuantizedTensorMetadata(parsed, lim)
				if metaErr != nil {
					t.Fatalf("metadata: %v", metaErr)
				}
				encoded, err = modes.EncodeQuantizedTensor(
					plaintext, meta.Dtype, meta.Shape, meta.Scale, meta.ZeroPoint, lim)
			default:
				t.Skipf("mode %s is not byte-deterministic", vector.Mode)
			}
			if err != nil {
				t.Fatalf("encode failed: %v", err)
			}
			if !bytes.Equal(encoded, want) {
				t.Fatalf("frame bytes differ for %s:\n got %s\nwant %s",
					name, hex.EncodeToString(encoded), vector.FrameHex)
			}
		})
	}
}
