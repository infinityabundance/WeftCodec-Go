package conformance_test

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

type kvProvPageVector struct {
	PageIndex      uint64 `json:"page_index"`
	SeqStart       uint64 `json:"seq_start"`
	SeqEnd         uint64 `json:"seq_end"`
	PageRows       uint64 `json:"page_rows"`
	PageBytes      uint64 `json:"page_bytes"`
	PayloadLen     uint64 `json:"payload_len"`
	DescriptorRoot string `json:"descriptor_root"`
	ElementsHex    string `json:"elements_hex"`
	FrameHex       string `json:"frame_hex"`
}

type kvProvAttentionVector struct {
	HeadCount     uint32 `json:"head_count"`
	HeadDim       uint32 `json:"head_dim"`
	SlidingWindow uint64 `json:"sliding_window"`
	RopeThetaBits uint64 `json:"rope_theta_bits"`
	RopeScaleBits uint64 `json:"rope_scale_bits"`
}

type kvProvTokenizerVector struct {
	Scheme    string `json:"scheme"`
	VocabSize uint64 `json:"vocab_size"`
}

type kvProvLayoutVector struct {
	Dtype      string `json:"dtype"`
	PageRows   uint64 `json:"page_rows"`
	LayerCount uint64 `json:"layer_count"`
}

type kvProvExtensionVector struct {
	Name  string          `json:"name"`
	Value json.RawMessage `json:"value"`
}

type kvProvProvenanceVector struct {
	ModelRoot    string                  `json:"model_root"`
	AdapterRoots []string                `json:"adapter_roots"`
	Tokenizer    kvProvTokenizerVector   `json:"tokenizer"`
	Attention    kvProvAttentionVector   `json:"attention"`
	Positional   string                  `json:"positional"`
	Layout       kvProvLayoutVector      `json:"layout"`
	Salt         string                  `json:"salt"`
	Extensions   []kvProvExtensionVector `json:"extensions"`
}

type kvProvVector struct {
	Name           string                  `json:"name"`
	Schema         string                  `json:"schema"`
	Model          string                  `json:"model"`
	Layer          uint32                  `json:"layer"`
	Session        string                  `json:"session"`
	Dtype          string                  `json:"dtype"`
	HeadCount      uint32                  `json:"head_count"`
	HeadDim        uint32                  `json:"head_dim"`
	PageCount      uint64                  `json:"page_count"`
	DescriptorRoot string                  `json:"descriptor_root"`
	Provenance     *kvProvProvenanceVector `json:"provenance"`
	Pages          []kvProvPageVector      `json:"pages"`
	Expect         *struct {
		OK bool `json:"ok"`
	} `json:"expect"`
	ExpectError  string `json:"expect_error"`
	FrameHex     string `json:"frame_hex"`
	SupersededBy string `json:"superseded_by"`
}

func jsonValueToWCB(value any) (wcb.Value, error) {
	switch v := value.(type) {
	case nil:
		return wcb.NullValue(), nil
	case bool:
		return wcb.BoolValue(v), nil
	case string:
		return wcb.TextValue(v), nil
	case json.Number:
		if unsigned, err := strconv.ParseUint(v.String(), 10, 64); err == nil {
			return wcb.U64Value(unsigned), nil
		}
		if signed, err := strconv.ParseInt(v.String(), 10, 64); err == nil {
			return wcb.I64Value(signed), nil
		}
		return wcb.Value{}, werrors.NewMalformed("extension number is not an integer")
	case []any:
		items := make([]wcb.Value, 0, len(v))
		for _, item := range v {
			converted, err := jsonValueToWCB(item)
			if err != nil {
				return wcb.Value{}, err
			}
			items = append(items, converted)
		}
		return wcb.ArrayValue(items), nil
	case map[string]any:
		out := make(map[string]wcb.Value, len(v))
		for key, item := range v {
			converted, err := jsonValueToWCB(item)
			if err != nil {
				return wcb.Value{}, err
			}
			out[key] = converted
		}
		return wcb.MapValue(out), nil
	default:
		return wcb.Value{}, werrors.NewMalformed("extension value has no WCB form")
	}
}

func rawJSONToWCB(t *testing.T, raw json.RawMessage) wcb.Value {
	t.Helper()
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		t.Fatalf("extension value: %v", err)
	}
	converted, err := jsonValueToWCB(value)
	if err != nil {
		t.Fatalf("extension value: %v", err)
	}
	return converted
}

func provenanceFromVector(t *testing.T, vector kvProvProvenanceVector) modes.KvProvenanceInput {
	t.Helper()
	modelRoot, err := hash.FromHex(vector.ModelRoot)
	if err != nil {
		t.Fatalf("model_root: %v", err)
	}
	salt, err := hash.FromHex(vector.Salt)
	if err != nil {
		t.Fatalf("salt: %v", err)
	}
	positional, err := modes.ParseKvPositional(vector.Positional)
	if err != nil {
		t.Fatalf("positional: %v", err)
	}
	dtype, err := modes.ParseKvDtype(vector.Layout.Dtype)
	if err != nil {
		t.Fatalf("layout dtype: %v", err)
	}
	input := modes.KvProvenanceInput{
		ModelRoot:          modelRoot,
		TokenizerScheme:    vector.Tokenizer.Scheme,
		TokenizerVocabSize: vector.Tokenizer.VocabSize,
		Attention: modes.KvAttention{
			HeadCount:     vector.Attention.HeadCount,
			HeadDim:       vector.Attention.HeadDim,
			SlidingWindow: vector.Attention.SlidingWindow,
			RopeThetaBits: vector.Attention.RopeThetaBits,
			RopeScaleBits: vector.Attention.RopeScaleBits,
		},
		Positional: positional,
		Layout: modes.KvLayout{
			Dtype:      dtype,
			PageRows:   vector.Layout.PageRows,
			LayerCount: vector.Layout.LayerCount,
		},
		Salt: salt,
	}
	for _, text := range vector.AdapterRoots {
		root, err := hash.FromHex(text)
		if err != nil {
			t.Fatalf("adapter root: %v", err)
		}
		input = input.WithAdapter(root)
	}
	for _, extension := range vector.Extensions {
		input = input.WithExtension(extension.Name, rawJSONToWCB(t, extension.Value))
	}
	return input
}

func TestKvProvVectors(t *testing.T) {
	files := jsonFiles(t, filepath.Join("v1", "kv-prov"))
	if len(files) != 21 {
		t.Fatalf("expected 21 KV_CACHE_PROV vectors, found %d", len(files))
	}
	lim := limits.Default()
	validVectors := 0
	validPages := 0
	rejects := 0
	for _, path := range files {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			var vector kvProvVector
			loadJSON(t, path, &vector)
			if vector.SupersededBy != "" {
				t.Skipf("superseded by %s", vector.SupersededBy)
			}
			if vector.ExpectError != "" {
				frameBytes := mustHex(t, "frame_hex", vector.FrameHex)
				parsed, err := frame.Parse(frameBytes, lim)
				if err != nil {
					if class := string(werrors.ClassOf(err)); class != vector.ExpectError {
						t.Fatalf("expected class %s, got %s (%v)", vector.ExpectError, class, err)
					}
					rejects++
					return
				}
				if _, err := modes.DecodeKvProvPage(parsed, lim); err == nil {
					t.Fatalf("expected %s, decode succeeded", vector.ExpectError)
				} else if class := string(werrors.ClassOf(err)); class != vector.ExpectError {
					t.Fatalf("expected class %s, got %s (%v)", vector.ExpectError, class, err)
				}
				rejects++
				return
			}
			if vector.Schema != modes.KvProvSchema {
				t.Fatalf("unexpected schema %q", vector.Schema)
			}
			if vector.Provenance == nil {
				t.Fatalf("valid vector has no provenance")
			}
			if vector.Expect == nil || !vector.Expect.OK {
				t.Fatalf("valid vector is not marked ok")
			}
			provenance := provenanceFromVector(t, *vector.Provenance)
			dtype, err := modes.ParseKvDtype(vector.Dtype)
			if err != nil {
				t.Fatalf("dtype: %v", err)
			}
			for _, page := range vector.Pages {
				frameBytes := mustHex(t, "frame_hex", page.FrameHex)
				elements := mustHex(t, "elements_hex", page.ElementsHex)
				input := modes.KvCacheInput{
					Model: vector.Model, Layer: vector.Layer, Session: vector.Session,
					PageIndex: page.PageIndex, PageCount: vector.PageCount,
					SeqStart: page.SeqStart, SeqEnd: page.SeqEnd, Dtype: dtype,
					HeadCount: vector.HeadCount, HeadDim: vector.HeadDim,
				}
				reencoded, err := modes.EncodeKvProvPage(input, elements, provenance, lim)
				if err != nil {
					t.Fatalf("re-encode: %v", err)
				}
				if !bytes.Equal(reencoded, frameBytes) {
					t.Fatalf("Go re-encode is not byte-identical to the committed Rust frame")
				}
				parsed, err := frame.Parse(frameBytes, lim)
				if err != nil {
					t.Fatalf("parse: %v", err)
				}
				decoded, err := modes.DecodeKvProvPage(parsed, lim)
				if err != nil {
					t.Fatalf("decode: %v", err)
				}
				if !bytes.Equal(decoded.Bytes, elements) {
					t.Fatalf("page %d bytes differ", page.PageIndex)
				}
				if decoded.Root.Hex() != page.DescriptorRoot {
					t.Fatalf("page %d root %s != %s", page.PageIndex, decoded.Root.Hex(), page.DescriptorRoot)
				}
				if decoded.Descriptor.Page.Shape[0] != page.PageRows ||
					decoded.Descriptor.Page.PageBytes != page.PageBytes ||
					decoded.Descriptor.Page.PayloadLen != page.PayloadLen {
					t.Fatalf("page %d descriptor differs", page.PageIndex)
				}
				if validPages == 0 && page.DescriptorRoot != vector.DescriptorRoot {
					t.Fatalf("vector descriptor_root disagrees with the first page")
				}
				validPages++
			}
			validVectors++
		})
	}
	if validVectors != 8 {
		t.Fatalf("expected 8 valid vectors, saw %d", validVectors)
	}
	if validPages != 9 {
		t.Fatalf("expected 9 valid pages, saw %d", validPages)
	}
	if rejects != 12 {
		t.Fatalf("expected 12 rejects, saw %d", rejects)
	}
}

// ---------------------------------------------------------------------------
// Adversarial court (codec level)
// ---------------------------------------------------------------------------

func hashRoot(byteValue byte) hash.Hash32 {
	var out [hash.HashLen]byte
	for index := range out {
		out[index] = byteValue
	}
	return hash.FromBytes(out)
}

func provFixture(adapters []byte, salt byte) modes.KvProvenanceInput {
	input := modes.KvProvenanceInput{
		ModelRoot:          hashRoot(0x11),
		TokenizerScheme:    "a-bpe",
		TokenizerVocabSize: 151936,
		Attention: modes.KvAttention{
			HeadCount: 2, HeadDim: 4,
			RopeThetaBits: 0x40C3880000000000, // 10_000.0
			RopeScaleBits: 0x3FF0000000000000, // 1.0
		},
		Positional: modes.KvPositionalRope,
		Layout: modes.KvLayout{
			Dtype:      modes.KvDtype{Name: "bf16", WordBytes: 2, Planes: 2},
			PageRows:   4,
			LayerCount: 8,
		},
		Salt: hashRoot(salt),
	}
	for _, adapter := range adapters {
		input = input.WithAdapter(hashRoot(adapter))
	}
	return input
}

func provInput() modes.KvCacheInput {
	return modes.KvCacheInput{
		Model: "prov-model", Layer: 0, Session: "prov-session",
		PageIndex: 0, PageCount: 4, SeqStart: 0, SeqEnd: 4,
		Dtype:     modes.KvDtype{Name: "bf16", WordBytes: 2, Planes: 2},
		HeadCount: 2, HeadDim: 4,
	}
}

func provElements() []byte {
	out := make([]byte, 64)
	for index := range out {
		out[index] = byte(index*37) ^ 0x5A
	}
	return out
}

func TestKvProvCollisionAndFailClosed(t *testing.T) {
	lim := limits.Default()
	data := provElements()
	frameA, err := modes.EncodeKvProvPage(provInput(), data, provFixture([]byte{0x21}, 0xA0), lim)
	if err != nil {
		t.Fatalf("frame A: %v", err)
	}
	frameB, err := modes.EncodeKvProvPage(provInput(), data, provFixture([]byte{0x22}, 0xA0), lim)
	if err != nil {
		t.Fatalf("frame B: %v", err)
	}
	a := decodeProv(t, frameA, lim)
	b := decodeProv(t, frameB, lim)
	if !bytes.Equal(a.Bytes, b.Bytes) {
		t.Fatalf("payloads must be byte-identical")
	}
	if a.Descriptor.PayloadHash != b.Descriptor.PayloadHash {
		t.Fatalf("payload hashes must be identical")
	}
	if a.Root == b.Root {
		t.Fatalf("different adapter roots produced the same descriptor root")
	}
	// A scope that names A's root must refuse B, emitting zero bytes.
	if _, emitted, err := resolveScoped(frameB, a.Root, lim); err == nil || emitted != 0 {
		t.Fatalf("cross-scope resolution must fail closed (emitted=%d)", emitted)
	}
	if _, emitted, err := resolveScoped(frameA, a.Root, lim); err != nil || emitted != len(data) {
		t.Fatalf("matching scope must decode (%v, emitted=%d)", err, emitted)
	}
}

func TestKvProvSaltUnlinkability(t *testing.T) {
	lim := limits.Default()
	data := provElements()
	frameA, err := modes.EncodeKvProvPage(provInput(), data, provFixture(nil, 0xB0), lim)
	if err != nil {
		t.Fatalf("frame A: %v", err)
	}
	frameB, err := modes.EncodeKvProvPage(provInput(), data, provFixture(nil, 0xB1), lim)
	if err != nil {
		t.Fatalf("frame B: %v", err)
	}
	a := decodeProv(t, frameA, lim)
	b := decodeProv(t, frameB, lim)
	if a.Root == b.Root {
		t.Fatalf("different salts produced the same root")
	}
	if a.Descriptor.PayloadHash != b.Descriptor.PayloadHash {
		t.Fatalf("salt must not change the payload hash")
	}
}

func TestKvProvAdapterOrdering(t *testing.T) {
	lim := limits.Default()
	data := provElements()
	ab, err := modes.EncodeKvProvPage(provInput(), data, provFixture([]byte{0x21, 0x22}, 0xA0), lim)
	if err != nil {
		t.Fatalf("ab: %v", err)
	}
	ba, err := modes.EncodeKvProvPage(provInput(), data, provFixture([]byte{0x22, 0x21}, 0xA0), lim)
	if err != nil {
		t.Fatalf("ba: %v", err)
	}
	onlyA, err := modes.EncodeKvProvPage(provInput(), data, provFixture([]byte{0x21}, 0xA0), lim)
	if err != nil {
		t.Fatalf("a: %v", err)
	}
	if !bytes.Equal(ab, ba) {
		t.Fatalf("{A,B} and {B,A} must be one identity")
	}
	if bytes.Equal(ab, onlyA) {
		t.Fatalf("{A} and {A,B} must differ")
	}
}

func TestKvProvForgeryIsRejected(t *testing.T) {
	lim := limits.Default()
	data := provElements()
	frameBytes, err := modes.EncodeKvProvPage(provInput(), data, provFixture([]byte{0x21}, 0xA0), lim)
	if err != nil {
		t.Fatalf("frame: %v", err)
	}
	parsed, err := frame.Parse(frameBytes, lim)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	decoded, err := modes.DecodeKvProvPage(parsed, lim)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}

	// A recomputed descriptor with a tampered dimension root is rejected.
	forged := decoded.Descriptor
	forged.Provenance.AdapterSetRoot = hashRoot(0x5A)
	encoded, err := forged.CanonicalBytes(lim)
	if err != nil {
		t.Fatalf("canonical: %v", err)
	}
	if _, err := modes.ParseKvProvDescriptor(encoded, lim); err == nil ||
		werrors.ClassOf(err) != werrors.Malformed {
		t.Fatalf("tampered adapter_set_root must be Malformed, got %v", err)
	}

	// A swapped payload under a valid descriptor is rejected by payload_hash.
	other := append([]byte(nil), data...)
	other[0] ^= 0xFF
	otherFrame, err := modes.EncodeKvProvPage(provInput(), other, provFixture([]byte{0x21}, 0xA0), lim)
	if err != nil {
		t.Fatalf("other frame: %v", err)
	}
	otherParsed, err := frame.Parse(otherFrame, lim)
	if err != nil {
		t.Fatalf("parse other: %v", err)
	}
	descriptor, _, err := splitBodyForTest(parsed.Body)
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	_, otherPayload, err := splitBodyForTest(otherParsed.Body)
	if err != nil {
		t.Fatalf("split other: %v", err)
	}
	swapped := make([]byte, 0, 4+len(descriptor)+len(otherPayload))
	length := uint32(len(descriptor))
	swapped = append(swapped, byte(length), byte(length>>8), byte(length>>16), byte(length>>24))
	swapped = append(swapped, descriptor...)
	swapped = append(swapped, otherPayload...)
	swappedFrame, err := frame.Encode(frame.Parts{
		FormatMajor: frame.FormatMajor, FormatMinor: frame.FormatMinor,
		Domain: frame.DomainPayload, Mode: frame.ModeKvCacheProv,
		DecodedLen: parsed.Header.DecodedLen, ContentHash: parsed.Header.ContentHash,
		Body: swapped,
	}, lim)
	if err != nil {
		t.Fatalf("encode swapped: %v", err)
	}
	swappedParsed, err := frame.Parse(swappedFrame, lim)
	if err != nil {
		t.Fatalf("parse swapped: %v", err)
	}
	if _, err := modes.DecodeKvProvPage(swappedParsed, lim); err == nil ||
		werrors.ClassOf(err) != werrors.Corrupt {
		t.Fatalf("swapped payload must be Corrupt, got %v", err)
	}
}

func splitBodyForTest(body []byte) ([]byte, []byte, error) {
	if len(body) < 4 {
		return nil, nil, werrors.NewMalformed("body too short")
	}
	length := uint64(body[0]) | uint64(body[1])<<8 | uint64(body[2])<<16 | uint64(body[3])<<24
	end := 4 + length
	if end > uint64(len(body)) {
		return nil, nil, werrors.NewMalformed("descriptor past body")
	}
	return body[4:end], body[end:], nil
}

func decodeProv(t *testing.T, frameBytes []byte, lim limits.Limits) modes.DecodedKvProvPage {
	t.Helper()
	parsed, err := frame.Parse(frameBytes, lim)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	decoded, err := modes.DecodeKvProvPage(parsed, lim)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	return decoded
}

// resolveScoped models a resolver that is asked for a page under a scope that
// names one descriptor root. It emits no bytes unless the root matches.
func resolveScoped(frameBytes []byte, scope hash.Hash32, lim limits.Limits) ([]byte, int, error) {
	parsed, err := frame.Parse(frameBytes, lim)
	if err != nil {
		return nil, 0, err
	}
	decoded, err := modes.DecodeKvProvPage(parsed, lim)
	if err != nil {
		return nil, 0, err
	}
	if decoded.Root != scope {
		return nil, 0, werrors.NewCorrupt("scope does not name this descriptor")
	}
	return decoded.Bytes, len(decoded.Bytes), nil
}
