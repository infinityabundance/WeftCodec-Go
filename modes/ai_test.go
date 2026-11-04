package modes_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func patternBytes(n int, state uint64) []byte {
	out := make([]byte, 0, n)
	for len(out) < n {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		out = append(out, byte(state))
	}
	return out
}

func tensorMeta(dtype string, shape []uint64, count, payloadBytes uint64, payloadHash []byte, layout, endian string) wcb.Value {
	shapeItems := make([]wcb.Value, 0, len(shape))
	for _, d := range shape {
		shapeItems = append(shapeItems, wcb.U64Value(d))
	}
	return wcb.MapValue(map[string]wcb.Value{
		"schema":        wcb.TextValue("weft.tensor.v1"),
		"dtype":         wcb.TextValue(dtype),
		"shape":         wcb.ArrayValue(shapeItems),
		"layout":        wcb.TextValue(layout),
		"endianness":    wcb.TextValue(endian),
		"element_count": wcb.U64Value(count),
		"payload_bytes": wcb.U64Value(payloadBytes),
		"payload_hash":  wcb.BytesValue(payloadHash),
	})
}

func tokenMeta(encoding string, count, payloadBytes uint64, payloadHash []byte) wcb.Value {
	return wcb.MapValue(map[string]wcb.Value{
		"schema":        wcb.TextValue("weft.tokens.v1"),
		"vocab_schema":  wcb.TextValue("weft.test.vocab.v1"),
		"encoding":      wcb.TextValue(encoding),
		"token_count":   wcb.U64Value(count),
		"payload_bytes": wcb.U64Value(payloadBytes),
		"payload_hash":  wcb.BytesValue(payloadHash),
	})
}

func rawAIFrame(t *testing.T, mode frame.Mode, meta wcb.Value, payload []byte, lim limits.Limits) []byte {
	t.Helper()
	metaBytes, err := wcb.Encode(meta, lim)
	if err != nil {
		t.Fatalf("metadata: %v", err)
	}
	body := make([]byte, 0, 4+len(metaBytes)+len(payload))
	var prefix [4]byte
	binary.LittleEndian.PutUint32(prefix[:], uint32(len(metaBytes)))
	body = append(body, prefix[:]...)
	body = append(body, metaBytes...)
	body = append(body, payload...)
	parts := frame.PayloadParts(mode, 0, 0, uint64(len(payload)),
		hash.Domain(hash.DomainChunk, payload), body)
	encoded, err := frame.Encode(parts, lim)
	if err != nil {
		t.Fatalf("frame: %v", err)
	}
	return encoded
}

func classOf(err error) string {
	if err == nil {
		return ""
	}
	return string(werrors.ClassOf(err))
}

func TestTensorRoundTrip(t *testing.T) {
	lim := limits.Default()
	cases := []struct {
		dtype string
		shape []uint64
	}{
		{"f16", []uint64{}},
		{"bf16", []uint64{7}},
		{"f32", []uint64{2, 3}},
		{"f64", []uint64{2, 3, 4, 5}},
		{"i8", []uint64{1, 255}},
		{"u8", []uint64{0, 4}},
		{"i16", []uint64{9, 9}},
		{"i32", []uint64{4, 4, 4}},
		{"i64", []uint64{3, 3, 3, 3, 3}},
	}
	sizes := map[string]uint64{"f16": 2, "bf16": 2, "i16": 2, "f32": 4, "i32": 4, "f64": 8, "i64": 8, "i8": 1, "u8": 1}
	for _, tc := range cases {
		product := uint64(1)
		for _, d := range tc.shape {
			product *= d
		}
		n := product * sizes[tc.dtype]
		if len(tc.shape) == 0 {
			n = sizes[tc.dtype]
		}
		payload := patternBytes(int(n), 0x1234_5678_9abc_def0^n)
		encoded, err := modes.EncodeTensor(payload, tc.dtype, tc.shape, lim)
		if err != nil {
			t.Fatalf("%s %v encode: %v", tc.dtype, tc.shape, err)
		}
		decoded, err := modes.DecodeBytes(context.Background(), encoded, nil, lim)
		if err != nil {
			t.Fatalf("%s %v decode: %v", tc.dtype, tc.shape, err)
		}
		if !bytes.Equal(decoded, payload) {
			t.Fatalf("%s %v payload mismatch", tc.dtype, tc.shape)
		}
		again, err := modes.EncodeTensor(decoded, tc.dtype, tc.shape, lim)
		if err != nil {
			t.Fatalf("re-encode: %v", err)
		}
		if !bytes.Equal(again, encoded) {
			t.Fatalf("%s %v frame is not canonical", tc.dtype, tc.shape)
		}
		f, err := frame.Parse(encoded, lim)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		meta, err := modes.TensorMetadata(f, lim)
		if err != nil {
			t.Fatalf("metadata: %v", err)
		}
		if meta.PayloadBytes != uint64(len(payload)) {
			t.Fatalf("payload bytes %d", meta.PayloadBytes)
		}
	}
}

func TestTensorZeroElement(t *testing.T) {
	lim := limits.Default()
	for _, shape := range [][]uint64{{0}, {0, 0}, {2, 0, 3}} {
		encoded, err := modes.EncodeTensor(nil, "f32", shape, lim)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		decoded, err := modes.DecodeBytes(context.Background(), encoded, nil, lim)
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		if len(decoded) != 0 {
			t.Fatalf("expected empty payload")
		}
	}
}

func TestTensorValidation(t *testing.T) {
	lim := limits.Default()
	if _, err := modes.EncodeTensor(make([]byte, 23), "f32", []uint64{2, 3}, lim); classOf(err) != "Malformed" {
		t.Fatalf("payload mismatch: %v", err)
	}
	nine := []uint64{1, 1, 1, 1, 1, 1, 1, 1, 1}
	if _, err := modes.EncodeTensor([]byte{0}, "u8", nine, lim); classOf(err) != "LimitExceeded" {
		t.Fatalf("rank: %v", err)
	}
	if _, err := modes.EncodeTensor(nil, "u8", []uint64{^uint64(0), 2}, lim); classOf(err) != "LimitExceeded" {
		t.Fatalf("overflow: %v", err)
	}
	lower := limits.Default()
	lower.MaxTensorElements = 5
	if _, err := modes.EncodeTensor(make([]byte, 6), "u8", []uint64{2, 3}, lower); classOf(err) != "LimitExceeded" {
		t.Fatalf("element limit: %v", err)
	}
}

func TestTensorDecodeRejectsBadMetadata(t *testing.T) {
	lim := limits.Default()
	payload := []byte{1, 2, 3, 4, 5, 6}
	h := hash.Domain(hash.DomainTensorPayload, payload)
	cases := []struct {
		name string
		meta wcb.Value
		want string
	}{
		{"shape", tensorMeta("u8", []uint64{2, 3}, 5, 6, h.Bytes(), "row_major", "little"), "Malformed"},
		{"bytes", tensorMeta("u8", []uint64{2, 3}, 6, 7, h.Bytes(), "row_major", "little"), "Malformed"},
		{"hash", tensorMeta("u8", []uint64{2, 3}, 6, 6, make([]byte, 32), "row_major", "little"), "Corrupt"},
		{"layout", tensorMeta("u8", []uint64{2, 3}, 6, 6, h.Bytes(), "column_major", "little"), "Malformed"},
		{"endian", tensorMeta("u8", []uint64{2, 3}, 6, 6, h.Bytes(), "row_major", "big"), "Malformed"},
		{"dtype", tensorMeta("q4", []uint64{6}, 6, 6, h.Bytes(), "row_major", "little"), "Malformed"},
	}
	for _, tc := range cases {
		encoded := rawAIFrame(t, frame.ModeTensor, tc.meta, payload, lim)
		_, err := modes.DecodeBytes(context.Background(), encoded, nil, lim)
		if classOf(err) != tc.want {
			t.Fatalf("%s: got %v want %s", tc.name, err, tc.want)
		}
	}
}

func TestTokenRoundTrip(t *testing.T) {
	lim := limits.Default()
	cases := [][]uint64{
		{},
		{0},
		{1},
		{^uint64(0)},
		{0, 1, 2, 3, 4},
		{4, 3, 2, 1, 0},
		{0, ^uint64(0), 0, ^uint64(0) - 1, 42, 42, 41},
	}
	for i := uint64(0); i < 4096; i++ {
		cases[len(cases)-1] = append(cases[len(cases)-1], (i*7919)%100000)
	}
	for _, ids := range cases {
		encoded, err := modes.EncodeTokens(ids, "weft.test.vocab.v1", lim)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		payload, err := modes.EncodeTokenPayload(ids, lim)
		if err != nil {
			t.Fatalf("payload: %v", err)
		}
		decoded, err := modes.DecodeBytes(context.Background(), encoded, nil, lim)
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		if !bytes.Equal(decoded, payload) {
			t.Fatalf("payload mismatch")
		}
		again, err := modes.EncodeTokens(ids, "weft.test.vocab.v1", lim)
		if err != nil {
			t.Fatalf("re-encode: %v", err)
		}
		if !bytes.Equal(again, encoded) {
			t.Fatalf("frame is not canonical")
		}
		roundTrip, err := modes.DecodeTokenPayload(payload, uint64(len(ids)), lim)
		if err != nil {
			t.Fatalf("payload decode: %v", err)
		}
		if len(roundTrip) != len(ids) {
			t.Fatalf("count %d", len(roundTrip))
		}
		for j := range ids {
			if roundTrip[j] != ids[j] {
				t.Fatalf("id %d: %d != %d", j, roundTrip[j], ids[j])
			}
		}
	}
}

func TestTokenDecodeRejectsMalformed(t *testing.T) {
	lim := limits.Default()
	ids := []uint64{1, 2, 3}
	payload, err := modes.EncodeTokenPayload(ids, lim)
	if err != nil {
		t.Fatal(err)
	}
	h := hash.Domain(hash.DomainTokensPayload, payload)
	cases := []struct {
		name    string
		meta    wcb.Value
		payload []byte
		want    string
	}{
		{"count", tokenMeta(modes.TokenEncoding, 2, uint64(len(payload)), h.Bytes()), payload, "Malformed"},
		{"encoding", tokenMeta("u32-fixed-little", 3, uint64(len(payload)), h.Bytes()), payload, "Malformed"},
		{"nonminimal", tokenMeta(modes.TokenEncoding, 1, 2, hash.Domain(hash.DomainTokensPayload, []byte{0x81, 0x00}).Bytes()), []byte{0x81, 0x00}, "NonCanonical"},
		{"trailing", tokenMeta(modes.TokenEncoding, 1, 2, hash.Domain(hash.DomainTokensPayload, []byte{0x01, 0x02}).Bytes()), []byte{0x01, 0x02}, "Malformed"},
		{"hash", tokenMeta(modes.TokenEncoding, 3, uint64(len(payload)), make([]byte, 32)), payload, "Corrupt"},
	}
	for _, tc := range cases {
		encoded := rawAIFrame(t, frame.ModeTokenSequence, tc.meta, tc.payload, lim)
		_, err := modes.DecodeBytes(context.Background(), encoded, nil, lim)
		if classOf(err) != tc.want {
			t.Fatalf("%s: got %v want %s", tc.name, err, tc.want)
		}
	}
}

func TestTokenLimits(t *testing.T) {
	lim := limits.Default()
	lim.MaxTokens = 2
	ids := []uint64{1, 2, 3}
	if _, err := modes.EncodeTokens(ids, "v", lim); classOf(err) != "LimitExceeded" {
		t.Fatalf("encode: %v", err)
	}
	base := limits.Default()
	payload, err := modes.EncodeTokenPayload(ids, base)
	if err != nil {
		t.Fatal(err)
	}
	h := hash.Domain(hash.DomainTokensPayload, payload)
	meta := tokenMeta(modes.TokenEncoding, 3, uint64(len(payload)), h.Bytes())
	encoded := rawAIFrame(t, frame.ModeTokenSequence, meta, payload, lim)
	if _, err := modes.DecodeBytes(context.Background(), encoded, nil, lim); classOf(err) != "LimitExceeded" {
		t.Fatalf("decode: %v", err)
	}
}

func TestAIDomainBinding(t *testing.T) {
	lim := limits.Default()
	payload := []byte{1, 2, 3}
	tensorHash := hash.Domain(hash.DomainTensorPayload, payload)
	tokenHash := hash.Domain(hash.DomainTokensPayload, payload)
	if tensorHash == tokenHash {
		t.Fatal("domains must differ")
	}
	meta := tensorMeta("u8", []uint64{3}, 3, 3, tokenHash.Bytes(), "row_major", "little")
	encoded := rawAIFrame(t, frame.ModeTensor, meta, payload, lim)
	if _, err := modes.DecodeBytes(context.Background(), encoded, nil, lim); classOf(err) != "Corrupt" {
		t.Fatalf("wrong domain: %v", err)
	}
	meta = tensorMeta("u8", []uint64{3}, 3, 3, tensorHash.Bytes(), "row_major", "little")
	encoded = rawAIFrame(t, frame.ModeTensor, meta, payload, lim)
	decoded, err := modes.DecodeBytes(context.Background(), encoded, nil, lim)
	if err != nil {
		t.Fatalf("right domain: %v", err)
	}
	if !bytes.Equal(decoded, payload) {
		t.Fatalf("payload")
	}
}
