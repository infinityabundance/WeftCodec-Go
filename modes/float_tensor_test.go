package modes_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func decodeFrame(t *testing.T, frameBytes []byte, lim limits.Limits) ([]byte, error) {
	t.Helper()
	return modes.DecodeBytes(context.Background(), frameBytes, dependency.NewMapResolver(), lim)
}

func floatMeta(
	dtype string,
	shape []uint64,
	count, sign, exponent, mantissa, expLen, mantLen uint64,
	payloadHash []byte,
) wcb.Value {
	shapeItems := make([]wcb.Value, 0, len(shape))
	for _, dim := range shape {
		shapeItems = append(shapeItems, wcb.U64Value(dim))
	}
	return wcb.MapValue(map[string]wcb.Value{
		"schema":             wcb.TextValue(modes.FloatTensorSchema),
		"dtype":              wcb.TextValue(dtype),
		"shape":              wcb.ArrayValue(shapeItems),
		"element_count":      wcb.U64Value(count),
		"sign_bits":          wcb.U64Value(sign),
		"exponent_bits":      wcb.U64Value(exponent),
		"mantissa_bits":      wcb.U64Value(mantissa),
		"exponent_plane_len": wcb.U64Value(expLen),
		"mantissa_plane_len": wcb.U64Value(mantLen),
		"payload_hash":       wcb.BytesValue(payloadHash),
	})
}

func rawFloatFrame(t *testing.T, meta wcb.Value, stored []byte, decodedLen uint64, lim limits.Limits) []byte {
	t.Helper()
	metaBytes, err := wcb.Encode(meta, lim)
	if err != nil {
		t.Fatalf("metadata: %v", err)
	}
	body := make([]byte, 0, 4+len(metaBytes)+len(stored))
	var prefix [4]byte
	binary.LittleEndian.PutUint32(prefix[:], uint32(len(metaBytes)))
	body = append(body, prefix[:]...)
	body = append(body, metaBytes...)
	body = append(body, stored...)
	plaintext := make([]byte, decodedLen)
	frameBytes, err := frame.Encode(frame.PayloadParts(
		frame.ModeFloatTensor, 0, uint16(modes.FloatTensorZstdLevel), decodedLen,
		hash.Domain(hash.DomainChunk, plaintext), body), lim)
	if err != nil {
		t.Fatalf("frame: %v", err)
	}
	return frameBytes
}

func wordsBytes(words []uint32, elementBytes uint64) []byte {
	out := make([]byte, 0, int(elementBytes)*len(words))
	var buf [4]byte
	for _, word := range words {
		binary.LittleEndian.PutUint32(buf[:], word)
		out = append(out, buf[:elementBytes]...)
	}
	return out
}

func TestFloatTensorRoundTripAllDtypes(t *testing.T) {
	lim := limits.Default()
	cases := []struct {
		dtype string
		shape []uint64
	}{
		{"bf16", []uint64{1}},
		{"bf16", []uint64{2, 3}},
		{"fp16", []uint64{3, 1, 2}},
		{"fp32", []uint64{2, 1, 1, 2}},
		{"fp8-e4m3", []uint64{7}},
		{"fp8-e5m2", []uint64{2, 5}},
	}
	for _, tc := range cases {
		dtype, err := modes.ParseFloatDtype(tc.dtype)
		if err != nil {
			t.Fatalf("dtype: %v", err)
		}
		count := uint64(1)
		for _, dim := range tc.shape {
			count *= dim
		}
		payload := patternBytes(int(count*dtype.ElementSize()), 0xfeedface^count)
		frameBytes, err := modes.EncodeFloatTensor(payload, dtype, tc.shape, lim)
		if err != nil {
			t.Fatalf("%s encode: %v", tc.dtype, err)
		}
		decoded, err := decodeFrame(t, frameBytes, lim)
		if err != nil {
			t.Fatalf("%s decode: %v", tc.dtype, err)
		}
		if !bytes.Equal(decoded, payload) {
			t.Fatalf("%s round-trip mismatch", tc.dtype)
		}
		parsed, err := frame.Parse(frameBytes, lim)
		if err != nil {
			t.Fatalf("%s parse: %v", tc.dtype, err)
		}
		if parsed.Header.Mode != frame.ModeFloatTensor || parsed.Header.Mode.String() != "float_tensor" {
			t.Fatalf("%s mode = %v", tc.dtype, parsed.Header.Mode)
		}
		if parsed.Header.DecodedLen != uint64(len(payload)) {
			t.Fatalf("%s decoded_len = %d", tc.dtype, parsed.Header.DecodedLen)
		}
		meta, err := modes.FloatTensorMetadata(parsed, lim)
		if err != nil {
			t.Fatalf("%s metadata: %v", tc.dtype, err)
		}
		if meta.ElementCount != count || meta.Dtype.Name != tc.dtype {
			t.Fatalf("%s metadata mismatch: %+v", tc.dtype, meta)
		}
		again, err := modes.EncodeFloatTensor(decoded, dtype, tc.shape, lim)
		if err != nil {
			t.Fatalf("%s re-encode: %v", tc.dtype, err)
		}
		if !bytes.Equal(again, frameBytes) {
			t.Fatalf("%s re-encode is not byte-identical", tc.dtype)
		}
	}
}

func TestFloatTensorSpecialValues(t *testing.T) {
	lim := limits.Default()
	cases := []struct {
		dtype string
		words []uint32
	}{
		{"bf16", []uint32{0x0000, 0x8000, 0x0001, 0x807f, 0x7f80, 0xff80, 0x7fc1, 0xffef}},
		{"fp16", []uint32{0x0000, 0x8000, 0x0001, 0x03ff, 0x7c00, 0xfc00, 0x7e01, 0xffff}},
		{"fp32", []uint32{0x00000000, 0x80000000, 0x00000001, 0x007fffff,
			0x7f800000, 0xff800000, 0x7f800001, 0xffc00001}},
		{"fp8-e4m3", []uint32{0x00, 0x80, 0x01, 0x07, 0x7f, 0xf0, 0x4f, 0xff}},
		{"fp8-e5m2", []uint32{0x00, 0x80, 0x01, 0x03, 0x7c, 0xfc, 0x7d, 0xff}},
	}
	for _, tc := range cases {
		dtype, err := modes.ParseFloatDtype(tc.dtype)
		if err != nil {
			t.Fatalf("dtype: %v", err)
		}
		payload := wordsBytes(tc.words, dtype.ElementSize())
		frameBytes, err := modes.EncodeFloatTensor(
			payload, dtype, []uint64{uint64(len(tc.words))}, lim)
		if err != nil {
			t.Fatalf("%s encode: %v", tc.dtype, err)
		}
		decoded, err := decodeFrame(t, frameBytes, lim)
		if err != nil {
			t.Fatalf("%s decode: %v", tc.dtype, err)
		}
		if !bytes.Equal(decoded, payload) {
			t.Fatalf("%s special-value round-trip mismatch", tc.dtype)
		}
	}
}

func TestFloatTensorSkewedExponentPlaneCompresses(t *testing.T) {
	lim := limits.Default()
	words := make([]uint32, 64)
	for i := 0; i < 60; i++ {
		words[i] = 0x0000
	}
	copy(words[60:], []uint32{0x7fc1, 0x7fc2, 0x7fc3, 0x7fc4})
	dtype, _ := modes.ParseFloatDtype("bf16")
	payload := wordsBytes(words, dtype.ElementSize())
	frameBytes, err := modes.EncodeFloatTensor(payload, dtype, []uint64{8, 8}, lim)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	parsed, err := frame.Parse(frameBytes, lim)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	meta, err := modes.FloatTensorMetadata(parsed, lim)
	if err != nil {
		t.Fatalf("metadata: %v", err)
	}
	if meta.MantissaPlaneLen != 64 {
		t.Fatalf("mantissa_plane_len = %d, want 64", meta.MantissaPlaneLen)
	}
	if meta.ExponentPlaneLen >= 64 {
		t.Fatalf("exponent_plane_len = %d, want < 64", meta.ExponentPlaneLen)
	}
	decoded, err := decodeFrame(t, frameBytes, lim)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(decoded, payload) {
		t.Fatalf("round-trip mismatch")
	}
}

func TestFloatTensorEncodeRejects(t *testing.T) {
	lim := limits.Default()
	fp32, _ := modes.ParseFloatDtype("fp32")
	if _, err := modes.EncodeFloatTensor(make([]byte, 7), fp32, []uint64{2}, lim); werrors.ClassOf(err) != werrors.Malformed {
		t.Fatalf("short payload class = %v", werrors.ClassOf(err))
	}
	if _, err := modes.EncodeFloatTensor(make([]byte, 4), fp32, []uint64{}, lim); werrors.ClassOf(err) != werrors.Malformed {
		t.Fatalf("rank 0 class = %v", werrors.ClassOf(err))
	}
	if _, err := modes.EncodeFloatTensor(nil, fp32, []uint64{2, 0}, lim); werrors.ClassOf(err) != werrors.Malformed {
		t.Fatalf("zero dimension class = %v", werrors.ClassOf(err))
	}
	nine := []uint64{1, 1, 1, 1, 1, 1, 1, 1, 1}
	if _, err := modes.EncodeFloatTensor([]byte{0}, fp32, nine, lim); werrors.ClassOf(err) != werrors.LimitExceeded {
		t.Fatalf("rank limit class = %v", werrors.ClassOf(err))
	}
	lowered := limits.Default()
	lowered.MaxTensorElements = 3
	fp8, _ := modes.ParseFloatDtype("fp8-e4m3")
	if _, err := modes.EncodeFloatTensor(make([]byte, 4), fp8, []uint64{4}, lowered); werrors.ClassOf(err) != werrors.LimitExceeded {
		t.Fatalf("element limit class = %v", werrors.ClassOf(err))
	}
}

func TestFloatTensorDecodeRejects(t *testing.T) {
	lim := limits.Default()
	fp32, _ := modes.ParseFloatDtype("fp32")
	payload := wordsBytes([]uint32{0x3f800000, 0x7f800001}, fp32.ElementSize())
	expPlane, mantPlane, err := modes.SplitFloatTensorPayload(payload, fp32, 2)
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	stored := append(append([]byte{}, expPlane...), mantPlane...)
	goodHash := hash.Domain(hash.DomainFloatTensorPayload, stored)

	// Schema mismatch.
	meta := floatMeta("fp32", []uint64{2}, 2, 1, 8, 23,
		uint64(len(expPlane)), uint64(len(mantPlane)), goodHash.Bytes())
	meta.Map["schema"] = wcb.TextValue("weft.float-tensor.v1")
	frameBytes := rawFloatFrame(t, meta, stored, 8, lim)
	if _, err := decodeFrame(t, frameBytes, lim); werrors.ClassOf(err) != werrors.Malformed {
		t.Fatalf("schema class = %v", werrors.ClassOf(err))
	}

	// Element count mismatch.
	meta = floatMeta("fp32", []uint64{2}, 3, 1, 8, 23,
		uint64(len(expPlane)), uint64(len(mantPlane)), goodHash.Bytes())
	frameBytes = rawFloatFrame(t, meta, stored, 8, lim)
	if _, err := decodeFrame(t, frameBytes, lim); werrors.ClassOf(err) != werrors.Malformed {
		t.Fatalf("count class = %v", werrors.ClassOf(err))
	}

	// Field widths disagree with the dtype.
	meta = floatMeta("fp16", []uint64{2}, 2, 1, 8, 7,
		uint64(len(expPlane)), uint64(len(mantPlane)), goodHash.Bytes())
	frameBytes = rawFloatFrame(t, meta, stored, 8, lim)
	if _, err := decodeFrame(t, frameBytes, lim); werrors.ClassOf(err) != werrors.Malformed {
		t.Fatalf("widths class = %v", werrors.ClassOf(err))
	}

	// Plane lengths overflow before allocation.
	meta = floatMeta("fp8-e4m3", []uint64{2}, 2, 1, 4, 3, ^uint64(0), 1, make([]byte, 32))
	frameBytes = rawFloatFrame(t, meta, nil, 2, lim)
	if _, err := decodeFrame(t, frameBytes, lim); werrors.ClassOf(err) != werrors.LimitExceeded {
		t.Fatalf("overflow class = %v", werrors.ClassOf(err))
	}

	// Non-zstd exponent plane.
	garbage := append([]byte{0xff, 0xff, 0xff}, mantPlane...)
	garbageHash := hash.Domain(hash.DomainFloatTensorPayload, garbage)
	meta = floatMeta("fp32", []uint64{2}, 2, 1, 8, 23,
		3, uint64(len(mantPlane)), garbageHash.Bytes())
	frameBytes = rawFloatFrame(t, meta, garbage, 8, lim)
	if _, err := decodeFrame(t, frameBytes, lim); werrors.ClassOf(err) != werrors.Corrupt {
		t.Fatalf("zstd class = %v", werrors.ClassOf(err))
	}

	// decoded_len disagrees with the descriptor.
	meta = floatMeta("fp32", []uint64{2}, 2, 1, 8, 23,
		uint64(len(expPlane)), uint64(len(mantPlane)), goodHash.Bytes())
	frameBytes = rawFloatFrame(t, meta, stored, 7, lim)
	if _, err := decodeFrame(t, frameBytes, lim); werrors.ClassOf(err) != werrors.Corrupt {
		t.Fatalf("decoded_len class = %v", werrors.ClassOf(err))
	}

	// Payload hash mismatch.
	meta = floatMeta("fp32", []uint64{2}, 2, 1, 8, 23,
		uint64(len(expPlane)), uint64(len(mantPlane)), make([]byte, 32))
	frameBytes = rawFloatFrame(t, meta, stored, 8, lim)
	if _, err := decodeFrame(t, frameBytes, lim); werrors.ClassOf(err) != werrors.Corrupt {
		t.Fatalf("hash class = %v", werrors.ClassOf(err))
	}
}

func TestFloatTensorMantissaPaddingMustBeZero(t *testing.T) {
	lim := limits.Default()
	fp16, _ := modes.ParseFloatDtype("fp16")
	payload := wordsBytes([]uint32{0x7c00, 0x0001}, fp16.ElementSize())
	expPlane, mantPlane, err := modes.SplitFloatTensorPayload(payload, fp16, 2)
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	if len(mantPlane) != 3 || mantPlane[2]&0b11 != 0 {
		t.Fatalf("unexpected mantissa plane %x", mantPlane)
	}
	mantPlane[2] |= 0b01
	stored := append(append([]byte{}, expPlane...), mantPlane...)
	storedHash := hash.Domain(hash.DomainFloatTensorPayload, stored)
	meta := floatMeta("fp16", []uint64{2}, 2, 1, 5, 10,
		uint64(len(expPlane)), uint64(len(mantPlane)), storedHash.Bytes())
	frameBytes := rawFloatFrame(t, meta, stored, 4, lim)
	if _, err := decodeFrame(t, frameBytes, lim); werrors.ClassOf(err) != werrors.Corrupt {
		t.Fatalf("padding class = %v", werrors.ClassOf(err))
	}
}
