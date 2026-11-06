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
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func sparseBaseFrame(t *testing.T, dtype string, shape []uint64, bytes []byte) []byte {
	t.Helper()
	encoded, err := modes.EncodeTensor(bytes, dtype, shape, limits.Default())
	if err != nil {
		t.Fatalf("base frame: %v", err)
	}
	return encoded
}

func sparseEncode(t *testing.T, target []byte, dtype modes.WeightDtype, shape []uint64, base []byte) []byte {
	t.Helper()
	encoded, err := modes.EncodeSparseDelta(
		context.Background(), target, dtype, shape, base, limits.Default())
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	return encoded
}

func sparseDecode(t *testing.T, delta, base []byte) ([]byte, error) {
	t.Helper()
	parsed, err := frame.Parse(delta, limits.Default())
	if err != nil {
		return nil, err
	}
	if base == nil {
		return modes.DecodeSparseDeltaFrame(
			context.Background(), parsed, dependency.NewMapResolver(), limits.Default())
	}
	return modes.DecodeSparseDeltaFrameWithBase(
		context.Background(), parsed, base, dependency.NewMapResolver(), limits.Default())
}

func sparseRunCount(t *testing.T, encoded []byte) uint32 {
	t.Helper()
	parsed, err := frame.Parse(encoded, limits.Default())
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	metaLen := int(binary.LittleEndian.Uint32(parsed.Body[:4]))
	return binary.LittleEndian.Uint32(parsed.Body[4+metaLen:])
}

func TestSparseDeltaBF16RoundTrip(t *testing.T) {
	baseWords := []uint16{0x3F80, 0xC000, 0x0000, 0x8000, 0x7FC0, 0x7F81, 0x0001, 0x7F80}
	targetWords := append([]uint16(nil), baseWords...)
	targetWords[1] = 0x4000
	targetWords[4] = 0x7FC1
	targetWords[6] = 0x8002
	var baseBytes, targetBytes []byte
	for _, word := range baseWords {
		baseBytes = append(baseBytes, byte(word), byte(word>>8))
	}
	for _, word := range targetWords {
		targetBytes = append(targetBytes, byte(word), byte(word>>8))
	}
	dtype, err := modes.ParseWeightDtype("bf16")
	if err != nil {
		t.Fatalf("dtype: %v", err)
	}
	shape := []uint64{2, 4}
	base := sparseBaseFrame(t, "bf16", shape, baseBytes)
	delta := sparseEncode(t, targetBytes, dtype, shape, base)
	decoded, err := sparseDecode(t, delta, base)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(decoded, targetBytes) {
		t.Fatalf("round trip is not byte-exact")
	}
	parsed, err := frame.Parse(delta, limits.Default())
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	meta, err := modes.SparseDeltaMetadata(parsed, limits.Default())
	if err != nil {
		t.Fatalf("metadata: %v", err)
	}
	if meta.Dtype.Name != "bf16" || meta.StorageBytes != 2 ||
		meta.Op != "mod-sub" || meta.SparseCodec != "zero-run-v1" {
		t.Fatalf("descriptor = %+v", meta)
	}
	if meta.Base != parsed.Dependencies[0].Hash {
		t.Fatalf("descriptor base does not equal the dependency hash")
	}
	if got := sparseRunCount(t, delta); got != 3 {
		t.Fatalf("run count = %d, want 3", got)
	}
	again := sparseEncode(t, targetBytes, dtype, shape, base)
	if !bytes.Equal(again, delta) {
		t.Fatalf("encode is not deterministic")
	}
}

func TestSparseDeltaOtherDtypesAndAllZero(t *testing.T) {
	// fp16: one changed word.
	var fp16Base []byte
	for index := 0; index < 5; index++ {
		fp16Base = append(fp16Base, byte(index*3), byte((index*3)>>8))
	}
	fp16Target := append([]byte(nil), fp16Base...)
	fp16Target[4], fp16Target[5] = 0x01, 0x3C
	shape5 := []uint64{5}
	fp16, _ := modes.ParseWeightDtype("fp16")
	base := sparseBaseFrame(t, "f16", shape5, fp16Base)
	delta := sparseEncode(t, fp16Target, fp16, shape5, base)
	if got := sparseRunCount(t, delta); got != 1 {
		t.Fatalf("fp16 run count = %d, want 1", got)
	}
	decoded, err := sparseDecode(t, delta, base)
	if err != nil || !bytes.Equal(decoded, fp16Target) {
		t.Fatalf("fp16 round trip failed: %v", err)
	}

	// fp32: three changed words.
	var fp32Base []byte
	for index := 0; index < 8; index++ {
		var word [4]byte
		binary.LittleEndian.PutUint32(word[:], uint32(0x3F000000+index*7))
		fp32Base = append(fp32Base, word[:]...)
	}
	fp32Target := append([]byte(nil), fp32Base...)
	for _, index := range []int{0, 3, 7} {
		var word [4]byte
		binary.LittleEndian.PutUint32(word[:], uint32(0x7FC00001+index))
		copy(fp32Target[index*4:], word[:])
	}
	fp32, _ := modes.ParseWeightDtype("fp32")
	shape24 := []uint64{2, 4}
	base = sparseBaseFrame(t, "f32", shape24, fp32Base)
	delta = sparseEncode(t, fp32Target, fp32, shape24, base)
	if got := sparseRunCount(t, delta); got != 3 {
		t.Fatalf("fp32 run count = %d, want 3", got)
	}
	decoded, err = sparseDecode(t, delta, base)
	if err != nil || !bytes.Equal(decoded, fp32Target) {
		t.Fatalf("fp32 round trip failed: %v", err)
	}

	// q8_0: wrapping deltas.
	q8Base := make([]byte, 64)
	for index := range q8Base {
		q8Base[index] = byte((index*37 + 11) % 251)
	}
	q8Target := append([]byte(nil), q8Base...)
	for index := range q8Target {
		if index%11 == 0 {
			q8Target[index] += 7
		}
	}
	q8, _ := modes.ParseWeightDtype("q8_0")
	shape64 := []uint64{8, 8}
	base = sparseBaseFrame(t, "i8", shape64, q8Base)
	delta = sparseEncode(t, q8Target, q8, shape64, base)
	if got := sparseRunCount(t, delta); got != 6 {
		t.Fatalf("q8_0 run count = %d, want 6", got)
	}
	decoded, err = sparseDecode(t, delta, base)
	if err != nil || !bytes.Equal(decoded, q8Target) {
		t.Fatalf("q8_0 round trip failed: %v", err)
	}

	// All-zero delta: run_count 0, payload is exactly the count.
	shape6 := []uint64{2, 3}
	base = sparseBaseFrame(t, "bf16", shape6, fp16Base[:12])
	dtype, _ := modes.ParseWeightDtype("bf16")
	delta = sparseEncode(t, fp16Base[:12], dtype, shape6, base)
	if got := sparseRunCount(t, delta); got != 0 {
		t.Fatalf("all-zero run count = %d, want 0", got)
	}
	parsed, err := frame.Parse(delta, limits.Default())
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	metaLen := int(binary.LittleEndian.Uint32(parsed.Body[:4]))
	if len(parsed.Body)-4-metaLen != 4 {
		t.Fatalf("all-zero payload is not exactly the u32le count")
	}
	if _, err := sparseDecode(t, delta, base); err != nil {
		t.Fatalf("all-zero decode: %v", err)
	}
	meta, err := modes.SparseDeltaMetadata(parsed, limits.Default())
	if err != nil {
		t.Fatalf("metadata: %v", err)
	}
	if meta.DeltaRoot != hash.Domain(tensor.DomainTensorValues, make([]byte, 12)) {
		t.Fatalf("all-zero delta_root is not the zero-word values hash")
	}
}

func TestSparseDeltaDensities(t *testing.T) {
	elementCount := 1 << 20
	shape := []uint64{uint64(elementCount)}
	baseWords := make([]uint32, elementCount)
	state := uint64(0x9E3779B97F4A7C15)
	next := func() uint64 {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		return state
	}
	for index := range baseWords {
		baseWords[index] = uint32(next() & 0xFFFF)
	}
	var base []byte
	for _, word := range baseWords {
		base = append(base, byte(word), byte(word>>8))
	}
	dtype, _ := modes.ParseWeightDtype("bf16")
	baseFrame := sparseBaseFrame(t, "bf16", shape, base)

	for _, densityPPM := range []uint64{0, 10, 10_000, 100_000, 500_000, 1_000_000} {
		targetWords := append([]uint32(nil), baseWords...)
		changed := uint32(0)
		for index := range targetWords {
			if next()%1_000_000 < densityPPM {
				targetWords[index] = (targetWords[index] + (uint32(next()&0xFFFF) | 1)) & 0xFFFF
				changed++
			}
		}
		var target []byte
		for _, word := range targetWords {
			target = append(target, byte(word), byte(word>>8))
		}
		delta := sparseEncode(t, target, dtype, shape, baseFrame)
		if got := sparseRunCount(t, delta); got != changed {
			t.Fatalf("density %d ppm run count = %d, want %d", densityPPM, got, changed)
		}
		decoded, err := sparseDecode(t, delta, baseFrame)
		if err != nil {
			t.Fatalf("density %d ppm decode: %v", densityPPM, err)
		}
		if !bytes.Equal(decoded, target) {
			t.Fatalf("density %d ppm round trip is not byte-exact", densityPPM)
		}
	}
}

func TestSparseDeltaRejects(t *testing.T) {
	baseBytes := make([]byte, 12)
	targetBytes := append([]byte(nil), baseBytes...)
	binary.LittleEndian.PutUint16(targetBytes[0:], 1)
	binary.LittleEndian.PutUint16(targetBytes[10:], 5)
	shape := []uint64{2, 3}
	base := sparseBaseFrame(t, "bf16", shape, baseBytes)
	dtype, _ := modes.ParseWeightDtype("bf16")
	delta := sparseEncode(t, targetBytes, dtype, shape, base)
	parsed, err := frame.Parse(delta, limits.Default())
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	meta, err := modes.SparseDeltaMetadata(parsed, limits.Default())
	if err != nil {
		t.Fatalf("metadata: %v", err)
	}
	metaLen := int(binary.LittleEndian.Uint32(parsed.Body[:4]))
	payload := parsed.Body[4+metaLen:]
	contentHash := parsed.Header.ContentHash

	frameOf := func(meta modes.SparseDeltaMeta, payload []byte) []byte {
		encoded, err := modes.EncodeSparseDeltaDescriptorFrame(
			meta, payload, 12, contentHash, 6, limits.Default())
		if err != nil {
			t.Fatalf("frame: %v", err)
		}
		return encoded
	}
	expectClass := func(what string, encoded []byte, baseFrame []byte, want werrors.Class) {
		t.Helper()
		_, err := sparseDecode(t, encoded, baseFrame)
		if err == nil {
			t.Fatalf("%s: expected %s, decode succeeded", what, want)
		}
		if got := werrors.ClassOf(err); got != want {
			t.Fatalf("%s: class = %s, want %s (%v)", what, got, want, err)
		}
	}

	unknownOp := meta
	unknownOp.Op = "xor"
	expectClass("unknown op", frameOf(unknownOp, payload), base, werrors.Malformed)

	unknownCodec := meta
	unknownCodec.SparseCodec = "rle-v2"
	expectClass("unknown codec", frameOf(unknownCodec, payload), base, werrors.Malformed)

	storage := meta
	storage.StorageBytes = 1
	expectClass("storage bytes", frameOf(storage, payload), base, werrors.Malformed)

	root := meta
	root.DeltaRoot[0] ^= 1
	expectClass("delta root", frameOf(root, payload), base, werrors.Corrupt)

	overCount := []byte{7, 0, 0, 0, 0}
	expectClass("run count", frameOf(meta, overCount), base, werrors.Malformed)

	pastEnd := []byte{1, 0, 0, 0, 6, 1, 0}
	expectClass("run past end", frameOf(meta, pastEnd), base, werrors.Malformed)

	truncated := []byte{1, 0, 0, 0, 0, 1}
	expectClass("truncated record", frameOf(meta, truncated), base, werrors.Malformed)

	trailing := append(append([]byte(nil), payload...), 0)
	expectClass("trailing bytes", frameOf(meta, trailing), base, werrors.Corrupt)

	zeroWord := []byte{1, 0, 0, 0, 0, 0, 0}
	expectClass("zero word", frameOf(meta, zeroWord), base, werrors.Corrupt)

	tooShort := []byte{0, 0}
	expectClass("short payload", frameOf(meta, tooShort), base, werrors.Malformed)

	// Missing base.
	if _, err := sparseDecode(t, delta, nil); err == nil {
		t.Fatalf("missing base: expected DependencyMissing")
	} else if got := werrors.ClassOf(err); got != werrors.DependencyMissing {
		t.Fatalf("missing base: class = %s", got)
	}

	// Base dtype and shape mismatches.
	fp32Base := sparseBaseFrame(t, "f32", shape, make([]byte, 24))
	if _, err := sparseDecode(t, delta, fp32Base); err == nil ||
		werrors.ClassOf(err) != werrors.DependencyMismatch {
		t.Fatalf("base dtype mismatch: %v", err)
	}
	wrongShapeBase := sparseBaseFrame(t, "bf16", []uint64{3, 2}, baseBytes)
	if _, err := sparseDecode(t, delta, wrongShapeBase); err == nil ||
		werrors.ClassOf(err) != werrors.DependencyMismatch {
		t.Fatalf("base shape mismatch: %v", err)
	}

	// The run bound is enforced.
	tight := limits.Default()
	tight.MaxSparseRuns = 1
	if _, err := modes.DecodeSparseDeltaFrameWithBase(
		context.Background(), parsed, base, dependency.NewMapResolver(), tight); err == nil ||
		werrors.ClassOf(err) != werrors.LimitExceeded {
		t.Fatalf("run limit: %v", err)
	}
}
