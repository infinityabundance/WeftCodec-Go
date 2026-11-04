package modes_test

import (
	"bytes"
	"context"
	"testing"

	"github.com/klauspost/compress/zstd"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
)

func bf16Words(words []uint16) []byte {
	out := make([]byte, 0, len(words)*2)
	for _, word := range words {
		out = append(out, byte(word), byte(word>>8))
	}
	return out
}

func xorshift(state *uint64) uint64 {
	*state ^= *state << 13
	*state ^= *state >> 7
	*state ^= *state << 17
	return *state
}

func TestSemanticNearDuplicateSelectsTheRightBase(t *testing.T) {
	lim := limits.Default()
	const elementCount = 500_000 // 1 MB of bf16; 0.1% = 500 elements
	shape := []uint64{elementCount}
	state := uint64(0x5EED1234ABCD9876)
	base := make([]byte, 0, elementCount*2)
	for i := 0; i < elementCount; i++ {
		word := uint16(0x3C00) | uint16(xorshift(&state)&0x03FF)
		base = append(base, byte(word), byte(word>>8))
	}
	target := append([]byte(nil), base...)
	changed := 0
	for i := 0; i < elementCount; i += 1000 {
		target[i*2] ^= 0x05
		changed++
	}
	if changed != 500 {
		t.Fatalf("changed %d, want 500", changed)
	}
	policy := modes.MantissaTruncatePolicy(3)
	dtype, _ := modes.ParseWeightDtype("bf16")
	targetRoot, err := modes.SemanticRoot(dtype, shape, target, policy)
	if err != nil {
		t.Fatalf("target root: %v", err)
	}
	baseRoot, err := modes.SemanticRoot(dtype, shape, base, policy)
	if err != nil {
		t.Fatalf("base root: %v", err)
	}
	if targetRoot != baseRoot {
		t.Fatalf("0.1%% low-bit changes must share the semantic root")
	}
	baseFrame, err := modes.EncodeTensor(base, "bf16", shape, lim)
	if err != nil {
		t.Fatalf("base frame: %v", err)
	}
	other := make([]byte, 0, elementCount*2)
	for i := 0; i < elementCount; i++ {
		word := uint16(0x4000) ^ uint16(i<<4) ^ uint16(i&0x0F)
		other = append(other, byte(word), byte(word>>8))
	}
	otherFrame, err := modes.EncodeTensor(other, "bf16", shape, lim)
	if err != nil {
		t.Fatalf("other frame: %v", err)
	}
	index, ok, err := modes.SelectSemanticBase(
		target, dtype, shape, policy, [][]byte{baseFrame, otherFrame}, lim)
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	if !ok || index != 0 {
		t.Fatalf("selected (%d,%v), want (0,true)", index, ok)
	}
	noneIndex, noneOK, err := modes.SelectSemanticBase(
		target, dtype, shape, policy, [][]byte{otherFrame}, lim)
	if err != nil {
		t.Fatalf("select other: %v", err)
	}
	if noneOK {
		t.Fatalf("different semantic root was offered as candidate index %d", noneIndex)
	}
	otherRecord, err := modes.NewSemanticRecord(dtype, shape, other, policy, lim)
	if err != nil {
		t.Fatalf("other record: %v", err)
	}
	targetRecord, err := modes.NewSemanticRecord(dtype, shape, target, policy, lim)
	if err != nil {
		t.Fatalf("target record: %v", err)
	}
	if modes.SemanticMatches(targetRecord, otherRecord) {
		t.Fatalf("different roots must not match")
	}

	delta, err := modes.EncodeSemanticDelta(
		context.Background(), target, dtype, shape, baseFrame, policy, lim)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	parsed, err := frame.Parse(delta, lim)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	decoded, err := modes.DecodeSemanticDeltaFrameWithBase(
		context.Background(), parsed, baseFrame, dependency.NewMapResolver(), lim)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(decoded, target) {
		t.Fatalf("reconstruction differs")
	}
	originalRecord, err := tensor.NewLogicalRecord("bf16", shape, target, lim)
	if err != nil {
		t.Fatalf("original record: %v", err)
	}
	originalRoot, err := originalRecord.Root(lim)
	if err != nil {
		t.Fatalf("original root: %v", err)
	}
	decodedRecord, err := tensor.NewLogicalRecord("bf16", shape, decoded, lim)
	if err != nil {
		t.Fatalf("decoded record: %v", err)
	}
	decodedRoot, err := decodedRecord.Root(lim)
	if err != nil {
		t.Fatalf("decoded root: %v", err)
	}
	if originalRoot != decodedRoot {
		t.Fatalf("logical roots differ")
	}
	encoder, err := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedDefault))
	if err != nil {
		t.Fatalf("zstd: %v", err)
	}
	zstdBytes := len(encoder.EncodeAll(target, nil))
	encoder.Close()
	factor := float64(zstdBytes) / float64(len(delta))
	if len(delta) >= zstdBytes {
		t.Fatalf("delta frame %d is not smaller than zstd %d", len(delta), zstdBytes)
	}
	t.Logf("near-duplicate: changed=%d/%d delta=%d zstd=%d factor=%.2fx",
		changed, elementCount, len(delta), zstdBytes, factor)
}

func TestSemanticRootPolicyValidation(t *testing.T) {
	lim := limits.Default()
	dtype, _ := modes.ParseWeightDtype("bf16")
	bytesIn := bf16Words([]uint16{0x3F80, 0xC000, 0x0000, 0x7FC0, 0x8000, 0x0001})
	if _, err := modes.NewSemanticRecord(
		dtype, []uint64{2, 3}, bytesIn, modes.MantissaTruncatePolicy(8), lim); err == nil {
		t.Fatalf("bf16 bits=8 should be rejected")
	}
	if _, err := modes.NewSemanticRecord(
		dtype, []uint64{2, 3}, bytesIn, modes.BlockQuantizePolicy(4, 16), lim); err == nil {
		t.Fatalf("block_size 4 must not divide 6")
	}
	if _, err := modes.NewSemanticRecord(
		dtype, []uint64{2, 3}, bytesIn, modes.BlockQuantizePolicy(3, 0), lim); err == nil {
		t.Fatalf("levels 0 should be rejected")
	}
	none, err := modes.NewSemanticRecord(dtype, []uint64{2, 3}, bytesIn, modes.NonePolicy(), lim)
	if err != nil {
		t.Fatalf("none: %v", err)
	}
	if none.QuantizedValuesHash != tensor.ValuesHash(bytesIn) {
		t.Fatalf("none policy must use the exact values hash")
	}
}
