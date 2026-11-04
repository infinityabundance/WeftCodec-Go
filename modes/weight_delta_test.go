package modes_test

import (
	"bytes"
	"context"
	"testing"

	"github.com/klauspost/compress/zstd"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func weightBaseFrame(t *testing.T, dtype, shapeText string, bytes []byte) []byte {
	t.Helper()
	shape, err := parseShape(shapeText)
	if err != nil {
		t.Fatalf("shape: %v", err)
	}
	encoded, err := modes.EncodeTensor(bytes, dtype, shape, limits.Default())
	if err != nil {
		t.Fatalf("base frame: %v", err)
	}
	return encoded
}

func parseShape(text string) ([]uint64, error) {
	shape := []uint64{}
	value := uint64(0)
	digits := 0
	for index := 0; index <= len(text); index++ {
		if index == len(text) || text[index] == ',' {
			if digits == 0 {
				return nil, werrors.NewMalformed("empty shape extent")
			}
			shape = append(shape, value)
			value = 0
			digits = 0
			continue
		}
		if text[index] < '0' || text[index] > '9' {
			return nil, werrors.NewMalformed("bad shape extent")
		}
		value = value*10 + uint64(text[index]-'0')
		digits++
	}
	return shape, nil
}

func weightDecode(t *testing.T, delta, base []byte) ([]byte, error) {
	t.Helper()
	parsed, err := frame.Parse(delta, limits.Default())
	if err != nil {
		return nil, err
	}
	if base == nil {
		return modes.DecodeWeightDeltaFrame(context.Background(), parsed, dependency.NewMapResolver(), limits.Default())
	}
	return modes.DecodeWeightDeltaFrameWithBase(
		context.Background(), parsed, base, dependency.NewMapResolver(), limits.Default())
}

func TestWeightDeltaBF16RoundTrip(t *testing.T) {
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
	base := weightBaseFrame(t, "bf16", "2,4", baseBytes)
	dtype, _ := modes.ParseWeightDtype("bf16")
	delta, err := modes.EncodeWeightDelta(
		context.Background(), targetBytes, dtype, []uint64{2, 4}, base, limits.Default())
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	decoded, err := weightDecode(t, delta, base)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(decoded, targetBytes) {
		t.Fatalf("bf16 reconstruction differs")
	}
	parsed, _ := frame.Parse(delta, limits.Default())
	meta, err := modes.WeightDeltaMetadata(parsed, limits.Default())
	if err != nil {
		t.Fatalf("metadata: %v", err)
	}
	if meta.WordBits != 16 || meta.Planes != 2 || meta.Op != modes.WeightDeltaOp {
		t.Fatalf("unexpected descriptor: %+v", meta)
	}
}

func TestWeightDeltaFP32ZeroDelta(t *testing.T) {
	words := []uint32{0x7FC00001, 0x80000000, 0x00000001, 0x7F800000, 0xFF800000, 0}
	var bytesIn []byte
	for _, word := range words {
		bytesIn = append(bytesIn, byte(word), byte(word>>8), byte(word>>16), byte(word>>24))
	}
	base := weightBaseFrame(t, "f32", "3,2", bytesIn)
	dtype, _ := modes.ParseWeightDtype("fp32")
	delta, err := modes.EncodeWeightDelta(
		context.Background(), bytesIn, dtype, []uint64{3, 2}, base, limits.Default())
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	parsed, _ := frame.Parse(delta, limits.Default())
	meta, err := modes.WeightDeltaMetadata(parsed, limits.Default())
	if err != nil {
		t.Fatalf("metadata: %v", err)
	}
	if meta.PayloadLen != 4*256 {
		t.Fatalf("zero-delta payload is %d bytes, want 4 single-symbol tables", meta.PayloadLen)
	}
	decoded, err := weightDecode(t, delta, base)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(decoded, bytesIn) {
		t.Fatalf("fp32 zero-delta reconstruction differs")
	}
}

func TestWeightDeltaQuantizedRoundTrip(t *testing.T) {
	q8Base := make([]byte, 64)
	for index := range q8Base {
		q8Base[index] = byte((index*37 + 11) % 251)
	}
	q8Target := append([]byte(nil), q8Base...)
	for index := range q8Target {
		if index%5 == 0 {
			q8Target[index] += 7
		}
	}
	q8Frame := weightBaseFrame(t, "i8", "8,8", q8Base)
	q8Dtype, _ := modes.ParseWeightDtype("q8_0")
	q8Delta, err := modes.EncodeWeightDelta(
		context.Background(), q8Target, q8Dtype, []uint64{8, 8}, q8Frame, limits.Default())
	if err != nil {
		t.Fatalf("q8 encode: %v", err)
	}
	if decoded, err := weightDecode(t, q8Delta, q8Frame); err != nil || !bytes.Equal(decoded, q8Target) {
		t.Fatalf("q8 round-trip failed: %v", err)
	}

	q4Base := make([]byte, 48)
	for index := range q4Base {
		q4Base[index] = byte(index % 16)
	}
	q4Target := make([]byte, len(q4Base))
	for index := range q4Base {
		q4Target[index] = (q4Base[index] + byte(index%7)) % 16
	}
	q4Frame := weightBaseFrame(t, "u8", "6,8", q4Base)
	q4Dtype, _ := modes.ParseWeightDtype("q4_0")
	q4Delta, err := modes.EncodeWeightDelta(
		context.Background(), q4Target, q4Dtype, []uint64{6, 8}, q4Frame, limits.Default())
	if err != nil {
		t.Fatalf("q4 encode: %v", err)
	}
	parsed, _ := frame.Parse(q4Delta, limits.Default())
	meta, err := modes.WeightDeltaMetadata(parsed, limits.Default())
	if err != nil {
		t.Fatalf("q4 metadata: %v", err)
	}
	if meta.WordBits != 4 || meta.Planes != 1 {
		t.Fatalf("q4 descriptor: %+v", meta)
	}
	if decoded, err := weightDecode(t, q4Delta, q4Frame); err != nil || !bytes.Equal(decoded, q4Target) {
		t.Fatalf("q4 round-trip failed: %v", err)
	}
}

func TestWeightDeltaBaseMismatchAndMissing(t *testing.T) {
	baseBytes := make([]byte, 16)
	targetBytes := bytes.Repeat([]byte{1}, 16)
	base := weightBaseFrame(t, "bf16", "2,4", baseBytes)
	dtype, _ := modes.ParseWeightDtype("bf16")
	delta, err := modes.EncodeWeightDelta(
		context.Background(), targetBytes, dtype, []uint64{2, 4}, base, limits.Default())
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	fp32Base := weightBaseFrame(t, "f32", "2,4", make([]byte, 32))
	if _, err := weightDecode(t, delta, fp32Base); err == nil ||
		werrors.ClassOf(err) != werrors.DependencyMismatch {
		t.Fatalf("fp32 base: expected DependencyMismatch, got %v", err)
	}
	wrongShape := weightBaseFrame(t, "bf16", "4,2", baseBytes)
	if _, err := weightDecode(t, delta, wrongShape); err == nil ||
		werrors.ClassOf(err) != werrors.DependencyMismatch {
		t.Fatalf("wrong shape: expected DependencyMismatch, got %v", err)
	}
	if _, err := weightDecode(t, delta, nil); err == nil ||
		werrors.ClassOf(err) != werrors.DependencyMissing {
		t.Fatalf("missing base: expected DependencyMissing, got %v", err)
	}
}

func TestWeightDeltaDependencyCycle(t *testing.T) {
	shape := []uint64{2, 3}
	target := make([]byte, 12)
	contentHash := hash.Domain(hash.DomainChunk, target)
	var payload []byte
	payload = append(payload, modes.HuffmanEncode(make([]byte, 6))...)
	payload = append(payload, modes.HuffmanEncode(make([]byte, 6))...)
	bf16, _ := modes.ParseWeightDtype("bf16")
	meta := modes.WeightDeltaMeta{
		Dtype:        bf16,
		Shape:        shape,
		ElementCount: 6,
		WordBits:     16,
		Base:         contentHash,
		BaseDtype:    bf16,
		BaseShape:    shape,
		Op:           modes.WeightDeltaOp,
		Planes:       2,
		PlaneCodec:   modes.WeightDeltaPlaneCodec,
		DeltaRoot:    hash.Domain(tensor.DomainTensorValues, make([]byte, 12)),
		PayloadLen:   uint64(len(payload)),
	}
	encoded, err := modes.EncodeWeightDeltaDescriptorFrame(
		meta, payload, 12, contentHash, uint32(frame.ModeTensor), limits.Default())
	if err != nil {
		t.Fatalf("frame: %v", err)
	}
	source := dependency.NewMapFrameSource()
	if _, err := source.InsertFrame(context.Background(), encoded, limits.Default()); err != nil {
		t.Fatalf("insert: %v", err)
	}
	if _, err := modes.DecodeDAG(context.Background(), source, encoded, limits.Default()); err == nil ||
		werrors.ClassOf(err) != werrors.DependencyCycle {
		t.Fatalf("expected DependencyCycle, got %v", err)
	}
}

func TestWeightDeltaFourMiB(t *testing.T) {
	const elementCount = 2 * 1024 * 1024
	shape := []uint64{elementCount}
	state := uint64(0x123456789ABCDEF0)
	baseBytes := make([]byte, elementCount*2)
	for index := 0; index < elementCount; index++ {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		baseBytes[index*2] = byte(state)
		baseBytes[index*2+1] = byte(state >> 8)
	}
	targetBytes := append([]byte(nil), baseBytes...)
	changed := 0
	for index := 0; index < elementCount; index += 100 {
		targetBytes[index*2] ^= 0x01
		targetBytes[index*2+1] ^= 0x20
		changed++
	}
	if changed != (elementCount+99)/100 {
		t.Fatalf("changed %d words", changed)
	}
	base := weightBaseFrame(t, "bf16", "2097152", baseBytes)
	dtype, _ := modes.ParseWeightDtype("bf16")
	delta, err := modes.EncodeWeightDelta(
		context.Background(), targetBytes, dtype, shape, base, limits.Default())
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	decoded, err := weightDecode(t, delta, base)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(decoded, targetBytes) {
		t.Fatalf("reconstruction is not byte-exact")
	}
	original, err := tensor.NewLogicalRecord("bf16", shape, targetBytes, limits.Default())
	if err != nil {
		t.Fatalf("logical record: %v", err)
	}
	originalRoot, err := original.Root(limits.Default())
	if err != nil {
		t.Fatalf("logical root: %v", err)
	}
	decodedRecord, err := tensor.NewLogicalRecord("bf16", shape, decoded, limits.Default())
	if err != nil {
		t.Fatalf("decoded logical record: %v", err)
	}
	decodedRoot, err := decodedRecord.Root(limits.Default())
	if err != nil {
		t.Fatalf("decoded logical root: %v", err)
	}
	if originalRoot != decodedRoot {
		t.Fatalf("logical roots differ")
	}
	encoder, err := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedDefault))
	if err != nil {
		t.Fatalf("zstd: %v", err)
	}
	zstdBytes := len(encoder.EncodeAll(targetBytes, nil))
	encoder.Close()
	t.Logf("4 MiB bf16 1%% delta: delta_frame=%d independent_zstd_default=%d base_frame=%d changed_words=%d",
		len(delta), zstdBytes, len(base), changed)
	if len(delta)*4 >= zstdBytes {
		t.Fatalf("delta frame %d is not dramatically smaller than zstd %d", len(delta), zstdBytes)
	}
}
