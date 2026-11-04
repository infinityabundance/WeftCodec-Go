package modes_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func shardedPatternQ8(count int) []int8 {
	out := make([]int8, count)
	for i := 0; i < count; i++ {
		out[i] = int8((i*7+3)%251 - 125)
	}
	return out
}

func shardedPatternQ4(count int) []int8 {
	out := make([]int8, count)
	for i := 0; i < count; i++ {
		out[i] = int8(i%16 - 8)
	}
	return out
}

func shardedQ8Case() modes.ShardedTensorInput {
	return modes.ShardedTensorInput{
		Dtype:      modes.ShardedQ8_0,
		Shape:      []uint64{4, 8},
		ShardAxis:  0,
		ShardCount: 2,
		BlockSize:  16,
		Scales:     []uint32{f32bits(0.5), f32bits(0.25)},
		ZeroPoints: []int64{0, 3},
		Words:      shardedPatternQ8(32),
	}
}

func shardedQ4Case() modes.ShardedTensorInput {
	return modes.ShardedTensorInput{
		Dtype:      modes.ShardedQ4_0,
		Shape:      []uint64{2, 4, 8},
		ShardAxis:  1,
		ShardCount: 4,
		BlockSize:  16,
		Scales:     []uint32{f32bits(0.5), f32bits(0.25), f32bits(0.125), f32bits(0.0625)},
		ZeroPoints: []int64{0, 1, -1, 2},
		Words:      shardedPatternQ4(64),
	}
}

func f32bits(value float32) uint32 {
	return math.Float32bits(value)
}

func int8Bytes(words []int8) []byte {
	out := make([]byte, len(words))
	for index, word := range words {
		out[index] = byte(word)
	}
	return out
}

func decodeShards(t *testing.T, frames [][]byte, lim limits.Limits) []modes.DecodedTensorShard {
	t.Helper()
	out := make([]modes.DecodedTensorShard, 0, len(frames))
	for _, encoded := range frames {
		parsed, err := frame.Parse(encoded, lim)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		shard, err := modes.DecodeShardFrame(parsed, lim)
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		out = append(out, shard)
	}
	return out
}

func expectedShardWords(t *testing.T, input modes.ShardedTensorInput, index uint32, lim limits.Limits) []int8 {
	t.Helper()
	words := make([]byte, len(input.Words))
	for i, word := range input.Words {
		words[i] = byte(word)
	}
	shards, err := tensor.SplitShards("i8", input.Shape, input.ShardAxis, input.ShardCount, words, lim)
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	out := make([]int8, len(shards[index]))
	for i, value := range shards[index] {
		out[i] = int8(value)
	}
	return out
}

func TestTensorShardedQ8Roundtrip(t *testing.T) {
	lim := limits.Default()
	input := shardedQ8Case()
	frames, err := modes.EncodeShardedTensor(input, lim)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(frames) != 2 {
		t.Fatalf("got %d frames, want 2", len(frames))
	}
	root, err := modes.LogicalRootSharded(
		input.Dtype, input.Shape, input.Words, input.Scales, input.ZeroPoints, lim)
	if err != nil {
		t.Fatalf("root: %v", err)
	}
	decoded := decodeShards(t, frames, lim)
	for index, shard := range decoded {
		if shard.Meta.ShardIndex != uint32(index) {
			t.Fatalf("shard index %d, want %d", shard.Meta.ShardIndex, index)
		}
		if shard.Meta.LayoutRoot != root {
			t.Fatalf("layout root mismatch on shard %d", index)
		}
		want := expectedShardWords(t, input, uint32(index), lim)
		if !bytes.Equal(int8Bytes(shard.Words), int8Bytes(want)) {
			t.Fatalf("shard %d words differ", index)
		}
		planesLen, err := shard.Meta.PlanesLen()
		if err != nil {
			t.Fatalf("planes len: %v", err)
		}
		if uint64(len(shard.Planes)) != planesLen {
			t.Fatalf("planes %d bytes, want %d", len(shard.Planes), planesLen)
		}
		reencoded, err := modes.EncodeShardFrame(shard.Meta, shard.Planes, lim)
		if err != nil {
			t.Fatalf("re-encode: %v", err)
		}
		if !bytes.Equal(reencoded, frames[index]) {
			t.Fatalf("frame bytes differ for shard %d", index)
		}
	}
	words, err := modes.AssembleShardedWords(decoded, lim)
	if err != nil {
		t.Fatalf("assemble: %v", err)
	}
	if !bytes.Equal(int8Bytes(words), int8Bytes(input.Words)) {
		t.Fatalf("assembled words differ")
	}
	shuffled := []modes.DecodedTensorShard{decoded[1], decoded[0]}
	got, err := modes.ShardedSetRoot(shuffled, lim)
	if err != nil {
		t.Fatalf("set root: %v", err)
	}
	if got != root {
		t.Fatalf("set root %s, want %s", got, root)
	}
}

func TestTensorShardedQ4Nibbles(t *testing.T) {
	lim := limits.Default()
	input := shardedQ4Case()
	frames, err := modes.EncodeShardedTensor(input, lim)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	decoded := decodeShards(t, frames, lim)
	for index, shard := range decoded {
		want := expectedShardWords(t, input, uint32(index), lim)
		if !bytes.Equal(int8Bytes(shard.Words), int8Bytes(want)) {
			t.Fatalf("shard %d words differ", index)
		}
		for pair := 0; pair < len(want)/2; pair++ {
			low := byte(shard.Words[2*pair]) & 0x0f
			high := byte(shard.Words[2*pair+1]) & 0x0f
			if shard.Planes[pair] != low|(high<<4) {
				t.Fatalf("shard %d nibble pair %d mismatches", index, pair)
			}
		}
	}
	shuffled := []modes.DecodedTensorShard{decoded[2], decoded[0], decoded[3], decoded[1]}
	if _, err := modes.ShardedSetRoot(shuffled, lim); err != nil {
		t.Fatalf("set root: %v", err)
	}
}

func TestTensorShardedSingleShard(t *testing.T) {
	lim := limits.Default()
	input := modes.ShardedTensorInput{
		Dtype:      modes.ShardedQ8_0,
		Shape:      []uint64{16},
		ShardAxis:  0,
		ShardCount: 1,
		BlockSize:  16,
		Scales:     []uint32{f32bits(1.0)},
		ZeroPoints: []int64{0},
		Words:      shardedPatternQ8(16),
	}
	frames, err := modes.EncodeShardedTensor(input, lim)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(frames) != 1 {
		t.Fatalf("got %d frames", len(frames))
	}
	decoded := decodeShards(t, frames, lim)
	root, err := modes.ShardedSetRoot(decoded, lim)
	if err != nil {
		t.Fatalf("root: %v", err)
	}
	if root != decoded[0].Meta.LayoutRoot {
		t.Fatalf("root mismatch")
	}
}

func TestTensorShardedRootIndependentOfAxis(t *testing.T) {
	lim := limits.Default()
	input := shardedQ8Case()
	root, err := modes.LogicalRootSharded(
		input.Dtype, input.Shape, input.Words, input.Scales, input.ZeroPoints, lim)
	if err != nil {
		t.Fatalf("root: %v", err)
	}
	input.ShardAxis = 1
	frames, err := modes.EncodeShardedTensor(input, lim)
	if err != nil {
		t.Fatalf("encode axis 1: %v", err)
	}
	decoded := decodeShards(t, frames, lim)
	got, err := modes.ShardedSetRoot(decoded, lim)
	if err != nil {
		t.Fatalf("set root: %v", err)
	}
	if got != root {
		t.Fatalf("root %s, want %s", got, root)
	}
}

func TestTensorShardedDuplicateAndMissing(t *testing.T) {
	lim := limits.Default()
	frames, err := modes.EncodeShardedTensor(shardedQ8Case(), lim)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	decoded := decodeShards(t, frames, lim)
	duplicate := []modes.DecodedTensorShard{decoded[0], decoded[0]}
	if _, err := modes.ShardedSetRoot(duplicate, lim); werrors.ClassOf(err) != werrors.NonCanonical {
		t.Fatalf("duplicate: got %v", err)
	}
	if _, err := modes.ShardedSetRoot(decoded[:1], lim); werrors.ClassOf(err) != werrors.Malformed {
		t.Fatalf("missing: got %v", err)
	}
}

func TestTensorShardedDescriptorRejects(t *testing.T) {
	lim := limits.Default()
	frames, err := modes.EncodeShardedTensor(shardedQ8Case(), lim)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	decoded := decodeShards(t, frames, lim)
	shard := decoded[0]
	stored := modes.HuffmanEncode(shard.Planes)

	cases := []struct {
		name string
		meta modes.ShardedTensorMeta
	}{
		{"element_count", func() modes.ShardedTensorMeta { m := shard.Meta; m.ElementCount = 31; return m }()},
		{"block_size", func() modes.ShardedTensorMeta { m := shard.Meta; m.BlockSize = 24; return m }()},
		{"divisibility", func() modes.ShardedTensorMeta { m := shard.Meta; m.ShardAxis = 1; m.ShardCount = 3; return m }()},
		{"index", func() modes.ShardedTensorMeta { m := shard.Meta; m.ShardIndex = 2; return m }()},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			meta := tc.meta
			meta.PayloadLen = uint64(len(stored))
			encoded, err := modes.EncodeShardFrameWithStored(meta, shard.Planes, stored, lim)
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			parsed, err := frame.Parse(encoded, lim)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			if _, err := modes.DecodeShardFrame(parsed, lim); werrors.ClassOf(err) != werrors.Malformed {
				t.Fatalf("got %v", err)
			}
		})
	}
}

func TestTensorShardedScaleMismatchAndHuffman(t *testing.T) {
	lim := limits.Default()
	frames, err := modes.EncodeShardedTensor(shardedQ8Case(), lim)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	decoded := decodeShards(t, frames, lim)
	shard := decoded[0]

	planes := append([]byte(nil), shard.Planes...)
	planes[len(planes)-1] ^= 0x01
	encoded, err := modes.EncodeShardFrame(shard.Meta, planes, lim)
	if err != nil {
		t.Fatalf("build mismatch: %v", err)
	}
	parsed, err := frame.Parse(encoded, lim)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if _, err := modes.DecodeShardFrame(parsed, lim); werrors.ClassOf(err) != werrors.Corrupt {
		t.Fatalf("scale mismatch: got %v", err)
	}

	stored := modes.HuffmanEncode(shard.Planes)
	stored = stored[:len(stored)-1]
	meta := shard.Meta
	meta.PayloadLen = uint64(len(stored))
	encoded, err = modes.EncodeShardFrameWithStored(meta, shard.Planes, stored, lim)
	if err != nil {
		t.Fatalf("build huffman: %v", err)
	}
	parsed, err = frame.Parse(encoded, lim)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	_, err = modes.DecodeShardFrame(parsed, lim)
	if class := werrors.ClassOf(err); class != werrors.Corrupt && class != werrors.NonCanonical {
		t.Fatalf("malformed huffman: got %v", err)
	}
}

func TestTensorShardedLimits(t *testing.T) {
	rankLimits := limits.Default()
	rankLimits.MaxTensorRank = 2
	_, err := modes.EncodeShardedTensor(modes.ShardedTensorInput{
		Dtype:      modes.ShardedQ8_0,
		Shape:      []uint64{2, 2, 2},
		ShardAxis:  0,
		ShardCount: 1,
		BlockSize:  16,
		Scales:     []uint32{f32bits(1.0)},
		ZeroPoints: []int64{0},
		Words:      make([]int8, 8),
	}, rankLimits)
	if werrors.ClassOf(err) != werrors.LimitExceeded {
		t.Fatalf("rank: got %v", err)
	}

	elementLimits := limits.Default()
	elementLimits.MaxTensorElements = 4
	_, err = modes.EncodeShardedTensor(modes.ShardedTensorInput{
		Dtype:      modes.ShardedQ8_0,
		Shape:      []uint64{8},
		ShardAxis:  0,
		ShardCount: 1,
		BlockSize:  16,
		Scales:     []uint32{f32bits(1.0)},
		ZeroPoints: []int64{0},
		Words:      make([]int8, 8),
	}, elementLimits)
	if werrors.ClassOf(err) != werrors.LimitExceeded {
		t.Fatalf("element: got %v", err)
	}
}
