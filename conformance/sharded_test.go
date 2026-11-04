package conformance_test

import (
	"bytes"
	"encoding/hex"
	"path/filepath"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

type shardedFrameVector struct {
	Index    *uint32 `json:"index"`
	FrameHex string  `json:"frame_hex"`
}

type shardedVector struct {
	Name         string               `json:"name"`
	Schema       string               `json:"schema"`
	Dtype        string               `json:"dtype"`
	Shape        []uint64             `json:"shape"`
	ElementCount uint64               `json:"element_count"`
	Axis         uint32               `json:"axis"`
	Shards       uint32               `json:"shards"`
	BlockSize    uint64               `json:"block_size"`
	Scales       []uint32             `json:"scales"`
	ZeroPoints   []int64              `json:"zero_points"`
	WordsHex     string               `json:"words_hex"`
	LayoutRoot   string               `json:"layout_root"`
	Frames       []shardedFrameVector `json:"frames"`
	Expect       struct {
		OK bool `json:"ok"`
	} `json:"expect"`
	ExpectError string `json:"expect_error"`
	Note        string `json:"note"`
}

func TestShardedVectors(t *testing.T) {
	files := jsonFiles(t, filepath.Join("v1", "sharded"))
	if len(files) != 11 {
		t.Fatalf("expected 11 TENSOR_SHARDED vectors, found %d", len(files))
	}
	lim := limits.Default()
	for _, path := range files {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			var vector shardedVector
			loadJSON(t, path, &vector)
			if vector.ExpectError != "" {
				shards := make([]modes.DecodedTensorShard, 0, len(vector.Frames))
				var decodeErr error
				for _, entry := range vector.Frames {
					encoded := mustHex(t, "frame", entry.FrameHex)
					parsed, err := frame.Parse(encoded, lim)
					if err != nil {
						decodeErr = err
						break
					}
					shard, err := modes.DecodeShardFrame(parsed, lim)
					if err != nil {
						decodeErr = err
						break
					}
					if entry.Index != nil && *entry.Index != shard.Meta.ShardIndex {
						t.Fatalf("frame index %d, descriptor index %d",
							*entry.Index, shard.Meta.ShardIndex)
					}
					shards = append(shards, shard)
				}
				if decodeErr == nil {
					_, decodeErr = modes.ShardedSetRoot(shards, lim)
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

			shards := make([]modes.DecodedTensorShard, 0, len(vector.Frames))
			for _, entry := range vector.Frames {
				encoded := mustHex(t, "frame", entry.FrameHex)
				parsed, err := frame.Parse(encoded, lim)
				if err != nil {
					t.Fatalf("parse: %v", err)
				}
				shard, err := modes.DecodeShardFrame(parsed, lim)
				if err != nil {
					t.Fatalf("decode: %v", err)
				}
				if entry.Index != nil && *entry.Index != shard.Meta.ShardIndex {
					t.Fatalf("frame index %d, descriptor index %d",
						*entry.Index, shard.Meta.ShardIndex)
				}
				reencoded, err := modes.EncodeShardFrame(shard.Meta, shard.Planes, lim)
				if err != nil {
					t.Fatalf("re-encode: %v", err)
				}
				if !bytes.Equal(reencoded, encoded) {
					t.Fatalf("frame bytes differ for shard %d", shard.Meta.ShardIndex)
				}
				shards = append(shards, shard)
			}
			root, err := modes.ShardedSetRoot(shards, lim)
			if err != nil {
				t.Fatalf("set root: %v", err)
			}
			if root.Hex() != vector.LayoutRoot {
				t.Fatalf("root %s, want %s", root, vector.LayoutRoot)
			}
			words, err := modes.AssembleShardedWords(shards, lim)
			if err != nil {
				t.Fatalf("assemble: %v", err)
			}
			wordsHex := make([]byte, len(words))
			for index, word := range words {
				wordsHex[index] = byte(word)
			}
			if hex.EncodeToString(wordsHex) != vector.WordsHex {
				t.Fatalf("assembled words differ")
			}
			meta := shards[0].Meta
			if meta.Dtype.Name() != vector.Dtype ||
				meta.ShardAxis != vector.Axis ||
				meta.ShardCount != vector.Shards ||
				meta.BlockSize != vector.BlockSize ||
				meta.ElementCount != vector.ElementCount {
				t.Fatalf("descriptor does not match the vector")
			}
		})
	}
}
