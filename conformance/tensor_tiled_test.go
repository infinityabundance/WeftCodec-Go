package conformance_test

import (
	"bytes"
	"context"
	"path/filepath"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

type tileRegionVector struct {
	Start        []uint64 `json:"start"`
	End          []uint64 `json:"end"`
	BytesHex     string   `json:"bytes_hex"`
	TilesTouched uint64   `json:"tiles_touched"`
}

type tiledVector struct {
	Name        string            `json:"name"`
	Mode        string            `json:"mode"`
	ModeValue   *uint8            `json:"mode_value"`
	Dtype       string            `json:"dtype"`
	Shape       []uint64          `json:"shape"`
	TileShape   []uint64          `json:"tile_shape"`
	TileCounts  []uint64          `json:"tile_counts"`
	TileCount   uint64            `json:"tile_count"`
	LogicalRoot string            `json:"logical_root"`
	PayloadLen  uint64            `json:"payload_len"`
	DecodedLen  uint64            `json:"decoded_len"`
	ContentHash string            `json:"content_hash"`
	Plaintext   string            `json:"plaintext_hex"`
	FrameHex    string            `json:"frame_hex"`
	Region      *tileRegionVector `json:"region"`
	Expect      expectation       `json:"expect"`
}

// TestTiledVectors replays conformance/v1/tensor-tiled. The frames are
// produced by the Rust generator; this independently checks the Go decoder
// reproduces the exact bytes, every descriptor field, and the tile region API.
func TestTiledVectors(t *testing.T) {
	files := jsonFiles(t, filepath.Join("v1", "tensor-tiled"))
	if len(files) < 11 {
		t.Fatalf("expected at least 11 tensor-tiled vectors, found %d", len(files))
	}
	ctx := context.Background()
	lim := limits.Default()
	resolver := dependency.NewMapResolver()
	valid, rejects := 0, 0
	for _, path := range files {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			var vector tiledVector
			loadJSON(t, path, &vector)
			frameBytes := mustHex(t, "frame", vector.FrameHex)
			if vector.Expect.Error != "" {
				_, err := modes.DecodeBytes(ctx, frameBytes, resolver, lim)
				if err == nil {
					t.Fatalf("expected %s, decode succeeded", vector.Expect.Error)
				}
				if got := string(werrors.ClassOf(err)); got != vector.Expect.Error {
					t.Fatalf("expected class %s, got %s (%v)", vector.Expect.Error, got, err)
				}
				rejects++
				return
			}
			parsed, err := frame.Parse(frameBytes, lim)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			if parsed.Header.Mode != frame.ModeTensorTiled {
				t.Fatalf("mode %s, want tensor_tiled", parsed.Header.Mode)
			}
			if vector.ModeValue != nil && uint8(parsed.Header.Mode) != *vector.ModeValue {
				t.Fatalf("mode value %d, want %d", parsed.Header.Mode, *vector.ModeValue)
			}
			descriptor, err := modes.ParseTiledTensor(parsed, lim)
			if err != nil {
				t.Fatalf("descriptor: %v", err)
			}
			if descriptor.Meta.Dtype != vector.Dtype {
				t.Fatalf("dtype %s, want %s", descriptor.Meta.Dtype, vector.Dtype)
			}
			if !u64Equal(descriptor.Meta.Shape, vector.Shape) {
				t.Fatalf("shape %v, want %v", descriptor.Meta.Shape, vector.Shape)
			}
			if !u64Equal(descriptor.Meta.TileShape, vector.TileShape) {
				t.Fatalf("tile shape %v, want %v", descriptor.Meta.TileShape, vector.TileShape)
			}
			if !u64Equal(descriptor.Meta.TileCounts, vector.TileCounts) {
				t.Fatalf("tile counts %v, want %v", descriptor.Meta.TileCounts, vector.TileCounts)
			}
			if uint64(len(descriptor.Meta.Tiles)) != vector.TileCount {
				t.Fatalf("tile count %d, want %d", len(descriptor.Meta.Tiles), vector.TileCount)
			}
			if descriptor.Meta.LogicalRoot.Hex() != vector.LogicalRoot {
				t.Fatalf("logical root %s, want %s", descriptor.Meta.LogicalRoot, vector.LogicalRoot)
			}
			if descriptor.Meta.PayloadLen != vector.PayloadLen {
				t.Fatalf("payload_len %d, want %d", descriptor.Meta.PayloadLen, vector.PayloadLen)
			}
			decoded, err := modes.DecodeBytes(ctx, frameBytes, resolver, lim)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			plaintext := mustHex(t, "plaintext", vector.Plaintext)
			if !bytes.Equal(decoded, plaintext) {
				t.Fatalf("plaintext mismatch: got %d bytes, want %d", len(decoded), len(plaintext))
			}
			if vector.Region != nil {
				region := tensor.NewRegion(vector.Region.Start, vector.Region.End)
				got, touched, err := modes.ReadRegion(descriptor, region, lim)
				if err != nil {
					t.Fatalf("region: %v", err)
				}
				if !bytes.Equal(got, mustHex(t, "region bytes", vector.Region.BytesHex)) {
					t.Fatal("region bytes differ")
				}
				if touched != vector.Region.TilesTouched {
					t.Fatalf("tiles touched %d, want %d", touched, vector.Region.TilesTouched)
				}
			}
			valid++
		})
	}
	if valid != 3 {
		t.Fatalf("valid vectors = %d, want 3", valid)
	}
	if rejects != 8 {
		t.Fatalf("reject vectors = %d, want 8", rejects)
	}
}

func u64Equal(left, right []uint64) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}
