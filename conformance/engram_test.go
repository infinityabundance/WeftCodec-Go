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
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

type engramGroupVector struct {
	Index         uint64  `json:"index"`
	RowStart      uint64  `json:"row_start"`
	RowEnd        uint64  `json:"row_end"`
	Storage       string  `json:"storage"`
	Offset        uint64  `json:"offset"`
	CompressedLen uint64  `json:"compressed_len"`
	GroupRoot     string  `json:"group_root"`
	Supersedes    *string `json:"supersedes"`
	Dependency    *uint64 `json:"dependency"`
}

type engramRegionVector struct {
	RowStart      uint64   `json:"row_start"`
	RowEnd        uint64   `json:"row_end"`
	BytesHex      string   `json:"bytes_hex"`
	GroupsTouched []uint32 `json:"groups_touched"`
}

type engramVector struct {
	Name             string              `json:"name"`
	Mode             string              `json:"mode"`
	ModeValue        *uint8              `json:"mode_value"`
	Schema           string              `json:"schema"`
	HashFunction     string              `json:"hash_function"`
	VocabSize        uint64              `json:"vocab_size"`
	TableShape       []uint64            `json:"table_shape"`
	Dtype            string              `json:"dtype"`
	RowGroupEdges    []uint64            `json:"row_group_edges"`
	GroupCount       uint64              `json:"group_count"`
	Groups           []engramGroupVector `json:"groups"`
	PayloadLen       uint64              `json:"payload_len"`
	PayloadHash      string              `json:"payload_hash"`
	DescriptorRoot   string              `json:"descriptor_root"`
	DecodedLen       uint64              `json:"decoded_len"`
	ContentHash      string              `json:"content_hash"`
	TableHex         string              `json:"table_hex"`
	FrameHex         string              `json:"frame_hex"`
	DependencyFrames []string            `json:"dependency_frames"`
	Region           *engramRegionVector `json:"region"`
	ExpectError      string              `json:"expect_error"`
	Note             string              `json:"note"`
}

// TestEngramVectors replays conformance/v1/engram. The frames are produced by
// the Rust generator; this independently checks the Go decoder reproduces the
// exact table bytes, the descriptor root, the group table and a region read.
func TestEngramVectors(t *testing.T) {
	files := jsonFiles(t, filepath.Join("v1", "engram"))
	if len(files) < 20 {
		t.Fatalf("expected at least 20 engram vectors, found %d", len(files))
	}
	ctx := context.Background()
	lim := limits.Default()
	valid, rejects := 0, 0
	for _, path := range files {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			var vector engramVector
			loadJSON(t, path, &vector)
			frameBytes := mustHex(t, "frame", vector.FrameHex)

			if vector.ExpectError != "" {
				source := dependency.NewMapFrameSource()
				for _, frameHex := range vector.DependencyFrames {
					if _, err := source.InsertFrame(ctx, mustHex(t, "dependency frame", frameHex), lim); err != nil {
						t.Fatalf("dependency frame: %v", err)
					}
				}
				parsed, err := frame.Parse(frameBytes, lim)
				if err == nil {
					_, err = modes.DecodeEngram(ctx, parsed, source, lim)
				}
				if err == nil {
					t.Fatalf("expected %s, decode succeeded", vector.ExpectError)
				}
				if got := string(werrors.ClassOf(err)); got != vector.ExpectError {
					t.Fatalf("expected class %s, got %s (%v)", vector.ExpectError, got, err)
				}
				rejects++
				return
			}

			parsed, err := frame.Parse(frameBytes, lim)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			if parsed.Header.Mode != frame.ModeEngramShard {
				t.Fatalf("mode %s, want engram_shard", parsed.Header.Mode)
			}
			if vector.ModeValue != nil && uint8(parsed.Header.Mode) != *vector.ModeValue {
				t.Fatalf("mode value %d, want %d", parsed.Header.Mode, *vector.ModeValue)
			}
			descriptor, err := modes.ParseEngram(parsed, lim)
			if err != nil {
				t.Fatalf("descriptor: %v", err)
			}

			if descriptor.Meta.HashFunction != vector.HashFunction {
				t.Fatalf("hash function %s, want %s", descriptor.Meta.HashFunction, vector.HashFunction)
			}
			if !u64Equal(descriptor.Meta.TableShape, vector.TableShape) {
				t.Fatalf("shape %v, want %v", descriptor.Meta.TableShape, vector.TableShape)
			}
			if !u64Equal(descriptor.Meta.RowGroupEdges, vector.RowGroupEdges) {
				t.Fatalf("edges %v, want %v", descriptor.Meta.RowGroupEdges, vector.RowGroupEdges)
			}
			if uint64(len(descriptor.Meta.Groups)) != vector.GroupCount {
				t.Fatalf("group count %d, want %d", len(descriptor.Meta.Groups), vector.GroupCount)
			}
			if descriptor.Meta.PayloadLen != vector.PayloadLen {
				t.Fatalf("payload_len %d, want %d", descriptor.Meta.PayloadLen, vector.PayloadLen)
			}
			root, err := descriptor.Meta.DescriptorRoot(lim)
			if err != nil {
				t.Fatalf("descriptor root: %v", err)
			}
			if root.Hex() != vector.DescriptorRoot {
				t.Fatalf("descriptor root %s, want %s", root, vector.DescriptorRoot)
			}
			decoded, err := modes.DecodeEngramWithoutDependencies(ctx, parsed, lim)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			table := mustHex(t, "table", vector.TableHex)
			if !bytes.Equal(decoded, table) {
				t.Fatalf("table mismatch: got %d bytes, want %d", len(decoded), len(table))
			}
			if vector.Region != nil {
				got, touched, err := modes.ReadEngramRegion(
					ctx, parsed, descriptor, vector.Region.RowStart, vector.Region.RowEnd,
					dependency.NewMapFrameSource(), lim)
				if err != nil {
					t.Fatalf("region: %v", err)
				}
				if !bytes.Equal(got, mustHex(t, "region", vector.Region.BytesHex)) {
					t.Fatalf("region bytes mismatch")
				}
				if len(touched) != len(vector.Region.GroupsTouched) {
					t.Fatalf("touched %v, want %v", touched, vector.Region.GroupsTouched)
				}
				for index := range touched {
					if touched[index] != vector.Region.GroupsTouched[index] {
						t.Fatalf("touched %v, want %v", touched, vector.Region.GroupsTouched)
					}
				}
			}
			valid++
		})
	}
	if valid != 6 || rejects != 14 {
		t.Fatalf("expected 6 valid / 14 rejects, got %d / %d", valid, rejects)
	}
}

// TestEngramGenericDispatch proves the ordinary DecodeBytes path handles an
// inline mode-20 frame without a special caller.
func TestEngramGenericDispatch(t *testing.T) {
	dir := filepath.Join("v1", "engram")
	var vector engramVector
	loadJSON(t, filepath.Join(dir, "001_engram_bf16_one_group.json"), &vector)
	ctx := context.Background()
	lim := limits.Default()
	decoded, err := modes.DecodeBytes(
		ctx, mustHex(t, "frame", vector.FrameHex), dependency.NewMapResolver(), lim)
	if err != nil {
		t.Fatalf("DecodeBytes: %v", err)
	}
	if !bytes.Equal(decoded, mustHex(t, "table", vector.TableHex)) {
		t.Fatalf("DecodeBytes table mismatch")
	}
}
