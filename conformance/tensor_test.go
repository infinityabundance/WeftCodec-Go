package conformance_test

import (
	"bytes"
	"encoding/hex"
	"path/filepath"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

type tensorVector struct {
	Name               string   `json:"name"`
	Kind               string   `json:"kind"`
	Dtype              string   `json:"dtype"`
	Shape              []uint64 `json:"shape"`
	ElementCount       uint64   `json:"element_count"`
	ValuesHex          string   `json:"values_hex"`
	ValuesHash         string   `json:"values_hash"`
	CanonicalHex       string   `json:"canonical_hex"`
	LogicalRoot        string   `json:"logical_root"`
	Layout             string   `json:"layout"`
	BlockShape         []uint64 `json:"block_shape"`
	Axis               *uint32  `json:"axis"`
	Shards             *uint32  `json:"shards"`
	Index              *uint32  `json:"index"`
	RecordHex          string   `json:"record_hex"`
	ExpectError        string   `json:"expect_error"`
	ShardHex           []string `json:"shard_hex"`
	ShardDescriptorHex []string `json:"shard_descriptor_hex"`
	ShardContainerHex  []string `json:"shard_container_hex"`
}

// TestTensorVectors replays conformance/v1/tensor. The artifacts are produced
// by the Rust generator; this independently checks the Go decoder reproduces
// every canonical record, root, and container byte-for-byte.
func TestTensorVectors(t *testing.T) {
	files := jsonFiles(t, filepath.Join("v1", "tensor"))
	if len(files) < 28 {
		t.Fatalf("expected at least 28 tensor vectors, found %d", len(files))
	}
	lim := limits.Default()
	logicalOK, logicalRejects := 0, 0
	layoutOK, layoutRejects := 0, 0
	shardSets := 0

	for _, path := range files {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			var vector tensorVector
			loadJSON(t, path, &vector)
			switch vector.Kind {
			case "logical":
				if vector.ExpectError != "" {
					record, err := tensor.ParseLogicalRecord(mustHex(t, "record", vector.RecordHex), lim)
					if err == nil {
						t.Fatalf("expected %s, decode succeeded (%v)", vector.ExpectError, record)
					}
					if got := string(werrors.ClassOf(err)); got != vector.ExpectError {
						t.Fatalf("expected class %s, got %s (%v)", vector.ExpectError, got, err)
					}
					logicalRejects++
					return
				}
				values := mustHex(t, "values", vector.ValuesHex)
				record, err := tensor.NewLogicalRecord(vector.Dtype, vector.Shape, values, lim)
				if err != nil {
					t.Fatalf("record: %v", err)
				}
				canonical, err := record.CanonicalBytes(lim)
				if err != nil {
					t.Fatalf("canonical: %v", err)
				}
				if got := hex.EncodeToString(canonical); got != vector.CanonicalHex {
					t.Fatalf("canonical bytes\n got %s\nwant %s", got, vector.CanonicalHex)
				}
				if got := record.ValuesHash.Hex(); got != vector.ValuesHash {
					t.Fatalf("values hash %s, want %s", got, vector.ValuesHash)
				}
				if record.ElementCount != vector.ElementCount {
					t.Fatalf("element count %d, want %d", record.ElementCount, vector.ElementCount)
				}
				root, err := record.Root(lim)
				if err != nil || root.Hex() != vector.LogicalRoot {
					t.Fatalf("root %s, want %s (%v)", root, vector.LogicalRoot, err)
				}
				direct, err := tensor.LogicalRoot(vector.Dtype, vector.Shape, values, lim)
				if err != nil || direct.Hex() != vector.LogicalRoot {
					t.Fatalf("direct root %s, want %s (%v)", direct, vector.LogicalRoot, err)
				}
				parsed, err := tensor.ParseLogicalRecord(mustHex(t, "canonical", vector.CanonicalHex), lim)
				if err != nil {
					t.Fatalf("parse: %v", err)
				}
				if !parsed.Equal(record) {
					t.Fatal("record did not round-trip")
				}
				logicalOK++
			case "layout":
				if vector.ExpectError != "" {
					descriptor, err := tensor.ParseDescriptor(mustHex(t, "record", vector.RecordHex), lim)
					if err == nil {
						t.Fatalf("expected %s, decode succeeded (%v)", vector.ExpectError, descriptor)
					}
					if got := string(werrors.ClassOf(err)); got != vector.ExpectError {
						t.Fatalf("expected class %s, got %s (%v)", vector.ExpectError, got, err)
					}
					layoutRejects++
					return
				}
				canonical := mustHex(t, "canonical", vector.CanonicalHex)
				descriptor, err := tensor.ParseDescriptor(canonical, lim)
				if err != nil {
					t.Fatalf("parse: %v", err)
				}
				if descriptor.Dtype != vector.Dtype {
					t.Fatalf("dtype %s, want %s", descriptor.Dtype, vector.Dtype)
				}
				if !equalU64s(descriptor.Shape, vector.Shape) {
					t.Fatalf("shape %v, want %v", descriptor.Shape, vector.Shape)
				}
				if descriptor.Layout.Name() != vector.Layout {
					t.Fatalf("layout %s, want %s", descriptor.Layout.Name(), vector.Layout)
				}
				switch descriptor.Layout.Kind {
				case tensor.Blocked:
					if !equalU64s(descriptor.Layout.BlockShape, vector.BlockShape) {
						t.Fatalf("block shape %v, want %v", descriptor.Layout.BlockShape, vector.BlockShape)
					}
				case tensor.Sharded:
					if vector.Axis == nil || descriptor.Layout.Axis != *vector.Axis {
						t.Fatalf("axis %d, want %v", descriptor.Layout.Axis, vector.Axis)
					}
					if vector.Shards == nil || descriptor.Layout.Shards != *vector.Shards {
						t.Fatalf("shards %d, want %v", descriptor.Layout.Shards, vector.Shards)
					}
					if vector.Index == nil || descriptor.Layout.Index != *vector.Index {
						t.Fatalf("index %d, want %v", descriptor.Layout.Index, vector.Index)
					}
				}
				reencoded, err := descriptor.CanonicalBytes(lim)
				if err != nil || !bytes.Equal(reencoded, canonical) {
					t.Fatalf("re-encode mismatch: %v", err)
				}
				layoutOK++
			case "shards":
				values := mustHex(t, "values", vector.ValuesHex)
				if vector.Axis == nil || vector.Shards == nil || vector.Shards != nil && *vector.Shards == 0 {
					t.Fatalf("shard vector missing axis/shards")
				}
				axis, shards := *vector.Axis, *vector.Shards
				root, err := tensor.LogicalRoot(vector.Dtype, vector.Shape, values, lim)
				if err != nil || root.Hex() != vector.LogicalRoot {
					t.Fatalf("logical root %s, want %s (%v)", root, vector.LogicalRoot, err)
				}
				if len(vector.ShardHex) != int(shards) ||
					len(vector.ShardDescriptorHex) != int(shards) ||
					len(vector.ShardContainerHex) != int(shards) {
					t.Fatalf("shard vector has %d/%d/%d entries for %d shards",
						len(vector.ShardHex), len(vector.ShardDescriptorHex),
						len(vector.ShardContainerHex), shards)
				}
				split, err := tensor.SplitShards(vector.Dtype, vector.Shape, axis, shards, values, lim)
				if err != nil {
					t.Fatalf("split: %v", err)
				}
				decoded := make([]tensor.DecodedShard, 0, shards)
				for index := uint32(0); index < shards; index++ {
					payload := mustHex(t, "shard", vector.ShardHex[index])
					if !bytes.Equal(split[index], payload) {
						t.Fatalf("shard %d bytes differ", index)
					}
					descriptor := tensor.ShardedDescriptor(vector.Dtype, vector.Shape, axis, shards, index)
					descriptorBytes, err := descriptor.CanonicalBytes(lim)
					if err != nil {
						t.Fatalf("descriptor %d: %v", index, err)
					}
					if !bytes.Equal(descriptorBytes, mustHex(t, "descriptor", vector.ShardDescriptorHex[index])) {
						t.Fatalf("shard %d descriptor bytes differ", index)
					}
					container, err := tensor.EncodeShard(descriptor, payload, lim)
					if err != nil {
						t.Fatalf("encode shard %d: %v", index, err)
					}
					if !bytes.Equal(container, mustHex(t, "container", vector.ShardContainerHex[index])) {
						t.Fatalf("shard %d container bytes differ", index)
					}
					shard, err := tensor.DecodeShard(container, lim)
					if err != nil {
						t.Fatalf("decode shard %d: %v", index, err)
					}
					decoded = append(decoded, shard)
				}
				got, err := tensor.ShardRoot(decoded, lim)
				if err != nil || got.Hex() != vector.LogicalRoot {
					t.Fatalf("shard root %s, want %s (%v)", got, vector.LogicalRoot, err)
				}
				assembled, err := tensor.AssembleShards(decoded, lim)
				if err != nil || !bytes.Equal(assembled, values) {
					t.Fatalf("reassembled bytes differ: %v", err)
				}
				// Shuffled shard order is reordered by index.
				reversed := make([]tensor.DecodedShard, len(decoded))
				for i := range decoded {
					reversed[i] = decoded[len(decoded)-1-i]
				}
				got, err = tensor.ShardRoot(reversed, lim)
				if err != nil || got.Hex() != vector.LogicalRoot {
					t.Fatalf("shuffled shard root %s, want %s (%v)", got, vector.LogicalRoot, err)
				}
				shardSets++
			default:
				t.Fatalf("unknown vector kind %q", vector.Kind)
			}
		})
	}

	if logicalOK != 3 {
		t.Fatalf("valid logical vectors: %d", logicalOK)
	}
	if logicalRejects < 8 {
		t.Fatalf("logical reject vectors: %d", logicalRejects)
	}
	if layoutOK != 4 {
		t.Fatalf("valid layout vectors: %d", layoutOK)
	}
	if layoutRejects < 8 {
		t.Fatalf("layout reject vectors: %d", layoutRejects)
	}
	if shardSets < 2 {
		t.Fatalf("shard vectors: %d", shardSets)
	}
}

func equalU64s(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
