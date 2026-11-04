package tensor_test

import (
	"bytes"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func pattern(length int, state uint64) []byte {
	out := make([]byte, 0, length)
	for i := 0; i < length; i++ {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		out = append(out, byte(state&0xff))
	}
	return out
}

func fixture() (string, []uint64, []byte, limits.Limits) {
	lim := limits.Default()
	shape := []uint64{4, 3, 2}
	return "bf16", shape, pattern(4*3*2*2, 0x5eed), lim
}

func sliceRegion(t *testing.T, logical []byte, shape []uint64, region tensor.Region) []byte {
	t.Helper()
	physical, err := tensor.NewRowMajor("bf16", shape, logical, limits.Default())
	if err != nil {
		t.Fatalf("row-major: %v", err)
	}
	read, err := physical.ReadRegion(region)
	if err != nil {
		t.Fatalf("read region: %v", err)
	}
	return read.Bytes
}

func TestFourLayoutsProduceOneLogicalRoot(t *testing.T) {
	dtype, shape, rowMajor, lim := fixture()
	expected, err := tensor.LogicalRoot(dtype, shape, rowMajor, lim)
	if err != nil {
		t.Fatalf("logical root: %v", err)
	}

	row, err := tensor.NewRowMajor(dtype, shape, rowMajor, lim)
	if err != nil {
		t.Fatalf("row-major: %v", err)
	}
	rootOf := func(name string, physical *tensor.Physical) {
		t.Helper()
		got, err := physical.LogicalRoot(lim)
		if err != nil {
			t.Fatalf("%s root: %v", name, err)
		}
		if got != expected {
			t.Fatalf("%s root %s != %s", name, got, expected)
		}
	}
	rootOf("row-major", row)

	column, err := tensor.ColumnMajorBytes(dtype, shape, rowMajor, lim)
	if err != nil {
		t.Fatalf("column-major bytes: %v", err)
	}
	columnPhysical, err := tensor.NewColumnMajor(dtype, shape, column, lim)
	if err != nil {
		t.Fatalf("column-major: %v", err)
	}
	assembled, err := columnPhysical.Assemble()
	if err != nil || !bytes.Equal(assembled, rowMajor) {
		t.Fatalf("column-major assemble mismatch: %v", err)
	}
	rootOf("column-major", columnPhysical)

	blockShape := []uint64{2, 2, 2}
	blocks, err := tensor.BlockedBytes(dtype, shape, blockShape, rowMajor, lim)
	if err != nil {
		t.Fatalf("blocked bytes: %v", err)
	}
	blocked, err := tensor.NewBlocked(dtype, shape, blockShape, blocks, lim)
	if err != nil {
		t.Fatalf("blocked: %v", err)
	}
	assembled, err = blocked.Assemble()
	if err != nil || !bytes.Equal(assembled, rowMajor) {
		t.Fatalf("blocked assemble mismatch: %v", err)
	}
	rootOf("blocked", blocked)

	shards, err := tensor.SplitShards(dtype, shape, 0, 4, rowMajor, lim)
	if err != nil {
		t.Fatalf("split shards: %v", err)
	}
	sharded, err := tensor.NewSharded(dtype, shape, 0, shards, lim)
	if err != nil {
		t.Fatalf("sharded: %v", err)
	}
	assembled, err = sharded.Assemble()
	if err != nil || !bytes.Equal(assembled, rowMajor) {
		t.Fatalf("sharded assemble mismatch: %v", err)
	}
	rootOf("sharded", sharded)

	decoded := make([]tensor.DecodedShard, 0, len(shards))
	for index, payload := range shards {
		descriptor := tensor.ShardedDescriptor(dtype, shape, 0, 4, uint32(index))
		container, err := tensor.EncodeShard(descriptor, payload, lim)
		if err != nil {
			t.Fatalf("encode shard %d: %v", index, err)
		}
		shard, err := tensor.DecodeShard(container, lim)
		if err != nil {
			t.Fatalf("decode shard %d: %v", index, err)
		}
		decoded = append(decoded, shard)
	}
	got, err := tensor.ShardRoot(decoded, lim)
	if err != nil || got != expected {
		t.Fatalf("shard root %s != %s (%v)", got, expected, err)
	}
}

func TestRegionReadsMatchLogicalSlice(t *testing.T) {
	dtype, shape, rowMajor, lim := fixture()
	column, err := tensor.ColumnMajorBytes(dtype, shape, rowMajor, lim)
	if err != nil {
		t.Fatal(err)
	}
	columnPhysical, err := tensor.NewColumnMajor(dtype, shape, column, lim)
	if err != nil {
		t.Fatal(err)
	}
	blockShape := []uint64{2, 2, 2}
	blocks, err := tensor.BlockedBytes(dtype, shape, blockShape, rowMajor, lim)
	if err != nil {
		t.Fatal(err)
	}
	blocked, err := tensor.NewBlocked(dtype, shape, blockShape, blocks, lim)
	if err != nil {
		t.Fatal(err)
	}
	shards, err := tensor.SplitShards(dtype, shape, 0, 4, rowMajor, lim)
	if err != nil {
		t.Fatal(err)
	}
	sharded, err := tensor.NewSharded(dtype, shape, 0, shards, lim)
	if err != nil {
		t.Fatal(err)
	}
	row, err := tensor.NewRowMajor(dtype, shape, rowMajor, lim)
	if err != nil {
		t.Fatal(err)
	}

	regions := []tensor.Region{
		tensor.FullRegion(shape),
		tensor.NewRegion([]uint64{0, 0, 0}, []uint64{1, 1, 1}),
		tensor.NewRegion([]uint64{3, 2, 1}, []uint64{4, 3, 2}),
		tensor.NewRegion([]uint64{0, 0, 1}, []uint64{4, 3, 2}),
		tensor.NewRegion([]uint64{1, 0, 0}, []uint64{3, 3, 2}),
	}
	for _, region := range regions {
		expected := sliceRegion(t, rowMajor, shape, region)
		for name, physical := range map[string]*tensor.Physical{
			"row": row, "column": columnPhysical, "blocked": blocked, "sharded": sharded,
		} {
			read, err := physical.ReadRegion(region)
			if err != nil {
				t.Fatalf("%s region %v: %v", name, region, err)
			}
			if !bytes.Equal(read.Bytes, expected) {
				t.Fatalf("%s region %v bytes differ", name, region)
			}
			if !equalU64(read.Shape, region.Extent()) {
				t.Fatalf("%s region shape %v != %v", name, read.Shape, region.Extent())
			}
		}
	}

	one := tensor.NewRegion([]uint64{0, 0, 0}, []uint64{1, 3, 2})
	if got := shardedReadUnits(t, sharded, one); got != 1 {
		t.Fatalf("one-shard region touched %d", got)
	}
	two := tensor.NewRegion([]uint64{0, 0, 0}, []uint64{2, 3, 2})
	if got := shardedReadUnits(t, sharded, two); got != 2 {
		t.Fatalf("two-shard region touched %d", got)
	}
	if got := shardedReadUnits(t, sharded, tensor.FullRegion(shape)); got != 4 {
		t.Fatalf("full region touched %d", got)
	}

	single := tensor.NewRegion([]uint64{0, 0, 0}, []uint64{1, 1, 1})
	read, err := blocked.ReadRegion(single)
	if err != nil || read.UnitsTouched != 1 {
		t.Fatalf("single block region touched %d (%v)", read.UnitsTouched, err)
	}
	read, err = blocked.ReadRegion(tensor.FullRegion(shape))
	if err != nil || read.UnitsTouched != 4 {
		t.Fatalf("full blocked region touched %d (%v)", read.UnitsTouched, err)
	}
}

func shardedReadUnits(t *testing.T, physical *tensor.Physical, region tensor.Region) uint64 {
	t.Helper()
	read, err := physical.ReadRegion(region)
	if err != nil {
		t.Fatalf("read region: %v", err)
	}
	return read.UnitsTouched
}

func TestShuffledShardsAreReorderedByIndex(t *testing.T) {
	dtype, shape, rowMajor, lim := fixture()
	shards, err := tensor.SplitShards(dtype, shape, 0, 4, rowMajor, lim)
	if err != nil {
		t.Fatal(err)
	}
	decoded := make([]tensor.DecodedShard, 0, 4)
	for index, payload := range shards {
		decoded = append(decoded, tensor.DecodedShard{
			Descriptor: tensor.ShardedDescriptor(dtype, shape, 0, 4, uint32(index)),
			Bytes:      append([]byte(nil), payload...),
		})
	}
	reverse(decoded)
	expected, _ := tensor.LogicalRoot(dtype, shape, rowMajor, lim)
	got, err := tensor.ShardRoot(decoded, lim)
	if err != nil || got != expected {
		t.Fatalf("shuffled shard root %s != %s (%v)", got, expected, err)
	}
}

func reverse(shards []tensor.DecodedShard) {
	for i, j := 0, len(shards)-1; i < j; i, j = i+1, j-1 {
		shards[i], shards[j] = shards[j], shards[i]
	}
}

func TestDuplicateMissingAndInconsistentShardsAreRejected(t *testing.T) {
	dtype, shape, rowMajor, lim := fixture()
	shards, err := tensor.SplitShards(dtype, shape, 0, 4, rowMajor, lim)
	if err != nil {
		t.Fatal(err)
	}
	makeShard := func(index uint32, payload []byte) tensor.DecodedShard {
		return tensor.DecodedShard{
			Descriptor: tensor.ShardedDescriptor(dtype, shape, 0, 4, index),
			Bytes:      append([]byte(nil), payload...),
		}
	}
	good := make([]tensor.DecodedShard, 0, 4)
	for index, payload := range shards {
		good = append(good, makeShard(uint32(index), payload))
	}
	if _, err := tensor.ShardRoot(good, lim); err != nil {
		t.Fatalf("good shards: %v", err)
	}

	duplicate := append([]tensor.DecodedShard(nil), good...)
	duplicate[3] = makeShard(0, shards[0])
	if _, err := tensor.ShardRoot(duplicate, lim); werrors.ClassOf(err) != werrors.NonCanonical {
		t.Fatalf("duplicate: got %v", err)
	}

	if _, err := tensor.ShardRoot(good[:3], lim); werrors.ClassOf(err) != werrors.Malformed {
		t.Fatalf("missing: got %v", err)
	}

	wrongShape := append([]tensor.DecodedShard(nil), good...)
	wrongShape[1].Descriptor = tensor.ShardedDescriptor(dtype, []uint64{2, 3, 4}, 0, 4, 1)
	if _, err := tensor.ShardRoot(wrongShape, lim); err == nil {
		t.Fatal("inconsistent shape accepted")
	}

	wrongLen := append([]tensor.DecodedShard(nil), good...)
	wrongLen[2].Bytes = wrongLen[2].Bytes[:len(wrongLen[2].Bytes)-1]
	if _, err := tensor.ShardRoot(wrongLen, lim); err == nil {
		t.Fatal("wrong shard length accepted")
	}
}

func TestShapeAndLayoutRejects(t *testing.T) {
	lim := limits.Default()
	check := func(name string, err error, want werrors.Class) {
		t.Helper()
		if err == nil {
			t.Fatalf("%s: expected %s, got nil", name, want)
		}
		if got := werrors.ClassOf(err); got != want {
			t.Fatalf("%s: expected %s, got %s (%v)", name, want, got, err)
		}
	}
	_, err := tensor.ValidateShape(nil, lim)
	check("rank zero", err, werrors.NonCanonical)
	_, err = tensor.ValidateShape([]uint64{2, 0, 3}, lim)
	check("zero dim", err, werrors.NonCanonical)
	_, err = tensor.ValidateShape([]uint64{4294967296, 4294967296, 4294967296}, lim)
	check("overflow", err, werrors.LimitExceeded)
	_, err = tensor.ValidateShape([]uint64{100000, 100000}, lim)
	check("count", err, werrors.LimitExceeded)
	_, err = tensor.ValidateShape([]uint64{1, 1, 1, 1, 1, 1, 1, 1, 1}, lim)
	check("rank", err, werrors.LimitExceeded)
	_, err = tensor.CheckPayload("bf16", []uint64{2, 3}, make([]byte, 11), lim)
	check("payload length", err, werrors.Malformed)

	check("not divisible", tensor.ShardedDescriptor("bf16", []uint64{2, 3, 4}, 0, 3, 0).Validate(lim), werrors.NonCanonical)
	check("index", tensor.ShardedDescriptor("bf16", []uint64{4, 3, 2}, 0, 4, 4).Validate(lim), werrors.NonCanonical)
	check("axis", tensor.ShardedDescriptor("bf16", []uint64{4, 3, 2}, 3, 1, 0).Validate(lim), werrors.NonCanonical)
	check("zero shards", tensor.ShardedDescriptor("bf16", []uint64{4, 3, 2}, 0, 0, 0).Validate(lim), werrors.NonCanonical)
	check("block rank", tensor.BlockedDescriptor("bf16", []uint64{4, 3, 2}, []uint64{2, 2}).Validate(lim), werrors.NonCanonical)
	check("zero block", tensor.BlockedDescriptor("bf16", []uint64{4, 3, 2}, []uint64{2, 2, 0}).Validate(lim), werrors.NonCanonical)
	check("column rank zero", tensor.ColumnMajorDescriptor("bf16", nil).Validate(lim), werrors.NonCanonical)
}

func TestLogicalRecordRoundTrips(t *testing.T) {
	dtype, shape, rowMajor, lim := fixture()
	record, err := tensor.NewLogicalRecord(dtype, shape, rowMajor, lim)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := record.CanonicalBytes(lim)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := tensor.ParseLogicalRecord(canonical, lim)
	if err != nil {
		t.Fatal(err)
	}
	if !decoded.Equal(record) {
		t.Fatal("logical record did not round-trip")
	}
	root, err := record.Root(lim)
	if err != nil {
		t.Fatal(err)
	}
	decodedRoot, err := decoded.Root(lim)
	if err != nil || decodedRoot != root {
		t.Fatalf("record roots differ: %v", err)
	}

	tampered := append([]byte(nil), canonical...)
	tampered[len(tampered)-1] ^= 0xff
	if _, err := tensor.ParseLogicalRecord(tampered, lim); err == nil {
		t.Fatal("tampered record accepted")
	}
}

func TestShardContainerRoundTripsAndBindsPayload(t *testing.T) {
	dtype, shape, rowMajor, lim := fixture()
	payload, err := tensor.ShardBytes(dtype, shape, 0, 4, 2, rowMajor, lim)
	if err != nil {
		t.Fatal(err)
	}
	descriptor := tensor.ShardedDescriptor(dtype, shape, 0, 4, 2)
	container, err := tensor.EncodeShard(descriptor, payload, lim)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := tensor.DecodeShard(container, lim)
	if err != nil {
		t.Fatal(err)
	}
	if !decoded.Descriptor.Equal(descriptor) || !bytes.Equal(decoded.Bytes, payload) {
		t.Fatal("shard container did not round-trip")
	}

	corrupted := append([]byte(nil), container...)
	corrupted[len(corrupted)-1] ^= 0x01
	if _, err := tensor.DecodeShard(corrupted, lim); werrors.ClassOf(err) != werrors.Corrupt {
		t.Fatalf("corrupt payload: got %v", err)
	}
	if _, err := tensor.DecodeShard(container[:len(container)-1], lim); err == nil {
		t.Fatal("truncated container accepted")
	}
}

func equalU64(a, b []uint64) bool {
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
