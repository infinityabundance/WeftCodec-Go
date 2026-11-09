package transport

import (
	"bytes"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func raw(t *testing.T, text string) []byte {
	t.Helper()
	b, err := modes.EncodeRaw([]byte(text), 0, limits.Default())
	if err != nil {
		t.Fatalf("encode raw: %v", err)
	}
	return b
}

func TestRoundTripAndAlignment(t *testing.T) {
	lim := limits.Default()
	a := raw(t, "alpha")
	b := raw(t, "beta")
	items := []Item{{Bytes: a, Role: RolePayload}, {Bytes: b, Role: RoleDependency}}
	container, err := Create(items, 512, lim)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if len(container)%512 != 0 {
		t.Fatalf("container length %d is not 512-aligned", len(container))
	}
	again, err := Create(items, 512, lim)
	if err != nil {
		t.Fatalf("create twice: %v", err)
	}
	if !bytes.Equal(container, again) {
		t.Fatal("encoder is not deterministic")
	}
	parsed, err := Verify(container, lim)
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if len(parsed.Sections) != 2 {
		t.Fatalf("sections %d, want 2", len(parsed.Sections))
	}
	if parsed.Sections[0].Offset%512 != 0 || parsed.Sections[1].Offset%512 != 0 {
		t.Fatal("section offset is misaligned")
	}
}

func TestMisalignedOffsetRejected(t *testing.T) {
	lim := limits.Default()
	container, err := Create([]Item{{Bytes: raw(t, "x"), Role: RolePayload}}, 4096, lim)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	container[HeaderLen]++
	if _, err := ParseMetadata(container, lim); err == nil {
		t.Fatal("misaligned offset was accepted")
	} else if string(werrors.ClassOf(err)) != "Malformed" {
		t.Fatalf("class %s, want Malformed", werrors.ClassOf(err))
	}
}

func TestWrongRootRejected(t *testing.T) {
	lim := limits.Default()
	container, err := Create([]Item{{Bytes: raw(t, "x"), Role: RolePayload}}, 4096, lim)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	container[HeaderLen+24] ^= 0x01
	if _, err := ParseMetadata(container, lim); err == nil {
		t.Fatal("wrong section root was accepted")
	} else if string(werrors.ClassOf(err)) != "Corrupt" {
		t.Fatalf("class %s, want Corrupt", werrors.ClassOf(err))
	}
}
