package wcb

import (
	"bytes"
	"encoding/hex"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func TestMapSortedByEncodedKeyBytes(t *testing.T) {
	// "b" encodes as 05 01 62 and "aa" as 05 02 61 61; the shorter encoded
	// key sorts first even though "aa" < "b" as strings.
	value := MapValue(map[string]Value{
		"b":  U64Value(1),
		"aa": U64Value(2),
	})
	encoded, err := EncodeDefault(value)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := hex.DecodeString("08020501620301050261610302")
	if !bytes.Equal(encoded, want) {
		t.Fatalf("got %x want %x", encoded, want)
	}
	decoded, err := DecodeDefault(encoded)
	if err != nil {
		t.Fatal(err)
	}
	reencoded, err := EncodeDefault(decoded)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(reencoded, encoded) {
		t.Fatalf("round-trip changed bytes")
	}
}

func TestScalarRoundTrip(t *testing.T) {
	values := []Value{
		NullValue(),
		BoolValue(true),
		BoolValue(false),
		U64Value(0),
		U64Value(^uint64(0)),
		I64Value(-1),
		I64Value(-1 << 63),
		TimestampValue(1_700_000_000_123_456_789),
		TextValue("héllo 😀"),
		BytesValue([]byte{0, 1, 2, 0xff}),
		ArrayValue([]Value{NullValue(), U64Value(1), BoolValue(false)}),
	}
	for _, value := range values {
		encoded, err := EncodeDefault(value)
		if err != nil {
			t.Fatalf("%v: %v", value, err)
		}
		decoded, err := DecodeDefault(encoded)
		if err != nil {
			t.Fatalf("%x: %v", encoded, err)
		}
		reencoded, err := EncodeDefault(decoded)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(reencoded, encoded) {
			t.Fatalf("round-trip changed bytes: %x -> %x", encoded, reencoded)
		}
	}
}

func TestRejectsNonCanonical(t *testing.T) {
	cases := map[string][]byte{
		"non_minimal_varint": {0x03, 0x80, 0x00},
		"unsorted_map_keys":  {0x08, 0x02, 0x05, 0x01, 'b', 0x03, 0x01, 0x05, 0x01, 'a', 0x03, 0x02},
		"duplicate_map_keys": {0x08, 0x02, 0x05, 0x01, 'a', 0x03, 0x01, 0x05, 0x01, 'a', 0x03, 0x02},
		"non_text_map_key":   {0x08, 0x01, 0x03, 0x01, 0x03, 0x02},
		"trailing_bytes":     {0x00, 0x00},
		"unknown_tag":        {0x0a},
		"invalid_utf8":       {0x05, 0x02, 0xff, 0xfe},
	}
	for name, input := range cases {
		if _, err := DecodeDefault(input); err == nil || werrors.ClassOf(err) != werrors.NonCanonical {
			t.Fatalf("%s: got %v, want NonCanonical", name, err)
		}
	}
	if _, err := DecodeDefault([]byte{0x05, 0x02, 'a'}); werrors.ClassOf(err) != werrors.Malformed {
		t.Fatalf("truncated text: got %v, want Malformed", err)
	}
}

func TestLimits(t *testing.T) {
	deep := make([]byte, 0, 131)
	for i := 0; i < 65; i++ {
		deep = append(deep, 0x07, 0x01)
	}
	deep = append(deep, 0x00)
	if _, err := DecodeDefault(deep); werrors.ClassOf(err) != werrors.LimitExceeded {
		t.Fatalf("depth: got %v, want LimitExceeded", err)
	}

	lim := limits.Default()
	lim.MaxStructuredItems = 2
	if _, err := Decode([]byte{0x07, 0x03, 0x00, 0x00, 0x00}, lim); werrors.ClassOf(err) != werrors.LimitExceeded {
		t.Fatalf("items: got %v, want LimitExceeded", err)
	}

	lim = limits.Default()
	lim.MaxStringBytes = 1
	if _, err := Decode([]byte{0x05, 0x02, 'a', 'b'}, lim); werrors.ClassOf(err) != werrors.LimitExceeded {
		t.Fatalf("string: got %v, want LimitExceeded", err)
	}
}
