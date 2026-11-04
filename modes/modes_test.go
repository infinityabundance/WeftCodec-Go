package modes

import (
	"bytes"
	"context"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func TestRoundTripAllModes(t *testing.T) {
	ctx := context.Background()
	lim := limits.Default()
	payloads := [][]byte{
		nil,
		{},
		{0xff, 0xfe, 0xfd, 0x80, 0x00, 0xc3, 0x28}, // deliberately invalid UTF-8
		pattern(5000, 0x9999111122223333),
		repeat([]byte("weft"), 20000),
	}
	for _, payload := range payloads {
		payload := payload
		t.Run("raw", func(t *testing.T) {
			for _, chunkSize := range []int{0, 1024} {
				encoded, err := EncodeRaw(payload, chunkSize, lim)
				if err != nil {
					t.Fatal(err)
				}
				decoded, err := DecodeBytes(ctx, encoded, resolverWith(), lim)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(decoded, payload) {
					t.Fatalf("raw chunk=%d mismatch", chunkSize)
				}
			}
		})
		t.Run("zstd", func(t *testing.T) {
			for _, chunkSize := range []int{0, 1024} {
				encoded, err := EncodeZstd(payload, 3, chunkSize, lim)
				if err != nil {
					t.Fatal(err)
				}
				decoded, err := DecodeBytes(ctx, encoded, resolverWith(), lim)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(decoded, payload) {
					t.Fatalf("zstd chunk=%d mismatch", chunkSize)
				}
			}
		})
		t.Run("reference", func(t *testing.T) {
			base := append([]byte(nil), payload...)
			encoded, err := EncodeReference(payload, base, lim)
			if err != nil {
				t.Fatal(err)
			}
			decoded, err := DecodeBytes(ctx, encoded, resolverWith(base), lim)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(decoded, payload) {
				t.Fatal("reference mismatch")
			}
		})
	}

	base := pattern(4096, 0x1234)
	target := append([]byte(nil), base[:100]...)
	target = append(target, []byte("INSERTED-REVISION-2")...)
	target = append(target, base[150:500]...)
	target = append(target, []byte("REPLACED-BLOCK")...)
	target = append(target, base[700:3000]...)
	target = append(target, base[3200:]...)
	t.Run("delta", func(t *testing.T) {
		encoded, err := EncodeDelta(target, base, lim)
		if err != nil {
			t.Fatal(err)
		}
		decoded, err := DecodeBytes(ctx, encoded, resolverWith(base), lim)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(decoded, target) {
			t.Fatal("delta mismatch")
		}
	})
	t.Run("delta-near-duplicate", func(t *testing.T) {
		near := append([]byte(nil), base...)
		near[1000] ^= 0x01
		near[2000] ^= 0x02
		near[3000] ^= 0x04
		encoded, err := EncodeDelta(near, base, lim)
		if err != nil {
			t.Fatal(err)
		}
		parsed, err := frame.Parse(encoded, lim)
		if err != nil {
			t.Fatal(err)
		}
		if parsed.Header.BodyLen >= uint64(len(near))/2 {
			t.Fatalf("near-duplicate delta body is %d bytes for %d input", parsed.Header.BodyLen, len(near))
		}
		decoded, err := DecodeBytes(ctx, encoded, resolverWith(base), lim)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(decoded, near) {
			t.Fatal("near-duplicate delta mismatch")
		}
	})
	t.Run("delta-empty", func(t *testing.T) {
		encoded, err := EncodeDelta(nil, nil, lim)
		if err != nil {
			t.Fatal(err)
		}
		decoded, err := DecodeBytes(ctx, encoded, resolverWith(nil), lim)
		if err != nil {
			t.Fatal(err)
		}
		if len(decoded) != 0 {
			t.Fatal("empty delta mismatch")
		}
	})
	t.Run("dictionary", func(t *testing.T) {
		dict := repeat([]byte("weft codec dictionary line alpha beta gamma\n"), 1024)
		data := repeat([]byte("weft codec dictionary line alpha beta gamma\n"), 300)
		encoded, err := EncodeDictionary(data, dict, 3, lim)
		if err != nil {
			t.Fatal(err)
		}
		decoded, err := DecodeBytes(ctx, encoded, resolverWith(dict), lim)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(decoded, data) {
			t.Fatal("dictionary mismatch")
		}
	})
	t.Run("structured", func(t *testing.T) {
		value := wcb.MapValue(map[string]wcb.Value{
			"name":   wcb.TextValue("vector"),
			"count":  wcb.U64Value(42),
			"signed": wcb.I64Value(-7),
			"flags":  wcb.ArrayValue([]wcb.Value{wcb.BoolValue(true), wcb.NullValue()}),
		})
		encoded, err := EncodeStructured(value, lim)
		if err != nil {
			t.Fatal(err)
		}
		decoded, err := DecodeBytes(ctx, encoded, resolverWith(), lim)
		if err != nil {
			t.Fatal(err)
		}
		want, err := wcb.Encode(value, lim)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(decoded, want) {
			t.Fatal("structured mismatch")
		}
	})
}

func TestZstdOutputLengthBound(t *testing.T) {
	lim := limits.Default()
	body, err := compressZstd([]byte("0123456789"), 3)
	if err != nil {
		t.Fatal(err)
	}
	build := func(decodedLen uint64) []byte {
		parts := frame.PayloadParts(
			frame.ModeZstd, 0, 3, decodedLen,
			hash.Domain(hash.DomainChunk, make([]byte, decodedLen)), body)
		encoded, err := frame.Encode(parts, lim)
		if err != nil {
			t.Fatal(err)
		}
		return encoded
	}
	_, err = DecodeBytes(context.Background(), build(100), resolverWith(), lim)
	requireClass(t, err, werrors.Corrupt)
	_, err = DecodeBytes(context.Background(), build(5), resolverWith(), lim)
	requireClass(t, err, werrors.Corrupt)
}

func TestStructuredBodyValidated(t *testing.T) {
	lim := limits.Default()
	body := []byte{0x0a}
	parts := frame.Parts{
		FormatMajor: frame.FormatMajor,
		FormatMinor: frame.FormatMinor,
		Domain:      frame.DomainStructured,
		Mode:        frame.ModeStructured,
		DecodedLen:  uint64(len(body)),
		ContentHash: hash.Domain(hash.DomainChunk, body),
		Body:        body,
	}
	encoded, err := frame.Encode(parts, lim)
	if err != nil {
		t.Fatal(err)
	}
	_, err = DecodeBytes(context.Background(), encoded, resolverWith(), lim)
	requireClass(t, err, werrors.NonCanonical)
}

func TestContextCancellation(t *testing.T) {
	lim := limits.Default()
	encoded, err := EncodeRaw([]byte("payload"), 0, lim)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = DecodeBytes(ctx, encoded, resolverWith(), lim)
	requireClass(t, err, werrors.Io)
}
