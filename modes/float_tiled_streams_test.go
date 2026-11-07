package modes_test

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func mustFloatDtype(t *testing.T, name string) modes.FloatDtype {
	t.Helper()
	dtype, err := modes.ParseFloatDtype(name)
	if err != nil {
		t.Fatal(err)
	}
	return dtype
}

func streamsPayload(len int, seed uint64) []byte {
	state := seed | 1
	out := make([]byte, 0, len)
	for i := 0; i < len; i++ {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		out = append(out, byte(state&0xff))
	}
	return out
}

func TestFloatTiledStreamsN1MatchesMode17Payload(t *testing.T) {
	lim := limits.Default()
	cases := []struct {
		dtype modes.FloatDtype
		shape []uint64
		tile  []uint64
		bytes []byte
	}{
		{mustFloatDtype(t, "bf16"), []uint64{8, 8}, []uint64{2, 4}, streamsPayload(8*8*2, 0x51)},
		{mustFloatDtype(t, "fp32"), []uint64{4, 2}, []uint64{2, 2}, streamsPayload(4*2*4, 0x77)},
	}
	for _, c := range cases {
		m17, err := modes.EncodeFloatTiled(c.bytes, c.dtype, c.shape, c.tile, lim)
		if err != nil {
			t.Fatalf("mode 17: %v", err)
		}
		m18, err := modes.EncodeFloatTiledStreams(c.bytes, c.dtype, c.shape, c.tile, 1, lim)
		if err != nil {
			t.Fatalf("mode 18: %v", err)
		}
		f17, err := frame.Parse(m17, lim)
		if err != nil {
			t.Fatal(err)
		}
		f18, err := frame.Parse(m18, lim)
		if err != nil {
			t.Fatal(err)
		}
		if f18.Header.Mode != frame.ModeFloatTiledStreams {
			t.Fatalf("mode %s", f18.Header.Mode)
		}
		d17, err := modes.ParseFloatTiled(f17, lim)
		if err != nil {
			t.Fatal(err)
		}
		d18, err := modes.ParseFloatTiledStreams(f18, lim)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(d17.Payload, d18.Payload) {
			t.Fatal("n_streams=1 payload differs from mode 17")
		}
		for i := range d17.Meta.Tiles {
			a, b := d17.Meta.Tiles[i], d18.Meta.Tiles[i]
			if a.Offset != b.Offset || a.CompressedLen != b.CompressedLen ||
				a.ExponentPlaneLen != b.ExponentPlaneLen || a.MantissaPlaneLen != b.MantissaPlaneLen {
				t.Fatalf("tile %d lengths differ", i)
			}
			if len(b.StreamOffsets) != 2 || b.StreamOffsets[0] != 0 ||
				b.StreamOffsets[1] != a.ExponentPlaneLen {
				t.Fatalf("tile %d offsets %v", i, b.StreamOffsets)
			}
		}
	}
}

func TestFloatTiledStreamsRoundTrip(t *testing.T) {
	lim := limits.Default()
	bytesPayload := streamsPayload(16*16*2, 0x2b)
	for _, n := range []uint64{1, 2, 3, 4, 8, 16} {
		encoded, err := modes.EncodeFloatTiledStreams(
			bytesPayload, mustFloatDtype(t, "bf16"),
			[]uint64{16, 16}, []uint64{8, 8}, n, lim)
		if err != nil {
			t.Fatalf("n=%d encode: %v", n, err)
		}
		parsed, err := frame.Parse(encoded, lim)
		if err != nil {
			t.Fatalf("n=%d parse: %v", n, err)
		}
		meta, err := modes.FloatTiledStreamsMetadata(parsed, lim)
		if err != nil {
			t.Fatalf("n=%d metadata: %v", n, err)
		}
		if meta.NStreams != n {
			t.Fatalf("n=%d declared %d", n, meta.NStreams)
		}
		for _, threads := range []int{1, 4} {
			decoded, err := modes.DecodeFloatTiledStreams(parsed, lim, threads)
			if err != nil {
				t.Fatalf("n=%d threads=%d decode: %v", n, threads, err)
			}
			if !bytes.Equal(decoded, bytesPayload) {
				t.Fatalf("n=%d threads=%d round trip differs", n, threads)
			}
		}
	}
}

func TestFloatTiledStreamsRegion(t *testing.T) {
	lim := limits.Default()
	payload := streamsPayload(16*16*2, 0x41)
	encoded, err := modes.EncodeFloatTiledStreams(
		payload, mustFloatDtype(t, "bf16"),
		[]uint64{16, 16}, []uint64{8, 8}, 4, lim)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := frame.Parse(encoded, lim)
	if err != nil {
		t.Fatal(err)
	}
	descriptor, err := modes.ParseFloatTiledStreams(parsed, lim)
	if err != nil {
		t.Fatal(err)
	}
	whole := tensor.Region{Start: []uint64{0, 0}, End: []uint64{16, 16}}
	out, tiles, streams, err := modes.ReadFloatTiledStreamsRegion(descriptor, whole, lim, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out, payload) || tiles != 4 || streams != 4*2*4 {
		t.Fatalf("whole: tiles=%d streams=%d", tiles, streams)
	}
	one := tensor.Region{Start: []uint64{15, 15}, End: []uint64{16, 16}}
	out, tiles, streams, err = modes.ReadFloatTiledStreamsRegion(descriptor, one, lim, 4)
	if err != nil {
		t.Fatal(err)
	}
	if tiles != 1 || streams != 2*4 {
		t.Fatalf("one: tiles=%d streams=%d", tiles, streams)
	}
	want := payload[15*16*2+15*2 : 15*16*2+15*2+2]
	if !bytes.Equal(out, want) {
		t.Fatal("one-element region differs")
	}
}

type streamsVector struct {
	Name         string   `json:"name"`
	Mode         string   `json:"mode"`
	ModeValue    *uint8   `json:"mode_value"`
	Dtype        string   `json:"dtype"`
	Shape        []uint64 `json:"shape"`
	TileShape    []uint64 `json:"tile_shape"`
	NStreams     *uint64  `json:"n_streams"`
	PlaintextHex string   `json:"plaintext_hex"`
	FrameHex     string   `json:"frame_hex"`
	Expect       struct {
		OK    *bool  `json:"ok"`
		Error string `json:"error"`
	} `json:"expect"`
}

// Every committed mode-18 vector is decoded here, and every valid vector is
// re-encoded from its plaintext. The re-encoded frame must equal the Rust
// frame byte for byte.
func TestFloatTiledStreamsVectors(t *testing.T) {
	lim := limits.Default()
	dir := filepath.Join("..", "conformance", "v1", "float-tiled-streams")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("cannot read %s: %v", dir, err)
	}
	var valid, rejects int
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var vector streamsVector
		if err := json.Unmarshal(raw, &vector); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		frameBytes, err := hex.DecodeString(vector.FrameHex)
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		if vector.Expect.Error != "" {
			_, err := modes.DecodeBytes(context.Background(), frameBytes,
				dependency.NewMapResolver(), lim)
			if err == nil {
				t.Fatalf("%s: expected %s, decoded", vector.Name, vector.Expect.Error)
			}
			if got := string(werrors.ClassOf(err)); got != vector.Expect.Error {
				t.Fatalf("%s: got %s, want %s (%v)", vector.Name, got, vector.Expect.Error, err)
			}
			rejects++
			continue
		}
		plaintext, err := hex.DecodeString(vector.PlaintextHex)
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		decoded, err := modes.DecodeBytes(context.Background(), frameBytes,
			dependency.NewMapResolver(), lim)
		if err != nil {
			t.Fatalf("%s: decode: %v", vector.Name, err)
		}
		if !bytes.Equal(decoded, plaintext) {
			t.Fatalf("%s: plaintext differs", vector.Name)
		}
		dtype, err := modes.ParseFloatDtype(vector.Dtype)
		if err != nil {
			t.Fatalf("%s: %v", vector.Name, err)
		}
		nStreams := uint64(modes.FloatTiledStreamsDefault)
		if vector.NStreams != nil {
			nStreams = *vector.NStreams
		}
		encoded, err := modes.EncodeFloatTiledStreams(
			plaintext, dtype, vector.Shape, vector.TileShape, nStreams, lim)
		if err != nil {
			t.Fatalf("%s: encode: %v", vector.Name, err)
		}
		if !bytes.Equal(encoded, frameBytes) {
			t.Fatalf("%s: Go frame bytes differ from the Rust vector\n got %x\nwant %x",
				vector.Name, encoded, frameBytes)
		}
		valid++
	}
	if valid < 4 {
		t.Fatalf("expected >= 4 valid mode-18 vectors, found %d", valid)
	}
	if rejects < 8 {
		t.Fatalf("expected >= 8 mode-18 rejects, found %d", rejects)
	}
}
