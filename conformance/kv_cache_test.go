package conformance_test

import (
	"bytes"
	"path/filepath"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

type kvCachePageVector struct {
	PageIndex   uint64 `json:"page_index"`
	SeqStart    uint64 `json:"seq_start"`
	SeqEnd      uint64 `json:"seq_end"`
	PageRows    uint64 `json:"page_rows"`
	PageBytes   uint64 `json:"page_bytes"`
	PayloadLen  uint64 `json:"payload_len"`
	ElementsHex string `json:"elements_hex"`
	FrameHex    string `json:"frame_hex"`
}

type kvCacheVector struct {
	Name        string              `json:"name"`
	Schema      string              `json:"schema"`
	Model       string              `json:"model"`
	Layer       uint32              `json:"layer"`
	Session     string              `json:"session"`
	Dtype       string              `json:"dtype"`
	HeadCount   uint32              `json:"head_count"`
	HeadDim     uint32              `json:"head_dim"`
	PageCount   uint64              `json:"page_count"`
	Pages       []kvCachePageVector `json:"pages"`
	FrameHex    string              `json:"frame_hex"`
	ExpectError string              `json:"expect_error"`
}

func TestKvCacheVectors(t *testing.T) {
	files := jsonFiles(t, filepath.Join("v1", "kv-cache"))
	if len(files) != 11 {
		t.Fatalf("expected 11 KV_CACHE vectors, found %d", len(files))
	}
	lim := limits.Default()
	validPages := 0
	rejects := 0
	for _, path := range files {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			var vector kvCacheVector
			loadJSON(t, path, &vector)
			if vector.ExpectError != "" {
				frameBytes := mustHex(t, "frame_hex", vector.FrameHex)
				parsed, err := frame.Parse(frameBytes, lim)
				if err != nil {
					t.Fatalf("parse: %v", err)
				}
				if _, err := modes.DecodeKvPage(parsed, lim); err == nil {
					t.Fatalf("expected %s, decode succeeded", vector.ExpectError)
				} else if class := string(werrors.ClassOf(err)); class != vector.ExpectError {
					t.Fatalf("expected class %s, got %s (%v)", vector.ExpectError, class, err)
				}
				rejects++
				return
			}
			dtype, err := modes.ParseKvDtype(vector.Dtype)
			if err != nil {
				t.Fatalf("dtype: %v", err)
			}
			for _, page := range vector.Pages {
				frameBytes := mustHex(t, "frame_hex", page.FrameHex)
				elements := mustHex(t, "elements_hex", page.ElementsHex)
				parsed, err := frame.Parse(frameBytes, lim)
				if err != nil {
					t.Fatalf("parse page: %v", err)
				}
				decoded, err := modes.DecodeKvPage(parsed, lim)
				if err != nil {
					t.Fatalf("decode page: %v", err)
				}
				if !bytes.Equal(decoded.Bytes, elements) {
					t.Fatalf("page %d bytes differ", page.PageIndex)
				}
				if decoded.Meta.PageBytes != page.PageBytes ||
					decoded.Meta.SeqStart != page.SeqStart ||
					decoded.Meta.SeqEnd != page.SeqEnd {
					t.Fatalf("page %d descriptor differs", page.PageIndex)
				}
				input := modes.KvCacheInput{
					Model: vector.Model, Layer: vector.Layer, Session: vector.Session,
					PageIndex: page.PageIndex, PageCount: vector.PageCount,
					SeqStart: page.SeqStart, SeqEnd: page.SeqEnd, Dtype: dtype,
					HeadCount: vector.HeadCount, HeadDim: vector.HeadDim,
				}
				reencoded, err := modes.EncodeKvPage(input, elements, lim)
				if err != nil {
					t.Fatalf("re-encode: %v", err)
				}
				if !bytes.Equal(reencoded, frameBytes) {
					t.Fatalf("Go re-encode is not byte-identical to the committed Rust frame")
				}
				validPages++
			}
		})
	}
	if validPages != 5 {
		t.Fatalf("expected 5 valid pages, saw %d", validPages)
	}
	if rejects != 8 {
		t.Fatalf("expected 8 rejects, saw %d", rejects)
	}
}

func TestKvPageSetDecodesOnePage(t *testing.T) {
	lim := limits.Default()
	dtype, err := modes.ParseKvDtype("bf16")
	if err != nil {
		t.Fatalf("dtype: %v", err)
	}
	frames := make([][]byte, 0, 8)
	expected := make([][]byte, 0, 8)
	for index := uint64(0); index < 8; index++ {
		elements := make([]byte, 64)
		for offset := range elements {
			elements[offset] = byte(index*64 + uint64(offset))
		}
		input := modes.KvCacheInput{
			Model: "m", Layer: 0, Session: "s", PageIndex: index, PageCount: 8,
			SeqStart: index * 4, SeqEnd: index*4 + 4, Dtype: dtype,
			HeadCount: 2, HeadDim: 4,
		}
		encoded, err := modes.EncodeKvPage(input, elements, lim)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		frames = append(frames, encoded)
		expected = append(expected, elements)
	}
	for index := 0; index < 7; index++ {
		frames[index] = frames[index][:4]
	}
	set := modes.NewKvPageSet(frames)
	page, err := set.DecodePage(7, lim)
	if err != nil {
		t.Fatalf("decode page 7: %v", err)
	}
	if !bytes.Equal(page.Bytes, expected[7]) {
		t.Fatalf("page 7 bytes differ")
	}
	if set.DecodedPages() != 1 {
		t.Fatalf("expected 1 decoded page, saw %d", set.DecodedPages())
	}
	slice, err := set.ReadPageRange(7, 8, 24, lim)
	if err != nil {
		t.Fatalf("range: %v", err)
	}
	if !bytes.Equal(slice, expected[7][8:24]) {
		t.Fatalf("range bytes differ")
	}
	if set.DecodedPages() != 2 {
		t.Fatalf("expected 2 decoded pages, saw %d", set.DecodedPages())
	}
}
