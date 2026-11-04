package conformance_test

import (
	"bytes"
	"path/filepath"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

type activationChunkVector struct {
	ChunkIndex  uint64 `json:"chunk_index"`
	DigestRoot  string `json:"digest_root"`
	PayloadLen  uint64 `json:"payload_len"`
	ElementsHex string `json:"elements_hex"`
	FrameHex    string `json:"frame_hex"`
}

type activationStreamVector struct {
	Name         string                  `json:"name"`
	Schema       string                  `json:"schema"`
	Stream       string                  `json:"stream"`
	Step         uint64                  `json:"step"`
	Rank         uint32                  `json:"rank"`
	World        uint32                  `json:"world"`
	ChunkCount   uint64                  `json:"chunk_count"`
	Dtype        string                  `json:"dtype"`
	Shape        []uint64                `json:"shape"`
	ElementCount uint64                  `json:"element_count"`
	Chunks       []activationChunkVector `json:"chunks"`
	StreamRoot   string                  `json:"stream_root"`
	FrameHex     string                  `json:"frame_hex"`
	ExpectError  string                  `json:"expect_error"`
}

func TestActivationStreamVectors(t *testing.T) {
	files := jsonFiles(t, filepath.Join("v1", "activation-stream"))
	if len(files) != 11 {
		t.Fatalf("expected 11 ACTIVATION_STREAM vectors, found %d", len(files))
	}
	lim := limits.Default()
	rejects := 0
	for _, path := range files {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			var vector activationStreamVector
			loadJSON(t, path, &vector)
			if vector.ExpectError != "" {
				frameBytes := mustHex(t, "frame_hex", vector.FrameHex)
				parsed, err := frame.Parse(frameBytes, lim)
				if err != nil {
					t.Fatalf("parse: %v", err)
				}
				if _, err := modes.DecodeActivationChunk(parsed, lim); err == nil {
					t.Fatalf("expected %s, decode succeeded", vector.ExpectError)
				} else if class := string(werrors.ClassOf(err)); class != vector.ExpectError {
					t.Fatalf("expected class %s, got %s (%v)", vector.ExpectError, class, err)
				}
				rejects++
				return
			}
			stream, err := modes.ParseStreamKind(vector.Stream)
			if err != nil {
				t.Fatalf("stream: %v", err)
			}
			dtype, err := modes.ParseStreamDtype(vector.Dtype)
			if err != nil {
				t.Fatalf("dtype: %v", err)
			}
			head := modes.ActivationStreamHead{
				Stream: stream, Step: vector.Step, Rank: vector.Rank, World: vector.World,
				ChunkCount: vector.ChunkCount, Dtype: dtype, Shape: vector.Shape,
			}
			frames := make([][]byte, 0, len(vector.Chunks))
			for _, chunk := range vector.Chunks {
				frameBytes := mustHex(t, "frame_hex", chunk.FrameHex)
				elements := mustHex(t, "elements_hex", chunk.ElementsHex)
				parsed, err := frame.Parse(frameBytes, lim)
				if err != nil {
					t.Fatalf("parse chunk: %v", err)
				}
				decoded, err := modes.DecodeActivationChunk(parsed, lim)
				if err != nil {
					t.Fatalf("decode chunk: %v", err)
				}
				if !bytes.Equal(decoded.Bytes, elements) {
					t.Fatalf("chunk %d bytes differ", chunk.ChunkIndex)
				}
				if decoded.Meta.DigestRoot.Hex() != chunk.DigestRoot {
					t.Fatalf("chunk %d digest differs", chunk.ChunkIndex)
				}
				if got := hash.Domain(tensor.DomainTensorValues, elements); got != decoded.Meta.DigestRoot {
					t.Fatalf("chunk %d digest_root is wrong", chunk.ChunkIndex)
				}
				reencoded, err := modes.EncodeActivationChunk(head, chunk.ChunkIndex, elements, lim)
				if err != nil {
					t.Fatalf("re-encode: %v", err)
				}
				if !bytes.Equal(reencoded, frameBytes) {
					t.Fatalf("Go re-encode is not byte-identical to the committed Rust frame")
				}
				frames = append(frames, frameBytes)
			}
			assembly, err := modes.AssembleStream(frames, lim)
			if err != nil {
				t.Fatalf("assemble: %v", err)
			}
			if assembly.StreamRoot.Hex() != vector.StreamRoot {
				t.Fatalf("stream root %s, want %s", assembly.StreamRoot.Hex(), vector.StreamRoot)
			}
		})
	}
	if rejects != 8 {
		t.Fatalf("expected 8 rejects, saw %d", rejects)
	}
}

func TestActivationAssemblyOrderAndDuplicates(t *testing.T) {
	lim := limits.Default()
	dtype, err := modes.ParseStreamDtype("bf16")
	if err != nil {
		t.Fatalf("dtype: %v", err)
	}
	head := modes.ActivationStreamHead{
		Stream: modes.ActivationStreamKind, Step: 11, Rank: 0, World: 2,
		ChunkCount: 3, Dtype: dtype, Shape: []uint64{2, 4},
	}
	chunks := make([][]byte, 3)
	frames := make([][]byte, 3)
	for index := 0; index < 3; index++ {
		elements := make([]byte, 16)
		for offset := range elements {
			elements[offset] = byte(index*16 + offset)
		}
		chunks[index] = elements
		encoded, err := modes.EncodeActivationChunk(head, uint64(index), elements, lim)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		frames[index] = encoded
	}
	assembly, err := modes.AssembleStream([][]byte{frames[2], frames[0], frames[1]}, lim)
	if err != nil {
		t.Fatalf("assemble: %v", err)
	}
	var expected []byte
	for _, chunk := range chunks {
		expected = append(expected, chunk...)
	}
	if !bytes.Equal(assembly.Bytes, expected) {
		t.Fatalf("assembled bytes differ")
	}
	if assembly.ChunksDecoded != 3 {
		t.Fatalf("expected 3 decoded chunks, saw %d", assembly.ChunksDecoded)
	}
	duplicate := [][]byte{frames[0], frames[0], frames[2]}
	if _, err := modes.AssembleStream(duplicate, lim); err == nil {
		t.Fatalf("expected duplicate rejection")
	} else if werrors.ClassOf(err) != werrors.NonCanonical {
		t.Fatalf("expected NonCanonical, got %v", err)
	}
	missing := [][]byte{frames[0], frames[1]}
	if _, err := modes.AssembleStream(missing, lim); err == nil {
		t.Fatalf("expected missing rejection")
	} else if werrors.ClassOf(err) != werrors.Malformed {
		t.Fatalf("expected Malformed, got %v", err)
	}
}

func TestActivationAppendNeverDecodes(t *testing.T) {
	lim := limits.Default()
	dtype, err := modes.ParseStreamDtype("bf16")
	if err != nil {
		t.Fatalf("dtype: %v", err)
	}
	head := modes.ActivationStreamHead{
		Stream: modes.ActivationStreamKind, Step: 1, Rank: 0, World: 1,
		ChunkCount: 2, Dtype: dtype, Shape: []uint64{2, 4},
	}
	writer, err := modes.NewActivationStreamWriter(head, lim)
	if err != nil {
		t.Fatalf("writer: %v", err)
	}
	first, err := writer.AppendChunk(make([]byte, 16), lim)
	if err != nil {
		t.Fatalf("append 0: %v", err)
	}
	second, err := writer.AppendChunk(make([]byte, 16), lim)
	if err != nil {
		t.Fatalf("append 1: %v", err)
	}
	if writer.EncodedChunks() != 2 {
		t.Fatalf("expected 2 encoded chunks, saw %d", writer.EncodedChunks())
	}
	if writer.DecodedChunks() != 0 {
		t.Fatalf("append decoded %d chunks", writer.DecodedChunks())
	}
	assembly, err := writer.Assemble([][]byte{first, second}, lim)
	if err != nil {
		t.Fatalf("assemble: %v", err)
	}
	if assembly.ChunksDecoded != 2 || writer.DecodedChunks() != 2 {
		t.Fatalf("assembly did not count both chunks")
	}
}

func TestActivationChunkSetDecodesOneChunk(t *testing.T) {
	lim := limits.Default()
	dtype, err := modes.ParseStreamDtype("bf16")
	if err != nil {
		t.Fatalf("dtype: %v", err)
	}
	head := modes.ActivationStreamHead{
		Stream: modes.ActivationStreamKind, Step: 1, Rank: 0, World: 1,
		ChunkCount: 4, Dtype: dtype, Shape: []uint64{2, 4},
	}
	frames := make([][]byte, 4)
	expected := make([][]byte, 4)
	for index := 0; index < 4; index++ {
		elements := make([]byte, 16)
		for offset := range elements {
			elements[offset] = byte(index*16 + offset)
		}
		expected[index] = elements
		encoded, err := modes.EncodeActivationChunk(head, uint64(index), elements, lim)
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		frames[index] = encoded
	}
	for index := 0; index < 3; index++ {
		frames[index] = frames[index][:4]
	}
	set := modes.NewActivationChunkSet(frames)
	chunk, err := set.DecodeChunk(3, lim)
	if err != nil {
		t.Fatalf("decode chunk 3: %v", err)
	}
	if !bytes.Equal(chunk.Bytes, expected[3]) {
		t.Fatalf("chunk 3 bytes differ")
	}
	if set.DecodedChunks() != 1 {
		t.Fatalf("expected 1 decoded chunk, saw %d", set.DecodedChunks())
	}
	slice, err := set.ReadChunkRange(3, 4, 12, lim)
	if err != nil {
		t.Fatalf("range: %v", err)
	}
	if !bytes.Equal(slice, expected[3][4:12]) {
		t.Fatalf("range bytes differ")
	}
	if set.DecodedChunks() != 2 {
		t.Fatalf("expected 2 decoded chunks, saw %d", set.DecodedChunks())
	}
}
