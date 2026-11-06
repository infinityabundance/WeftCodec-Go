package modes

// Benchmarks for huffman-canonical-v1 decoding: the shipped table-driven
// decoder against the frozen bit-at-a-time reference decoder.
//
// Run with:
//
//	go test ./modes -run '^$' -bench 'BenchmarkHuffmanDecode' -benchtime=2s -count=3
//
// b.SetBytes is the decoded plane size, so Go reports throughput as MB/s where
// MB is 10^6 bytes; divide by 1.048576 for MiB/s.

import (
	"fmt"
	"testing"
)

var huffmanBenchSink []byte

var huffmanBenchDistributions = []string{
	"bf16-exponent-16",
	"uniform-256",
	"single-symbol",
}

var huffmanBenchSizes = []int{1 << 20, 16 << 20}

// huffmanBenchInput builds the same deterministic plane for a given
// distribution and size in both benchmark functions.
func huffmanBenchInput(distribution string, size int) []byte {
	state := uint64(0x243f6a8885a308d3)
	switch distribution {
	case "bf16-exponent-16":
		return huffmanSkewedPlane(&state, 16, size)
	case "uniform-256":
		return huffmanUniformPlane(&state, size)
	case "single-symbol":
		out := make([]byte, size)
		for index := range out {
			out[index] = 0x3f
		}
		return out
	default:
		panic("unknown distribution " + distribution)
	}
}

func benchmarkHuffmanDecode(b *testing.B, reference bool) {
	for _, size := range huffmanBenchSizes {
		for _, distribution := range huffmanBenchDistributions {
			name := fmt.Sprintf("%s/%dMiB", distribution, size>>20)
			b.Run(name, func(b *testing.B) {
				input := huffmanBenchInput(distribution, size)
				encoded := HuffmanEncode(input)
				decode := HuffmanDecode
				if reference {
					decode = huffmanDecodeReference
				}
				// One out-of-band correctness check, then time decode only.
				first, err := decode(encoded, len(input))
				if err != nil {
					b.Fatalf("warm-up decode: %v", err)
				}
				if len(first) != len(input) {
					b.Fatalf("warm-up length %d, want %d", len(first), len(input))
				}
				b.SetBytes(int64(len(input)))
				b.ReportAllocs()
				b.ResetTimer()
				for index := 0; index < b.N; index++ {
					out, err := decode(encoded, len(input))
					if err != nil {
						b.Fatalf("decode: %v", err)
					}
					huffmanBenchSink = out
				}
			})
		}
	}
}

func BenchmarkHuffmanDecodeFast(b *testing.B) {
	benchmarkHuffmanDecode(b, false)
}

func BenchmarkHuffmanDecodeReference(b *testing.B) {
	benchmarkHuffmanDecode(b, true)
}
