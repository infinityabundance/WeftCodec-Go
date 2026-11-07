package modes

// Differential test: the word-level optimised kernels must agree with the
// scalar kernels on randomised input. 10,000 Huffman streams and 10,000 BF16
// plane pairs per run, with accept/reject decisions compared, not only
// successful decodes. Go has no SIMD intrinsics; the gain here is word-level
// only (an 8-byte refill and a two-byte pack instead of a 64-bit accumulator).

import (
	"bytes"
	"math/bits"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func kernelXorshift(state *uint64) uint64 {
	x := *state
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	*state = x
	return x * 0x2545F4914F6CDD1D
}

func kernelBelow(state *uint64, bound int) int {
	if bound == 0 {
		return 0
	}
	return int(kernelXorshift(state) % uint64(bound))
}

func kernelPlane(kind, length int, state *uint64) []byte {
	out := make([]byte, length)
	for index := range out {
		switch kind {
		case 0:
			out[index] = byte(kernelXorshift(state))
		case 1:
			r := kernelXorshift(state)
			out[index] = byte(bits.LeadingZeros64(r)) * 7
		case 2:
			out[index] = 0x2a
		default:
			if kernelXorshift(state)&1 == 0 {
				out[index] = 0x01
			} else {
				out[index] = 0xfe
			}
		}
	}
	return out
}

func TestHuffmanOptimisedMatchesScalarOn10000Streams(t *testing.T) {
	state := uint64(0x123456789abcdef1)
	compared := 0
	rejected := 0
	for testCase := 0; testCase < 10000; testCase++ {
		kind := testCase % 4
		length := kernelBelow(&state, 2048)
		source := kernelPlane(kind, length, &state)
		encoded := HuffmanEncode(source)
		if testCase%3 == 0 && len(encoded) > 0 {
			encoded[kernelBelow(&state, len(encoded))] ^= 1 << uint(kernelBelow(&state, 8))
		}
		fastBytes, fastConsumed, fastErr := HuffmanDecodePrefixWordOptimised(encoded, len(source))
		slowBytes, slowConsumed, slowErr := HuffmanDecodePrefixScalar(encoded, len(source))
		if (fastErr == nil) != (slowErr == nil) {
			t.Fatalf("case %d: one kernel accepted: fast=%v slow=%v", testCase, fastErr, slowErr)
		}
		if fastErr != nil {
			if werrors.ClassOf(fastErr) != werrors.ClassOf(slowErr) {
				t.Fatalf("case %d: error class differs: %v vs %v", testCase, fastErr, slowErr)
			}
			rejected++
			continue
		}
		if !bytes.Equal(fastBytes, slowBytes) {
			t.Fatalf("case %d: bytes differ", testCase)
		}
		if fastConsumed != slowConsumed {
			t.Fatalf("case %d: consumed differs: %d vs %d", testCase, fastConsumed, slowConsumed)
		}
		if testCase%3 != 0 && !bytes.Equal(fastBytes, source) {
			t.Fatalf("case %d: round trip differs", testCase)
		}
		compared++
	}
	if compared+rejected != 10000 {
		t.Fatalf("counted %d + %d != 10000", compared, rejected)
	}
	if compared < 5000 {
		t.Fatalf("too few accepted cases: %d", compared)
	}
	if rejected == 0 {
		t.Fatalf("mutations were all harmless")
	}
}

func TestBF16JoinMatchesWordReferenceOn10000PlanePairs(t *testing.T) {
	state := uint64(0x0f1e2d3c4b5a6978)
	dtype, err := ParseFloatDtype("bf16")
	if err != nil {
		t.Fatalf("dtype: %v", err)
	}
	lim := limits.Default()
	for testCase := 0; testCase < 10000; testCase++ {
		elements := kernelBelow(&state, 4096)
		exponent := make([]byte, elements)
		mantissa := make([]byte, elements)
		for index := 0; index < elements; index++ {
			exponent[index] = byte(kernelXorshift(&state))
			mantissa[index] = byte(kernelXorshift(&state))
		}
		joined, err := JoinFloatTensorPayload(exponent, mantissa, dtype, uint64(elements), lim)
		if err != nil {
			t.Fatalf("case %d: %v", testCase, err)
		}
		if len(joined) != elements*2 {
			t.Fatalf("case %d: length %d", testCase, len(joined))
		}
		for index := 0; index < elements; index++ {
			expectedLow := (mantissa[index] & 0x7f) | ((exponent[index] & 1) << 7)
			expectedHigh := ((exponent[index] >> 1) & 0x7f) | (mantissa[index] & 0x80)
			if joined[index*2] != expectedLow || joined[index*2+1] != expectedHigh {
				t.Fatalf("case %d element %d: got %#x %#x", testCase, index, joined[index*2], joined[index*2+1])
			}
		}
	}
}

func kernelBenchPlane(size int) ([]byte, []byte, []byte) {
	state := uint64(0x9e3779b97f4a7c15)
	source := make([]byte, size)
	for index := range source {
		r := kernelXorshift(&state)
		source[index] = byte(bits.LeadingZeros64(r)) * 7
	}
	encoded := HuffmanEncode(source)
	return source, encoded, nil
}

func BenchmarkHuffmanDecodeWordOptimised(b *testing.B) {
	source, encoded, _ := kernelBenchPlane(1 << 20)
	b.SetBytes(int64(len(source)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := HuffmanDecodePrefixWordOptimised(encoded, len(source)); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkHuffmanDecodeScalar(b *testing.B) {
	source, encoded, _ := kernelBenchPlane(1 << 20)
	b.SetBytes(int64(len(source)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := HuffmanDecodePrefixScalar(encoded, len(source)); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBF16JoinOptimised(b *testing.B) {
	dtype, _ := ParseFloatDtype("bf16")
	lim := limits.Default()
	elements := 1 << 19
	exponent := make([]byte, elements)
	mantissa := make([]byte, elements)
	state := uint64(0x243f6a8885a308d3)
	for index := 0; index < elements; index++ {
		exponent[index] = byte(kernelXorshift(&state))
		mantissa[index] = byte(kernelXorshift(&state))
	}
	b.SetBytes(int64(elements * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := JoinFloatTensorPayload(exponent, mantissa, dtype, uint64(elements), lim); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBF16JoinAccumulator(b *testing.B) {
	elements := 1 << 19
	exponent := make([]byte, elements)
	mantissa := make([]byte, elements)
	out := make([]byte, elements*2)
	state := uint64(0x243f6a8885a308d3)
	for index := 0; index < elements; index++ {
		exponent[index] = byte(kernelXorshift(&state))
		mantissa[index] = byte(kernelXorshift(&state))
	}
	b.SetBytes(int64(elements * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// The pre-optimisation accumulator loop for the BF16 width: one
		// 8-bit group per element, read through the 64-bit accumulator.
		var accumulator uint64
		var accumulatedBits uint
		planePos := 0
		for index := 0; index < elements; index++ {
			exponentByte := exponent[index]
			for accumulatedBits <= 56 && planePos < len(mantissa) {
				accumulator |= uint64(mantissa[planePos]) << (56 - accumulatedBits)
				planePos++
				accumulatedBits += 8
			}
			value := accumulator >> 56
			accumulator <<= 8
			accumulatedBits -= 8
			sign := value >> 7
			mant := value & 0x7f
			word := sign<<15 | uint64(exponentByte)<<7 | mant
			out[index*2] = byte(word)
			out[index*2+1] = byte(word >> 8)
		}
	}
}

func BenchmarkBF16JoinScalar(b *testing.B) {
	elements := 1 << 19
	exponent := make([]byte, elements)
	mantissa := make([]byte, elements)
	out := make([]byte, elements*2)
	state := uint64(0x243f6a8885a308d3)
	for index := 0; index < elements; index++ {
		exponent[index] = byte(kernelXorshift(&state))
		mantissa[index] = byte(kernelXorshift(&state))
	}
	b.SetBytes(int64(elements * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for index := 0; index < elements; index++ {
			exponentByte := exponent[index]
			value := uint64(mantissa[index])
			sign := value >> 7
			mant := value & 0x7f
			word := sign<<15 | uint64(exponentByte)<<7 | mant
			out[index*2] = byte(word)
			out[index*2+1] = byte(word >> 8)
		}
	}
}
