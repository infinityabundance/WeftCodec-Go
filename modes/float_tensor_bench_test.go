package modes

// Benchmarks for the FLOAT_TENSOR plane join: the shipped word-at-a-time join
// against the frozen per-bit reference join, plus frame encode and full decode.
//
// Run with:
//
//	go test ./modes -run '^$' -bench 'BenchmarkJoinFloatTensorPayload' -benchtime=2s -count=3
//
// b.SetBytes is the decoded logical size, so Go reports throughput as MB/s
// where MB is 10^6 bytes; divide by 1.048576 for MiB/s.

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/limits"
)

var floatTensorBenchSink []byte

var floatTensorBenchDtypes = []string{"bf16", "fp16", "fp32"}

var floatTensorBenchSizes = []int{1 << 20, 16 << 20}

// floatTensorBenchPayload builds a weight-like payload: uniformly random
// mantissa and sign bits, exponent bytes drawn from a skewed halving
// distribution over the low half of the exponent range.
func floatTensorBenchPayload(dtype FloatDtype, size int) []byte {
	elementBytes := int(dtype.ElementSize())
	elementCount := size / elementBytes
	exponentSymbols := 1 << uint(dtype.ExponentBits)
	subset := exponentSymbols / 2
	if subset == 0 {
		subset = 1
	}
	weights := make([]uint64, subset)
	remaining := uint64(1) << 20
	for index := range weights {
		take := remaining / 2
		if take == 0 {
			take = 1
		}
		weights[index] = take
		if remaining > take {
			remaining -= take
		} else {
			remaining = 1
		}
	}
	total := uint64(0)
	for _, weight := range weights {
		total += weight
	}
	state := uint64(0x243f6a8885a308d3)
	out := make([]byte, elementCount*elementBytes)
	mantissaMask := (uint64(1) << uint(dtype.MantissaBits)) - 1
	for index := 0; index < elementCount; index++ {
		pick := joinXorshift64Star(&state) % total
		exponent := 0
		for candidate, weight := range weights {
			if pick < weight {
				exponent = candidate
				break
			}
			pick -= weight
		}
		sign := joinXorshift64Star(&state) >> 63
		mantissa := joinXorshift64Star(&state) & mantissaMask
		word := (sign << uint(dtype.ExponentBits+dtype.MantissaBits)) |
			(uint64(exponent) << uint(dtype.MantissaBits)) |
			mantissa
		for b := 0; b < elementBytes; b++ {
			out[index*elementBytes+b] = byte(word >> uint(8*b))
		}
	}
	return out
}

func benchmarkJoinFloatTensorPayload(b *testing.B, reference bool) {
	lim := limits.Default()
	join := JoinFloatTensorPayload
	if reference {
		join = joinFloatTensorPayloadReference
	}
	for _, name := range floatTensorBenchDtypes {
		dtype, err := ParseFloatDtype(name)
		if err != nil {
			b.Fatalf("dtype %s: %v", name, err)
		}
		for _, size := range floatTensorBenchSizes {
			b.Run(fmt.Sprintf("%s/%dMiB", name, size>>20), func(b *testing.B) {
				payload := floatTensorBenchPayload(dtype, size)
				elementCount := uint64(len(payload)) / dtype.ElementSize()
				compressed, mantissa, err := SplitFloatTensorPayload(payload, dtype, elementCount)
				if err != nil {
					b.Fatalf("split: %v", err)
				}
				exponent, err := HuffmanDecode(compressed, int(elementCount))
				if err != nil {
					b.Fatalf("huffman decode: %v", err)
				}
				// One out-of-band correctness check, then time the join only.
				warm, err := join(exponent, mantissa, dtype, elementCount, lim)
				if err != nil {
					b.Fatalf("warm-up join: %v", err)
				}
				if !bytes.Equal(warm, payload) {
					b.Fatalf("warm-up join output mismatch")
				}
				b.SetBytes(int64(len(payload)))
				b.ReportAllocs()
				b.ResetTimer()
				for index := 0; index < b.N; index++ {
					out, err := join(exponent, mantissa, dtype, elementCount, lim)
					if err != nil {
						b.Fatalf("join: %v", err)
					}
					floatTensorBenchSink = out
				}
			})
		}
	}
}

func BenchmarkJoinFloatTensorPayloadFast(b *testing.B) {
	benchmarkJoinFloatTensorPayload(b, false)
}

func BenchmarkJoinFloatTensorPayloadReference(b *testing.B) {
	benchmarkJoinFloatTensorPayload(b, true)
}

func benchmarkEncodeFloatTensor(b *testing.B) {
	lim := limits.Default()
	for _, name := range floatTensorBenchDtypes {
		dtype, err := ParseFloatDtype(name)
		if err != nil {
			b.Fatalf("dtype %s: %v", name, err)
		}
		for _, size := range floatTensorBenchSizes {
			b.Run(fmt.Sprintf("%s/%dMiB", name, size>>20), func(b *testing.B) {
				payload := floatTensorBenchPayload(dtype, size)
				shape := []uint64{uint64(len(payload)) / dtype.ElementSize()}
				if _, err := EncodeFloatTensor(payload, dtype, shape, lim); err != nil {
					b.Fatalf("warm-up encode: %v", err)
				}
				b.SetBytes(int64(len(payload)))
				b.ReportAllocs()
				b.ResetTimer()
				for index := 0; index < b.N; index++ {
					out, err := EncodeFloatTensor(payload, dtype, shape, lim)
					if err != nil {
						b.Fatalf("encode: %v", err)
					}
					floatTensorBenchSink = out
				}
			})
		}
	}
}

func BenchmarkEncodeFloatTensor(b *testing.B) {
	benchmarkEncodeFloatTensor(b)
}
