package modes

// Test-only reference decoder for huffman-canonical-v1.
//
// huffmanDecodePrefixReference is a verbatim copy of the pre-table
// bit-at-a-time decoder. It is never compiled into the library and exists only
// so the table-driven path can be proved equivalent on every input shape, as
// well as to give the benchmark a same-computation baseline.

import (
	"bytes"
	"sort"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func huffmanDecodePrefixReference(encoded []byte, expectedLen int) ([]byte, int, error) {
	if len(encoded) < huffmanTableBytes {
		return nil, 0, werrors.Corruptf(
			"huffman-canonical-v1 stream is %d bytes, shorter than its %d-byte code-length table",
			len(encoded), huffmanTableBytes)
	}
	lengths := encoded[:huffmanTableBytes]
	stream := encoded[huffmanTableBytes:]

	var counts [huffmanMaxCodeBits + 1]uint32
	used := 0
	maxLen := 0
	for symbol, lengthByte := range lengths {
		length := int(lengthByte)
		if length > huffmanMaxCodeBits {
			return nil, 0, werrors.NonCanonicalf(
				"huffman-canonical-v1 symbol %d has code length %d > %d",
				symbol, length, huffmanMaxCodeBits)
		}
		if length > 0 {
			used++
			counts[length]++
			if length > maxLen {
				maxLen = length
			}
		}
	}

	if used == 0 {
		if expectedLen != 0 {
			return nil, 0, werrors.NewCorrupt(
				"huffman-canonical-v1 table has no symbols but the descriptor declares a non-empty plane")
		}
		return []byte{}, huffmanTableBytes, nil
	}

	if used == 1 {
		if maxLen != 1 {
			return nil, 0, werrors.NewNonCanonical(
				"huffman-canonical-v1 single-symbol table must use code length 1")
		}
		if expectedLen == 0 {
			return nil, 0, werrors.NewNonCanonical(
				"huffman-canonical-v1 non-empty table for an empty plane")
		}
		symbol := byte(0)
		for index, length := range lengths {
			if length > 0 {
				symbol = byte(index)
				break
			}
		}
		out := make([]byte, expectedLen)
		for index := range out {
			out[index] = symbol
		}
		return out, huffmanTableBytes, nil
	}

	if expectedLen == 0 {
		return nil, 0, werrors.NewNonCanonical(
			"huffman-canonical-v1 non-empty table for an empty plane")
	}

	space := uint64(1) << huffmanMaxCodeBits
	usedSpace := uint64(0)
	for length := 1; length <= huffmanMaxCodeBits; length++ {
		usedSpace += uint64(counts[length]) << uint(huffmanMaxCodeBits-length)
		if usedSpace > space {
			return nil, 0, werrors.NewNonCanonical(
				"huffman-canonical-v1 code lengths are over-subscribed")
		}
	}
	if usedSpace != space {
		return nil, 0, werrors.NewNonCanonical(
			"huffman-canonical-v1 code lengths are incomplete")
	}

	order := make([]huffmanOrderEntry, 0, used)
	for symbol := 0; symbol < 256; symbol++ {
		if lengths[symbol] > 0 {
			order = append(order, huffmanOrderEntry{length: lengths[symbol], symbol: uint8(symbol)})
		}
	}
	sort.Slice(order, func(i, j int) bool {
		if order[i].length != order[j].length {
			return order[i].length < order[j].length
		}
		return order[i].symbol < order[j].symbol
	})
	var firstCode [huffmanMaxCodeBits + 1]uint32
	var firstIndex [huffmanMaxCodeBits + 1]uint32
	var assigned [huffmanMaxCodeBits + 1]uint32
	symbols := make([]byte, 0, used)
	code := uint32(0)
	previous := uint8(0)
	for _, item := range order {
		code <<= uint(item.length - previous)
		slot := int(item.length)
		if assigned[slot] == 0 {
			firstCode[slot] = code
			firstIndex[slot] = uint32(len(symbols))
		}
		assigned[slot]++
		symbols = append(symbols, item.symbol)
		code++
		previous = item.length
	}

	capacity := expectedLen
	if capacity > 1<<20 {
		capacity = 1 << 20
	}
	out := make([]byte, 0, capacity)
	bitPos := 0
	currentCode := uint32(0)
	currentLen := 0
	for len(out) < expectedLen {
		if currentLen > 0 {
			slot := currentLen
			if currentCode >= firstCode[slot] {
				offset := currentCode - firstCode[slot]
				if offset < counts[slot] {
					out = append(out, symbols[int(firstIndex[slot]+offset)])
					currentCode = 0
					currentLen = 0
					continue
				}
			}
			if currentLen >= maxLen {
				return nil, 0, werrors.NewCorrupt(
					"huffman-canonical-v1 bitstream contains an invalid code")
			}
		}
		if bitPos >= len(stream)*8 {
			return nil, 0, werrors.Corruptf(
				"huffman-canonical-v1 bitstream ended after %d of %d bytes",
				len(out), expectedLen)
		}
		bit := (stream[bitPos/8] >> uint(7-bitPos%8)) & 1
		currentCode = (currentCode << 1) | uint32(bit)
		currentLen++
		bitPos++
	}

	consumedStream := (bitPos + 7) / 8
	usedBitsInLast := bitPos % 8
	if usedBitsInLast != 0 {
		last := stream[consumedStream-1]
		mask := byte(1<<uint(8-usedBitsInLast)) - 1
		if last&mask != 0 {
			return nil, 0, werrors.NewCorrupt(
				"huffman-canonical-v1 padding bits must be zero")
		}
	}
	return out, huffmanTableBytes + consumedStream, nil
}

func huffmanDecodeReference(encoded []byte, expectedLen int) ([]byte, error) {
	out, consumed, err := huffmanDecodePrefixReference(encoded, expectedLen)
	if err != nil {
		return nil, err
	}
	if consumed != len(encoded) {
		return nil, werrors.NewCorrupt(
			"huffman-canonical-v1 bitstream has trailing bytes after the declared output")
	}
	return out, nil
}

// huffmanOrderEntry is the reference decoder's (length, symbol) sort tuple.
type huffmanOrderEntry struct {
	length uint8
	symbol uint8
}

// huffmanXorshift64Star is the PRNG family used by the other WeftFS
// differential courts.
func huffmanXorshift64Star(state *uint64) uint64 {
	x := *state
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	*state = x
	return x * 0x2545F4914F6CDD1D
}

func huffmanRandomBelow(state *uint64, bound int) int {
	return int(huffmanXorshift64Star(state) % uint64(bound))
}

func huffmanSkewedPlane(state *uint64, symbols, length int) []byte {
	weights := make([]uint64, symbols)
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
	out := make([]byte, length)
	for index := range out {
		pick := huffmanXorshift64Star(state) % total
		symbol := 0
		for candidate, weight := range weights {
			if pick < weight {
				symbol = candidate
				break
			}
			pick -= weight
		}
		out[index] = byte(symbol)
	}
	return out
}

func huffmanUniformPlane(state *uint64, length int) []byte {
	out := make([]byte, length)
	for index := range out {
		out[index] = byte(huffmanXorshift64Star(state) & 0xff)
	}
	return out
}

func huffmanFibonacciPlane(symbols int) []byte {
	fib := make([]uint64, 32)
	fib[0], fib[1] = 1, 1
	for index := 2; index < symbols; index++ {
		fib[index] = fib[index-1] + fib[index-2]
	}
	out := make([]byte, 0, 1<<16)
	for symbol := 0; symbol < symbols; symbol++ {
		for count := uint64(0); count < fib[symbol]; count++ {
			out = append(out, byte(symbol))
		}
	}
	return out
}

type huffmanOutcome struct {
	bytes    []byte
	consumed int
	errClass werrors.Class
	ok       bool
}

func huffmanClassOf(err error) werrors.Class {
	if err == nil {
		return ""
	}
	return werrors.ClassOf(err)
}

func huffmanPrefixOutcome(decode func([]byte, int) ([]byte, int, error), encoded []byte, expectedLen int) huffmanOutcome {
	out, consumed, err := decode(encoded, expectedLen)
	if err != nil {
		return huffmanOutcome{errClass: huffmanClassOf(err)}
	}
	return huffmanOutcome{bytes: out, consumed: consumed, ok: true}
}

func huffmanWholeOutcome(decode func([]byte, int) ([]byte, error), encoded []byte, expectedLen int) huffmanOutcome {
	out, err := decode(encoded, expectedLen)
	if err != nil {
		return huffmanOutcome{errClass: huffmanClassOf(err)}
	}
	return huffmanOutcome{bytes: out, ok: true}
}

func huffmanAssertSame(t *testing.T, encoded []byte, expectedLen int, label string) {
	t.Helper()
	fast := huffmanPrefixOutcome(HuffmanDecodePrefix, encoded, expectedLen)
	reference := huffmanPrefixOutcome(huffmanDecodePrefixReference, encoded, expectedLen)
	if fast.ok != reference.ok || fast.errClass != reference.errClass ||
		!bytes.Equal(fast.bytes, reference.bytes) || fast.consumed != reference.consumed {
		t.Fatalf("prefix decoder differs from reference for %s (expected_len=%d encoded_len=%d): fast=%+v reference=%+v",
			label, expectedLen, len(encoded), fast, reference)
	}
	fastWhole := huffmanWholeOutcome(HuffmanDecode, encoded, expectedLen)
	referenceWhole := huffmanWholeOutcome(huffmanDecodeReference, encoded, expectedLen)
	if fastWhole.ok != referenceWhole.ok || fastWhole.errClass != referenceWhole.errClass ||
		!bytes.Equal(fastWhole.bytes, referenceWhole.bytes) {
		t.Fatalf("whole decoder differs from reference for %s (expected_len=%d encoded_len=%d): fast=%+v reference=%+v",
			label, expectedLen, len(encoded), fastWhole, referenceWhole)
	}
}

func TestHuffmanReferenceDifferential(t *testing.T) {
	const cases = 10000
	const kinds = 12
	state := uint64(0x9e3779b97f4a7c15)
	for run := 0; run < cases; run++ {
		switch run % kinds {
		case 0:
			huffmanAssertSame(t, HuffmanEncode(nil), 0, "empty plane")
		case 1:
			symbol := byte(huffmanXorshift64Star(&state) & 0xff)
			length := huffmanRandomBelow(&state, 4096) + 1
			input := bytes.Repeat([]byte{symbol}, length)
			huffmanAssertSame(t, HuffmanEncode(input), len(input), "single symbol")
		case 2:
			length := huffmanRandomBelow(&state, 4096) + 1
			input := huffmanUniformPlane(&state, length)
			huffmanAssertSame(t, HuffmanEncode(input), len(input), "uniform 256")
		case 3:
			length := huffmanRandomBelow(&state, 4096) + 1
			input := huffmanSkewedPlane(&state, 8, length)
			huffmanAssertSame(t, HuffmanEncode(input), len(input), "skewed 8")
		case 4:
			length := huffmanRandomBelow(&state, 4096) + 1
			input := huffmanSkewedPlane(&state, 16, length)
			huffmanAssertSame(t, HuffmanEncode(input), len(input), "skewed 16")
		case 5:
			symbols := huffmanRandomBelow(&state, 24) + 2
			input := huffmanFibonacciPlane(symbols)
			huffmanAssertSame(t, HuffmanEncode(input), len(input), "fibonacci")
		case 6:
			length := huffmanRandomBelow(&state, 1024) + 1
			input := huffmanSkewedPlane(&state, 12, length)
			encoded := HuffmanEncode(input)
			cut := huffmanRandomBelow(&state, len(encoded)-huffmanTableBytes+1)
			truncated := encoded[:huffmanTableBytes+cut]
			huffmanAssertSame(t, truncated, len(input), "truncated")
		case 7:
			length := huffmanRandomBelow(&state, 1024) + 1
			input := huffmanUniformPlane(&state, length)
			encoded := HuffmanEncode(input)
			extra := huffmanRandomBelow(&state, 4) + 1
			for index := 0; index < extra; index++ {
				encoded = append(encoded, byte(huffmanXorshift64Star(&state)&0xff))
			}
			huffmanAssertSame(t, encoded, len(input), "trailing bytes")
		case 8:
			length := huffmanRandomBelow(&state, 1024) + 1
			input := huffmanSkewedPlane(&state, 6, length)
			encoded := HuffmanEncode(input)
			if len(encoded) > huffmanTableBytes {
				encoded[len(encoded)-1] |= 1
			}
			huffmanAssertSame(t, encoded, len(input), "non-zero padding")
		case 9:
			encoded := make([]byte, huffmanTableBytes)
			entries := huffmanRandomBelow(&state, 256) + 1
			for index := 0; index < entries; index++ {
				symbol := huffmanRandomBelow(&state, 256)
				encoded[symbol] = byte(huffmanXorshift64Star(&state) % 25)
			}
			streamLen := huffmanRandomBelow(&state, 64)
			for index := 0; index < streamLen; index++ {
				encoded = append(encoded, byte(huffmanXorshift64Star(&state)&0xff))
			}
			expectedLen := huffmanRandomBelow(&state, 128)
			huffmanAssertSame(t, encoded, expectedLen, "arbitrary table")
		case 10:
			encoded := make([]byte, huffmanTableBytes)
			symbol := huffmanRandomBelow(&state, 256)
			encoded[symbol] = byte(huffmanXorshift64Star(&state)%231 + 25)
			expectedLen := huffmanRandomBelow(&state, 32)
			huffmanAssertSame(t, encoded, expectedLen, "over-long length")
		default:
			encoded := make([]byte, huffmanTableBytes)
			entries := huffmanRandomBelow(&state, 8) + 2
			for index := 0; index < entries; index++ {
				symbol := huffmanRandomBelow(&state, 256)
				encoded[symbol] = byte(huffmanXorshift64Star(&state)%4 + 1)
			}
			streamLen := huffmanRandomBelow(&state, 16)
			for index := 0; index < streamLen; index++ {
				encoded = append(encoded, byte(huffmanXorshift64Star(&state)&0xff))
			}
			expectedLen := huffmanRandomBelow(&state, 64)
			huffmanAssertSame(t, encoded, expectedLen, "kraft-mismatch table")
		}
	}
}

func TestHuffmanReferencePrefixConsumption(t *testing.T) {
	// Concatenated planes (modes 11-15) depend on the exact consumed byte
	// count, so check it directly over a mixed corpus.
	state := uint64(0xa5a55a5a01234567)
	var packed []byte
	var lengths []int
	for index := 0; index < 64; index++ {
		length := huffmanRandomBelow(&state, 2048) + 1
		input := huffmanSkewedPlane(&state, 10, length)
		packed = append(packed, HuffmanEncode(input)...)
		lengths = append(lengths, length)
	}
	offset := 0
	for _, expectedLen := range lengths {
		fast, fastConsumed, err := HuffmanDecodePrefix(packed[offset:], expectedLen)
		if err != nil {
			t.Fatalf("fast decode: %v", err)
		}
		reference, referenceConsumed, err := huffmanDecodePrefixReference(packed[offset:], expectedLen)
		if err != nil {
			t.Fatalf("reference decode: %v", err)
		}
		if !bytes.Equal(fast, reference) {
			t.Fatalf("output mismatch at offset %d", offset)
		}
		if fastConsumed != referenceConsumed {
			t.Fatalf("consumed mismatch at offset %d: %d vs %d", offset, fastConsumed, referenceConsumed)
		}
		offset += fastConsumed
	}
	if offset != len(packed) {
		t.Fatalf("consumed %d of %d packed bytes", offset, len(packed))
	}
}
