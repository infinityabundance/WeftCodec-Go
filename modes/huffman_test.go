package modes_test

import (
	"bytes"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func huffmanPattern(length int, state uint64) []byte {
	out := make([]byte, 0, length)
	for i := 0; i < length; i++ {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		out = append(out, byte(state&0xff))
	}
	return out
}

func huffmanRoundTrip(t *testing.T, input []byte) {
	t.Helper()
	encoded := modes.HuffmanEncode(input)
	decoded, err := modes.HuffmanDecode(encoded, len(input))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !bytes.Equal(decoded, input) {
		t.Fatalf("round-trip mismatch")
	}
	if again := modes.HuffmanEncode(input); !bytes.Equal(again, encoded) {
		t.Fatalf("encoder is not deterministic")
	}
}

func TestHuffmanEmptyPlane(t *testing.T) {
	encoded := modes.HuffmanEncode(nil)
	if len(encoded) != 256 {
		t.Fatalf("empty encoding is %d bytes, want 256", len(encoded))
	}
	for _, value := range encoded {
		if value != 0 {
			t.Fatalf("empty table has a non-zero length")
		}
	}
	decoded, err := modes.HuffmanDecode(encoded, 0)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(decoded) != 0 {
		t.Fatalf("empty decode returned %d bytes", len(decoded))
	}
}

func TestHuffmanSingleDistinctByte(t *testing.T) {
	input := bytes.Repeat([]byte{0x5a}, 4096)
	encoded := modes.HuffmanEncode(input)
	if len(encoded) != 256 {
		t.Fatalf("single-symbol encoding is %d bytes, want 256", len(encoded))
	}
	if encoded[0x5a] != 1 {
		t.Fatalf("single-symbol length = %d, want 1", encoded[0x5a])
	}
	used := 0
	for _, value := range encoded {
		if value > 0 {
			used++
		}
	}
	if used != 1 {
		t.Fatalf("used symbols = %d, want 1", used)
	}
	huffmanRoundTrip(t, input)
}

func TestHuffmanAll256Distinct(t *testing.T) {
	input := make([]byte, 256)
	for index := range input {
		input[index] = byte(index)
	}
	huffmanRoundTrip(t, input)
	encoded := modes.HuffmanEncode(input)
	for symbol := 0; symbol < 256; symbol++ {
		if encoded[symbol] == 0 {
			t.Fatalf("symbol %d has no code length", symbol)
		}
	}
}

func TestHuffmanHighlySkewed(t *testing.T) {
	input := make([]byte, 1_000_000)
	for index := 990_000; index < len(input); index++ {
		input[index] = byte(index%255 + 1)
	}
	huffmanRoundTrip(t, input)
}

func TestHuffmanRandomUniform(t *testing.T) {
	huffmanRoundTrip(t, huffmanPattern(250_000, 0x123456789abcdef0))
}

func TestHuffmanTwoMegabytePlane(t *testing.T) {
	input := huffmanPattern(2_000_000, 0xdeadbeefcafebabe)
	for index := 0; index < 1_500_000; index++ {
		input[index] &= 0x0f
	}
	huffmanRoundTrip(t, input)
}

func TestHuffmanLengthLimiting(t *testing.T) {
	fib := make([]uint64, 32)
	fib[0], fib[1] = 1, 1
	total := uint64(0)
	for index := 2; index < len(fib); index++ {
		fib[index] = fib[index-1] + fib[index-2]
	}
	for _, count := range fib {
		total += count
	}
	input := make([]byte, 0, total)
	for symbol, count := range fib {
		for index := uint64(0); index < count; index++ {
			input = append(input, byte(symbol))
		}
	}
	encoded := modes.HuffmanEncode(input)
	for symbol := 0; symbol < 256; symbol++ {
		if encoded[symbol] > 24 {
			t.Fatalf("symbol %d code length %d exceeds 24", symbol, encoded[symbol])
		}
	}
	huffmanRoundTrip(t, input)
}

func huffmanTable(entries map[int]byte) []byte {
	table := make([]byte, 256)
	for symbol, length := range entries {
		table[symbol] = length
	}
	return table
}

func TestHuffmanRejectsCodeLengthTooLong(t *testing.T) {
	encoded := huffmanTable(map[int]byte{0: 25})
	if _, err := modes.HuffmanDecode(encoded, 1); werrors.ClassOf(err) != werrors.NonCanonical {
		t.Fatalf("class = %v", werrors.ClassOf(err))
	}
}

func TestHuffmanRejectsOverSubscribedTable(t *testing.T) {
	encoded := bytes.Repeat([]byte{1}, 256)
	encoded = append(encoded, 0, 0)
	if _, err := modes.HuffmanDecode(encoded, 2); werrors.ClassOf(err) != werrors.NonCanonical {
		t.Fatalf("class = %v", werrors.ClassOf(err))
	}
}

func TestHuffmanRejectsIncompleteTable(t *testing.T) {
	encoded := huffmanTable(map[int]byte{0: 2, 1: 2})
	if _, err := modes.HuffmanDecode(encoded, 2); werrors.ClassOf(err) != werrors.NonCanonical {
		t.Fatalf("class = %v", werrors.ClassOf(err))
	}
}

func TestHuffmanRejectsSingleSymbolWithExtraBits(t *testing.T) {
	encoded := huffmanTable(map[int]byte{7: 1})
	encoded = append(encoded, 0)
	if _, err := modes.HuffmanDecode(encoded, 4); werrors.ClassOf(err) != werrors.Corrupt {
		t.Fatalf("class = %v", werrors.ClassOf(err))
	}
}

func TestHuffmanRejectsTruncation(t *testing.T) {
	encoded := huffmanTable(map[int]byte{0: 1, 1: 1})
	if _, err := modes.HuffmanDecode(encoded, 4); werrors.ClassOf(err) != werrors.Corrupt {
		t.Fatalf("class = %v", werrors.ClassOf(err))
	}
}

func TestHuffmanRejectsNonZeroPadding(t *testing.T) {
	encoded := huffmanTable(map[int]byte{0: 1, 1: 1})
	encoded = append(encoded, 0b0100_0000)
	if _, err := modes.HuffmanDecode(encoded, 1); werrors.ClassOf(err) != werrors.Corrupt {
		t.Fatalf("class = %v", werrors.ClassOf(err))
	}
}

func TestHuffmanRejectsTrailingBytes(t *testing.T) {
	encoded := huffmanTable(map[int]byte{0: 1, 1: 1})
	encoded = append(encoded, 0x00, 0x00)
	if _, err := modes.HuffmanDecode(encoded, 1); werrors.ClassOf(err) != werrors.Corrupt {
		t.Fatalf("class = %v", werrors.ClassOf(err))
	}
}

func TestHuffmanRejectsShortTable(t *testing.T) {
	if _, err := modes.HuffmanDecode([]byte{0, 0, 0, 0}, 1); werrors.ClassOf(err) != werrors.Corrupt {
		t.Fatalf("class = %v", werrors.ClassOf(err))
	}
}
