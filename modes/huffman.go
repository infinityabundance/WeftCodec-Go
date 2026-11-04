package modes

// huffman-canonical-v1: deterministic order-0 canonical Huffman coder.
//
// Serialized form:
//
//	u8[256] code lengths (0 = symbol absent)
//	|| bitstream, MSB-first, zero-padded to a byte boundary
//
// The uncompressed length is carried by the caller (FLOAT_TENSOR's
// element_count). A zero-symbol table is valid only for an empty plane; a
// one-symbol table uses length 1 and an empty bitstream. Code lengths are
// limited to 24 bits by repeatedly halving (floor, minimum 1) every non-zero
// frequency and rebuilding the tree. Table-shape failures are NonCanonical;
// bitstream and length failures are Corrupt.

import (
	"sort"

	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// HuffmanCoderID is the normative coder identifier in the FLOAT_TENSOR
// descriptor.
const HuffmanCoderID = "huffman-canonical-v1"

const (
	huffmanTableBytes  = 256
	huffmanMaxCodeBits = 24
)

type huffmanNode struct {
	weight    uint64
	minSymbol uint8
	left      int
	right     int
	symbol    int
	active    bool
}

func huffmanNodeLess(nodes []huffmanNode, first, second int) bool {
	a, b := nodes[first], nodes[second]
	if a.weight != b.weight {
		return a.weight < b.weight
	}
	if a.minSymbol != b.minSymbol {
		return a.minSymbol < b.minSymbol
	}
	return first < second
}

func huffmanMinActive(nodes []huffmanNode, exclude int) int {
	best := -1
	for index := range nodes {
		if !nodes[index].active || index == exclude {
			continue
		}
		if best < 0 || huffmanNodeLess(nodes, index, best) {
			best = index
		}
	}
	return best
}

// huffmanLengths builds the deterministic Huffman code lengths for the given
// frequencies. present lists the symbols with non-zero weight in ascending
// order.
func huffmanLengths(weights *[256]uint64, present []int) [256]uint8 {
	nodes := make([]huffmanNode, 0, 2*len(present)-1)
	for _, symbol := range present {
		nodes = append(nodes, huffmanNode{
			weight:    weights[symbol],
			minSymbol: uint8(symbol),
			left:      -1,
			right:     -1,
			symbol:    symbol,
			active:    true,
		})
	}
	active := len(present)
	for active > 1 {
		first := huffmanMinActive(nodes, -1)
		second := huffmanMinActive(nodes, first)
		nodes[first].active = false
		nodes[second].active = false
		minSymbol := nodes[first].minSymbol
		if nodes[second].minSymbol < minSymbol {
			minSymbol = nodes[second].minSymbol
		}
		nodes = append(nodes, huffmanNode{
			weight:    nodes[first].weight + nodes[second].weight,
			minSymbol: minSymbol,
			left:      first,
			right:     second,
			symbol:    -1,
			active:    true,
		})
		active--
	}

	var lengths [256]uint8
	root := -1
	for index := range nodes {
		if nodes[index].active {
			root = index
			break
		}
	}
	type frame struct {
		index int
		depth uint16
	}
	stack := []frame{{index: root}}
	for len(stack) > 0 {
		current := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		node := nodes[current.index]
		if node.symbol >= 0 {
			lengths[node.symbol] = uint8(current.depth)
			continue
		}
		stack = append(stack,
			frame{index: node.left, depth: current.depth + 1},
			frame{index: node.right, depth: current.depth + 1})
	}
	return lengths
}

func buildHuffmanLengths(weights *[256]uint64) [256]uint8 {
	var lengths [256]uint8
	present := make([]int, 0, 256)
	for symbol := 0; symbol < 256; symbol++ {
		if weights[symbol] > 0 {
			present = append(present, symbol)
		}
	}
	if len(present) == 0 {
		return lengths
	}
	if len(present) == 1 {
		lengths[present[0]] = 1
		return lengths
	}
	scaled := *weights
	for {
		candidate := huffmanLengths(&scaled, present)
		maxLen := 0
		for _, length := range candidate {
			if int(length) > maxLen {
				maxLen = int(length)
			}
		}
		if maxLen <= huffmanMaxCodeBits {
			return candidate
		}
		for _, symbol := range present {
			halved := scaled[symbol] >> 1
			if halved == 0 {
				halved = 1
			}
			scaled[symbol] = halved
		}
	}
}

func canonicalHuffmanCodes(lengths *[256]uint8) [256]uint32 {
	type entry struct {
		length uint8
		symbol uint8
	}
	order := make([]entry, 0, 256)
	for symbol := 0; symbol < 256; symbol++ {
		if lengths[symbol] > 0 {
			order = append(order, entry{length: lengths[symbol], symbol: uint8(symbol)})
		}
	}
	sort.Slice(order, func(i, j int) bool {
		if order[i].length != order[j].length {
			return order[i].length < order[j].length
		}
		return order[i].symbol < order[j].symbol
	})
	var codes [256]uint32
	code := uint32(0)
	previous := uint8(0)
	for _, item := range order {
		code <<= uint(item.length - previous)
		codes[item.symbol] = code
		code++
		previous = item.length
	}
	return codes
}

// HuffmanEncode encodes a byte plane with huffman-canonical-v1.
func HuffmanEncode(input []byte) []byte {
	var weights [256]uint64
	for _, symbol := range input {
		weights[symbol]++
	}
	lengths := buildHuffmanLengths(&weights)
	used := 0
	for _, length := range lengths {
		if length > 0 {
			used++
		}
	}
	out := make([]byte, 0, huffmanTableBytes+len(input))
	out = append(out, lengths[:]...)
	if used < 2 {
		return out
	}
	codes := canonicalHuffmanCodes(&lengths)
	var buffer uint64
	var pending uint
	for _, symbol := range input {
		length := uint(lengths[symbol])
		buffer = (buffer << length) | uint64(codes[symbol])
		pending += length
		for pending >= 8 {
			pending -= 8
			out = append(out, byte(buffer>>pending))
		}
		if pending == 0 {
			buffer = 0
		} else {
			buffer &= (uint64(1) << pending) - 1
		}
	}
	if pending > 0 {
		out = append(out, byte(buffer<<(8-pending)))
	}
	return out
}

// HuffmanDecode decodes a huffman-canonical-v1 stream into exactly
// expectedLen bytes. The whole slice is one stream: any trailing byte after
// the declared output is Corrupt.
func HuffmanDecode(encoded []byte, expectedLen int) ([]byte, error) {
	out, consumed, err := HuffmanDecodePrefix(encoded, expectedLen)
	if err != nil {
		return nil, err
	}
	if consumed != len(encoded) {
		return nil, werrors.NewCorrupt(
			"huffman-canonical-v1 bitstream has trailing bytes after the declared output")
	}
	return out, nil
}

// HuffmanDecodePrefix decodes one huffman-canonical-v1 stream from the start
// of encoded and returns the decoded bytes plus the number of bytes consumed
// (the 256-byte table plus its own padded bitstream). Used where several
// independently coded planes are concatenated (mode 11).
func HuffmanDecodePrefix(encoded []byte, expectedLen int) ([]byte, int, error) {
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

	type entry struct {
		length uint8
		symbol uint8
	}
	order := make([]entry, 0, used)
	for symbol := 0; symbol < 256; symbol++ {
		if lengths[symbol] > 0 {
			order = append(order, entry{length: lengths[symbol], symbol: uint8(symbol)})
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

	// Padding is checked only inside the bytes this stream consumed; any
	// further bytes belong to the caller (the next plane).
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
