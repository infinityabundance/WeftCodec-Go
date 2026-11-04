// Package varint implements minimal unsigned LEB128 varints
// (WEFTCODEC-FORMAT §1).
//
// Rules: at most 10 bytes; a multi-byte encoding must not end in 0x00; the
// tenth byte must be 0x00 or 0x01. Non-minimal encodings are NonCanonical;
// overlong or truncated encodings are Malformed.
package varint

import "github.com/infinityabundance/WeftCodec-Go/werrors"

// MaxLen is the maximum encoded length of a u64 varint.
const MaxLen = 10

// EncodedLen reports the number of bytes Encode writes.
func EncodedLen(v uint64) int {
	n := 1
	for v >= 0x80 {
		v >>= 7
		n++
	}
	return n
}

// Append appends the minimal LEB128 encoding of v to dst.
func Append(dst []byte, v uint64) []byte {
	for {
		b := byte(v & 0x7f)
		v >>= 7
		if v == 0 {
			return append(dst, b)
		}
		dst = append(dst, b|0x80)
	}
}

// Encode returns the minimal LEB128 encoding of v.
func Encode(v uint64) []byte { return Append(nil, v) }

// EncodeTo writes v into buf and returns the number of bytes written.
func EncodeTo(buf []byte, v uint64) int {
	n := 0
	for {
		b := byte(v & 0x7f)
		v >>= 7
		if v == 0 {
			buf[n] = b
			return n + 1
		}
		buf[n] = b | 0x80
		n++
	}
}

// DecodeAt decodes one varint at pos, returning the value and next position.
func DecodeAt(input []byte, pos int) (uint64, int, error) {
	var value uint64
	shift := uint(0)
	offset := pos
	count := 0
	for {
		if offset >= len(input) || offset < 0 {
			return 0, 0, werrors.NewMalformed("truncated varint")
		}
		b := input[offset]
		offset++
		count++
		if count > MaxLen {
			return 0, 0, werrors.NewMalformed("varint longer than 10 bytes")
		}
		if count == MaxLen && b > 0x01 {
			return 0, 0, werrors.NewMalformed("varint overflows u64")
		}
		low := uint64(b & 0x7f)
		if shift < 64 {
			value |= low << shift
		} else if low != 0 {
			return 0, 0, werrors.NewMalformed("varint overflows u64")
		}
		if b&0x80 == 0 {
			if count > 1 && b == 0x00 {
				return 0, 0, werrors.NewNonCanonical("non-minimal varint")
			}
			return value, offset, nil
		}
		shift += 7
		if shift > 63 {
			return 0, 0, werrors.NewMalformed("varint exceeds 10 bytes")
		}
	}
}

// Decode decodes one varint at *pos, advancing the cursor on success.
func Decode(input []byte, pos *int) (uint64, error) {
	value, next, err := DecodeAt(input, *pos)
	if err != nil {
		return 0, err
	}
	*pos = next
	return value, nil
}

// ZigzagEncode maps a signed integer to an unsigned one.
func ZigzagEncode(v int64) uint64 {
	return uint64(uint64(v<<1) ^ uint64(v>>63))
}

// ZigzagDecode is the inverse of ZigzagEncode.
func ZigzagDecode(v uint64) int64 {
	return int64(v>>1) ^ -int64(v&1)
}
