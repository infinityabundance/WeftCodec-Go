package modes

// Differential and reject tests for the FLOAT_TENSOR (mode 9) plane join.
//
// JoinFloatTensorPayload reads the sign+mantissa plane with a 64-bit MSB
// accumulator, one whole element at a time. joinFloatTensorPayloadReference
// below is a verbatim copy of the previous per-bit takeBits loop. It is
// test-only and exists to prove the two implementations agree byte-for-byte,
// including their error classes on malformed planes.
//
// Coverage: all five dtypes, including the non-byte-aligned sign+mantissa
// widths (fp16 11 bits, fp8-e4m3 4 bits, fp8-e5m2 3 bits), element counts from
// 0 upward including counts that leave padding bits, and randomized element
// words (NaN payloads, denormals, signed zeros, and infinities are ordinary bit
// patterns here).

import (
	"bytes"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

var joinAllDtypes = []string{"bf16", "fp16", "fp32", "fp8-e4m3", "fp8-e5m2"}

func joinDtype(t *testing.T, name string) FloatDtype {
	t.Helper()
	dtype, err := ParseFloatDtype(name)
	if err != nil {
		t.Fatalf("dtype %s: %v", name, err)
	}
	return dtype
}

func joinXorshift64Star(state *uint64) uint64 {
	x := *state
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	*state = x
	return x * 0x2545F4914F6CDD1D
}

// ---------------------------------------------------------------------------
// Test-only reference: a verbatim copy of the previous per-bit join.
// ---------------------------------------------------------------------------
func takeBitsReference(plane []byte, pos *uint64, count uint64) uint64 {
	var value uint64
	for i := uint64(0); i < count; i++ {
		bit := (plane[*pos/8] >> uint(7-(*pos%8))) & 1
		value = (value << 1) | uint64(bit)
		*pos = *pos + 1
	}
	return value
}

func joinFloatTensorPayloadReference(
	exponentPlane, mantissaPlane []byte,
	dtype FloatDtype,
	elementCount uint64,
	lim limits.Limits,
) ([]byte, error) {
	if uint64(len(exponentPlane)) != elementCount {
		return nil, werrors.Corruptf(
			"float tensor exponent plane has %d bytes but element_count is %d",
			len(exponentPlane), elementCount)
	}
	decodedBytes, err := limits.MulChecked(elementCount, dtype.ElementSize(),
		"float tensor decoded byte count")
	if err != nil {
		return nil, err
	}
	if err := limits.CheckLen(decodedBytes, lim.MaxDecodedBytes, "float tensor decoded bytes"); err != nil {
		return nil, err
	}
	mantissaBits, err := limits.MulChecked(elementCount,
		dtype.SignBits+dtype.MantissaBits, "float tensor mantissa bit count")
	if err != nil {
		return nil, err
	}
	expectedMantissaBytes, err := limits.AddChecked(mantissaBits, 7,
		"float tensor mantissa length")
	if err != nil {
		return nil, err
	}
	expectedMantissaBytes /= 8
	if uint64(len(mantissaPlane)) != expectedMantissaBytes {
		return nil, werrors.Corruptf(
			"float tensor mantissa plane has %d bytes but descriptor arithmetic requires %d",
			len(mantissaPlane), expectedMantissaBytes)
	}
	capacity, err := limits.ToInt(decodedBytes, "float tensor decoded bytes")
	if err != nil {
		return nil, err
	}
	capHint := capacity
	if capHint > 1<<20 {
		capHint = 1 << 20
	}
	out := make([]byte, 0, capHint)
	elementBytes := int(dtype.ElementSize())
	var bitPos uint64
	for _, exponentByte := range exponentPlane {
		if uint64(exponentByte)>>uint(dtype.ExponentBits) != 0 {
			return nil, werrors.NewCorrupt(
				"float tensor exponent byte exceeds the dtype exponent width")
		}
		sign := takeBitsReference(mantissaPlane, &bitPos, dtype.SignBits)
		mantissa := takeBitsReference(mantissaPlane, &bitPos, dtype.MantissaBits)
		word := (sign << uint(dtype.ExponentBits+dtype.MantissaBits)) |
			(uint64(exponentByte) << uint(dtype.MantissaBits)) |
			mantissa
		var buf [8]byte
		putLE(buf[:], word)
		out = append(out, buf[:elementBytes]...)
	}
	usedBits := uint64(len(mantissaPlane)) * 8
	padding := usedBits - bitPos
	if padding > 0 {
		last := mantissaPlane[len(mantissaPlane)-1]
		mask := byte(1<<uint(padding)) - 1
		if last&mask != 0 {
			return nil, werrors.NewCorrupt(
				"float tensor mantissa padding bits must be zero")
		}
	}
	return out, nil
}

func putLE(buf []byte, word uint64) {
	for index := 0; index < 8; index++ {
		buf[index] = byte(word >> uint(8*index))
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
func joinAssertSame(
	t *testing.T,
	exponentPlane, mantissaPlane []byte,
	dtype FloatDtype,
	elementCount uint64,
	lim limits.Limits,
	label string,
) {
	t.Helper()
	fastOut, fastErr := JoinFloatTensorPayload(exponentPlane, mantissaPlane, dtype, elementCount, lim)
	refOut, refErr := joinFloatTensorPayloadReference(exponentPlane, mantissaPlane, dtype, elementCount, lim)
	fastClass := werrors.ClassOf(fastErr)
	refClass := werrors.ClassOf(refErr)
	if fastClass != refClass {
		t.Fatalf("%s: %s count %d: class fast=%q reference=%q (fast=%v reference=%v)",
			label, dtype.Name, elementCount, fastClass, refClass, fastErr, refErr)
	}
	if !bytes.Equal(fastOut, refOut) {
		t.Fatalf("%s: %s count %d: fast output differs from reference (%d vs %d bytes)",
			label, dtype.Name, elementCount, len(fastOut), len(refOut))
	}
}

func joinRandomWord(state *uint64, dtype FloatDtype) uint32 {
	bits := dtype.ElementBits()
	mask := uint32(0xffffffff)
	if bits < 32 {
		mask = uint32(uint64(1)<<uint(bits)) - 1
	}
	switch joinXorshift64Star(state) % 16 {
	case 0:
		return 0
	case 1:
		return 1
	case 2:
		return mask
	case 3:
		exponentMask := (uint64(1)<<uint(dtype.ExponentBits) - 1) << uint(dtype.MantissaBits)
		return uint32(exponentMask) & mask
	default:
		return uint32(joinXorshift64Star(state)) & mask
	}
}

func joinRandomPayload(state *uint64, dtype FloatDtype, count int) []byte {
	elementBytes := int(dtype.ElementSize())
	out := make([]byte, count*elementBytes)
	for index := 0; index < count; index++ {
		word := joinRandomWord(state, dtype)
		for b := 0; b < elementBytes; b++ {
			out[index*elementBytes+b] = byte(word >> uint(8*b))
		}
	}
	return out
}

func joinPlanes(t *testing.T, payload []byte, dtype FloatDtype, count uint64) ([]byte, []byte) {
	t.Helper()
	compressed, mantissa, err := SplitFloatTensorPayload(payload, dtype, count)
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	exponent, err := HuffmanDecode(compressed, int(count))
	if err != nil {
		t.Fatalf("huffman decode: %v", err)
	}
	return exponent, mantissa
}

func joinWordsBytes(words []uint32, elementBytes uint64) []byte {
	out := make([]byte, 0, int(elementBytes)*len(words))
	var buf [4]byte
	for _, word := range words {
		for index := 0; index < int(elementBytes); index++ {
			buf[index] = byte(word >> uint(8*index))
		}
		out = append(out, buf[:elementBytes]...)
	}
	return out
}

// ---------------------------------------------------------------------------
// Randomized differential: valid planes
// ---------------------------------------------------------------------------
func TestJoinFloatTensorPayloadMatchesReference(t *testing.T) {
	lim := limits.Default()
	state := uint64(0x9e3779b97f4a7c15)
	cases := 0
	for _, name := range joinAllDtypes {
		dtype := joinDtype(t, name)
		for run := 0; run < 10000; run++ {
			var count int
			switch run {
			case 0:
				count = 0
			case 1:
				count = 1
			case 2:
				count = 2
			case 3:
				count = 3
			case 4:
				count = 7
			default:
				count = int(joinXorshift64Star(&state)%257) + 1
			}
			payload := joinRandomPayload(&state, dtype, count)
			exponent, mantissa := joinPlanes(t, payload, dtype, uint64(count))
			fast, err := JoinFloatTensorPayload(exponent, mantissa, dtype, uint64(count), lim)
			if err != nil {
				t.Fatalf("valid plane rejected for %s count %d: %v", dtype.Name, count, err)
			}
			if !bytes.Equal(fast, payload) {
				t.Fatalf("%s count %d: fast join does not reproduce the source bytes", dtype.Name, count)
			}
			joinAssertSame(t, exponent, mantissa, dtype, uint64(count), lim, "valid plane")
			cases++
		}
	}
	if cases != len(joinAllDtypes)*10000 {
		t.Fatalf("ran %d cases, want %d", cases, len(joinAllDtypes)*10000)
	}
}

// ---------------------------------------------------------------------------
// Randomized differential: mutated planes (error classes must match too)
// ---------------------------------------------------------------------------
func TestJoinFloatTensorPayloadMatchesReferenceOnMutatedPlanes(t *testing.T) {
	lim := limits.Default()
	state := uint64(0x2545f4914f6cdd1d)
	corruptCases := 0
	for _, name := range joinAllDtypes {
		dtype := joinDtype(t, name)
		for run := 0; run < 4000; run++ {
			count := int(joinXorshift64Star(&state)%64) + 1
			payload := joinRandomPayload(&state, dtype, count)
			exponent, mantissa := joinPlanes(t, payload, dtype, uint64(count))
			for flips := joinXorshift64Star(&state)%3 + 1; flips > 0; flips-- {
				if joinXorshift64Star(&state)%2 == 0 {
					if len(exponent) > 0 {
						index := int(joinXorshift64Star(&state) % uint64(len(exponent)))
						exponent[index] ^= byte(1) << uint(joinXorshift64Star(&state)%8)
					}
				} else if len(mantissa) > 0 {
					index := int(joinXorshift64Star(&state) % uint64(len(mantissa)))
					mantissa[index] ^= byte(1) << uint(joinXorshift64Star(&state)%8)
				}
			}
			if _, err := joinFloatTensorPayloadReference(exponent, mantissa, dtype, uint64(count), lim); err != nil {
				corruptCases++
			}
			joinAssertSame(t, exponent, mantissa, dtype, uint64(count), lim, "mutated plane")
		}
	}
	if corruptCases <= 100 {
		t.Fatalf("mutation corpus produced only %d rejects", corruptCases)
	}
}

// ---------------------------------------------------------------------------
// Special values
// ---------------------------------------------------------------------------
func TestJoinFloatTensorPayloadSpecialValues(t *testing.T) {
	lim := limits.Default()
	cases := []struct {
		dtype string
		words []uint32
	}{
		{"bf16", []uint32{0x0000, 0x8000, 0x0001, 0x807f, 0x7f80, 0xff80, 0x7fc1, 0xffef}},
		{"fp16", []uint32{0x0000, 0x8000, 0x0001, 0x03ff, 0x7c00, 0xfc00, 0x7e01, 0xffff}},
		{"fp32", []uint32{0x00000000, 0x80000000, 0x00000001, 0x007fffff,
			0x7f800000, 0xff800000, 0x7f800001, 0xffc00001}},
		{"fp8-e4m3", []uint32{0x00, 0x80, 0x01, 0x07, 0x7f, 0xf0, 0x4f, 0xff}},
		{"fp8-e5m2", []uint32{0x00, 0x80, 0x01, 0x03, 0x7c, 0xfc, 0x7d, 0xff}},
	}
	for _, tc := range cases {
		dtype := joinDtype(t, tc.dtype)
		payload := joinWordsBytes(tc.words, dtype.ElementSize())
		count := uint64(len(tc.words))
		exponent, mantissa := joinPlanes(t, payload, dtype, count)
		fast, err := JoinFloatTensorPayload(exponent, mantissa, dtype, count, lim)
		if err != nil {
			t.Fatalf("%s: fast join: %v", dtype.Name, err)
		}
		if !bytes.Equal(fast, payload) {
			t.Fatalf("%s: special-value output mismatch", dtype.Name)
		}
		joinAssertSame(t, exponent, mantissa, dtype, count, lim, "special values")
	}
}

// ---------------------------------------------------------------------------
// Explicit rejects
// ---------------------------------------------------------------------------
func TestJoinFloatTensorPayloadRejectsOutOfWidthExponent(t *testing.T) {
	lim := limits.Default()
	for _, name := range []string{"fp16", "fp8-e4m3", "fp8-e5m2"} {
		dtype := joinDtype(t, name)
		count := uint64(3)
		payload := joinWordsBytes([]uint32{0, 0, 0}, dtype.ElementSize())
		exponent, mantissa := joinPlanes(t, payload, dtype, count)
		exponent[1] = byte(uint64(1) << uint(dtype.ExponentBits))
		_, err := JoinFloatTensorPayload(exponent, mantissa, dtype, count, lim)
		if werrors.ClassOf(err) != werrors.Corrupt {
			t.Fatalf("%s: out-of-width exponent class = %q, want Corrupt", dtype.Name, werrors.ClassOf(err))
		}
		joinAssertSame(t, exponent, mantissa, dtype, count, lim, "out-of-width exponent")

		if dtype.ExponentBits < 8 {
			allOnes := append([]byte{}, exponent...)
			allOnes[0] = 0xff
			if _, err := JoinFloatTensorPayload(allOnes, mantissa, dtype, count, lim); werrors.ClassOf(err) != werrors.Corrupt {
				t.Fatalf("%s: 0xff exponent class = %q, want Corrupt", dtype.Name, werrors.ClassOf(err))
			}
		}
	}
}

func TestJoinFloatTensorPayloadRejectsNonZeroPadding(t *testing.T) {
	lim := limits.Default()
	cases := []struct {
		dtype string
		count int
	}{
		{"fp16", 1},
		{"fp8-e4m3", 1},
		{"fp8-e5m2", 1},
		{"fp16", 3},
		{"fp8-e5m2", 5},
	}
	for _, tc := range cases {
		dtype := joinDtype(t, tc.dtype)
		payload := joinWordsBytes(make([]uint32, tc.count), dtype.ElementSize())
		exponent, mantissa := joinPlanes(t, payload, dtype, uint64(tc.count))
		usedBits := uint64(tc.count) * (dtype.SignBits + dtype.MantissaBits)
		padding := uint64(len(mantissa))*8 - usedBits
		if padding == 0 {
			t.Fatalf("%s count %d has no padding bits", dtype.Name, tc.count)
		}
		last := len(mantissa) - 1
		if mantissa[last]&(byte(1)<<uint(padding)-1) != 0 {
			t.Fatalf("%s count %d: padding bits are not zero before mutation", dtype.Name, tc.count)
		}
		mantissa[last] |= 1
		_, err := JoinFloatTensorPayload(exponent, mantissa, dtype, uint64(tc.count), lim)
		if werrors.ClassOf(err) != werrors.Corrupt {
			t.Fatalf("%s count %d: non-zero padding class = %q, want Corrupt",
				dtype.Name, tc.count, werrors.ClassOf(err))
		}
		joinAssertSame(t, exponent, mantissa, dtype, uint64(tc.count), lim, "non-zero mantissa padding")
	}
}

func TestJoinFloatTensorPayloadRejectsWrongPlaneLengths(t *testing.T) {
	lim := limits.Default()
	dtype := joinDtype(t, "bf16")
	payload := joinWordsBytes([]uint32{0x3f80, 0x0001, 0x7fc0, 0x8000}, 2)
	count := uint64(4)
	exponent, mantissa := joinPlanes(t, payload, dtype, count)

	// Exponent plane length is part of the descriptor.
	for _, delta := range []int{-1, 1} {
		wrong := append([]byte{}, exponent...)
		if delta < 0 {
			wrong = wrong[:len(wrong)-1]
		} else {
			wrong = append(wrong, 0)
		}
		if _, err := JoinFloatTensorPayload(wrong, mantissa, dtype, count, lim); werrors.ClassOf(err) != werrors.Corrupt {
			t.Fatalf("exponent plane delta %d class = %q, want Corrupt", delta, werrors.ClassOf(err))
		}
	}

	// Mantissa plane length must equal ceil(count * (sign + mantissa) / 8).
	for _, delta := range []int{-1, 1} {
		wrong := append([]byte{}, mantissa...)
		if delta < 0 {
			wrong = wrong[:len(wrong)-1]
		} else {
			wrong = append(wrong, 0)
		}
		if _, err := JoinFloatTensorPayload(exponent, wrong, dtype, count, lim); werrors.ClassOf(err) != werrors.Corrupt {
			t.Fatalf("mantissa plane delta %d class = %q, want Corrupt", delta, werrors.ClassOf(err))
		}
	}

	// The declared element count must match the exponent plane length.
	if _, err := JoinFloatTensorPayload(exponent, mantissa, dtype, count+1, lim); werrors.ClassOf(err) != werrors.Corrupt {
		t.Fatalf("element count class = %q, want Corrupt", werrors.ClassOf(err))
	}

	// The decoded byte limit is enforced before allocation.
	tight := limits.Default()
	tight.MaxDecodedBytes = count*2 - 1
	if _, err := JoinFloatTensorPayload(exponent, mantissa, dtype, count, tight); werrors.ClassOf(err) != werrors.LimitExceeded {
		t.Fatalf("decoded byte limit class = %q, want LimitExceeded", werrors.ClassOf(err))
	}
	joinAssertSame(t, exponent, mantissa, dtype, count, tight, "decoded byte limit")
}

func TestJoinFloatTensorPayloadPaddingBitCounts(t *testing.T) {
	lim := limits.Default()
	state := uint64(0xdeadbeefcafef00d)
	for _, name := range joinAllDtypes {
		dtype := joinDtype(t, name)
		for count := 0; count <= 32; count++ {
			payload := joinRandomPayload(&state, dtype, count)
			exponent, mantissa := joinPlanes(t, payload, dtype, uint64(count))
			fast, err := JoinFloatTensorPayload(exponent, mantissa, dtype, uint64(count), lim)
			if err != nil {
				t.Fatalf("%s count %d: %v", dtype.Name, count, err)
			}
			if !bytes.Equal(fast, payload) {
				t.Fatalf("%s count %d: output mismatch", dtype.Name, count)
			}
			joinAssertSame(t, exponent, mantissa, dtype, uint64(count), lim, "padding-bit counts")
		}
	}
}
