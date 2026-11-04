// Command fuzzdiff is the Go half of the deterministic cross-language
// differential fuzz court (scripts/fuzz-diff.sh in the WeftFS parent repo).
//
// It reads the shared seed corpus, generates exactly the same N inputs as
// WeftCodec/examples/fuzz_differential.rs, decodes each one through the
// codec's normal entry points with limits.Default(), and prints one
// tab-separated line per input:
//
//	<i>\tok\t<content_root_hex>
//	<i>\terr\t<Class>
//	<i>\tpanic\t<message>
//
// At the end it prints metadata lines that are not per-input results:
//
//	corpus_hash\t<blake3-hex>
//	inputs\t<N>
//	elapsed_ms\t<milliseconds>
//	peak_rss_kib\t<kilobytes>
//
// # Generator specification (shared verbatim with the Rust implementation)
//
// PRNG: xorshift64* with initial state 0x9E3779B97F4A7C15.
//
//	next_u64():
//	    x = state
//	    x ^= x >> 12
//	    x ^= x << 25
//	    x ^= x >> 27
//	    state = x
//	    return x * 0x2545F4914F6CDD1D   (wrapping 64-bit)
//
// below(bound) is next_u64() % bound for bound > 0.
//
// next_byte() returns the eight little-endian bytes of successive next_u64()
// outputs, exhausting a buffer before drawing again.
//
// For input index i in [0, N), draw selector = below(8):
//
//   - 0 — raw random (1/8): len = below(2049) bytes, each from next_byte().
//   - 1 — un-repaired mutation (1/8): choose a seed, apply 1+below(8) legacy
//     mutations, no checksum repair.
//   - 2..=5 — repaired mutation (4/8): choose a seed, apply 1+below(4)
//     mutations to its body and/or dependency/chunk table, then recompute the
//     header CRC32C, body length, and trailing CRC32C.
//   - 6 — semantic fault (1/8): a structurally checksum-valid frame with a
//     deliberately wrong dependency hash, wrong chunk extent, wrong mode byte,
//     out-of-range declared size, unknown critical flag, non-canonical
//     structured body, wrong content hash, or a self-referential dependency
//     cycle.
//   - 7 — clean seed (1/8): an unmodified valid seed, which populates the
//     successful-decode / content-root comparison path.
//
// Legacy mutation operation below(5) (branch 1): bit flip, byte replace,
// insert up to 32 bytes (bounded total 4096), delete a range, truncate.
//
// Repaired-body mutation operation below(4) (branches 2..=5): body bit flip,
// body byte replace, body insert/delete (bounded total 4096), or a table byte
// flip/replace. After the mutations the runner parses the header layout from
// the normative offsets in spec/WEFTCODEC-FORMAT.md §2.1–§2.3, sets body_len to
// the actual post-mutation body length, and recomputes header_crc32c over
// [0,72) || [76,96) and frame_crc32c over everything before the trailing four
// bytes. It deliberately does not repair the content hash, so repaired body
// mutations exercise the decoder's real content-hash, WCB, chunk, dependency,
// and mode checks rather than the CRC gate.
//
// Semantic fault operation below(10) (branch 6):
//
//	0,1,2 — replace dependency 0's hash with 0xFF..FF (missing / lying /
//	         denying resolver, respectively).
//	3     — rewrite a seed as a valid REFERENCE frame whose dependency hash is
//	         its own content hash, decoded through the DAG entry point so the
//	         cycle detector runs.
//	4     — set the mode byte to one of 9, 127, 128, 255.
//	5     — declare decoded_len = u64::MAX.
//	6     — set chunk 0's encoded_off past the body.
//	7     — flip one content-hash byte.
//	8     — set unknown critical flag bit 4.
//	9     — append a byte to a STRUCTURED body and bump decoded_len.
//
// Every semantic fault is followed by the same CRC repair, so the frame is
// structurally checksum-valid unless the fault intentionally violates a
// declared length limit.
//
// The corpus hash is BLAKE3-256 over each generated input in order, framed
// unambiguously as LE64(len(input)) || input. The two runners compare this hash
// before any per-input comparison.
//
// Inputs are never retained: the runner streams generation, decoding, and
// output so peak memory stays bounded regardless of N.
//
// The resolver is a fixed in-memory table shared with the Rust runner:
// BASE0 = bytes 0x00..=0xFF, BASE1 = "weftcodec fuzz base payload one\n"
// repeated 8 times, BASE2 = empty, BASE3 = "delta-base-A" repeated 20 times.
// Seeds that use REFERENCE, DELTA, or DICTIONARY reference exactly these bases
// so they decode instead of failing with DependencyMissing.
//
// Panics are recovered per input and reported as a panic line, which forces a
// non-zero exit. This is not coverage-guided fuzzing. It is a deterministic,
// synthetic, CPU-only differential conformance court.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"hash/crc32"
	"os"
	"runtime/debug"
	"strconv"
	"strings"
	"time"

	"lukechampine.com/blake3"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

const (
	prngSeed     uint64 = 0x9E3779B97F4A7C15
	maxRandomLen uint64 = 2048
	maxInputLen         = 4096
	multiplier   uint64 = 0x2545F4914F6CDD1D

	headerLen     = 96
	depEntryLen   = 48
	chunkEntryLen = 48
	trailerLen    = 4
)

var castagnoli = crc32.MakeTable(crc32.Castagnoli)

type inputKind int

const (
	kindStandard inputKind = iota
	kindLying
	kindDenied
	kindCycle
)

// Fixed dependency bases, identical to the Rust runner and seed generator.
func base0() []byte {
	out := make([]byte, 256)
	for i := range out {
		out[i] = byte(i)
	}
	return out
}

func base1() []byte { return bytes.Repeat([]byte("weftcodec fuzz base payload one\n"), 8) }
func base2() []byte { return []byte{} }
func base3() []byte { return bytes.Repeat([]byte("delta-base-A"), 20) }

func buildResolver() *dependency.MapResolver {
	resolver := dependency.NewMapResolver()
	for _, base := range [][]byte{base0(), base1(), base2(), base3()} {
		resolver.InsertBytes(base)
	}
	return resolver
}

type fixedResolver struct {
	denied bool
}

func (f fixedResolver) Resolve(ctx context.Context, h hash.Hash32, decodedLen uint64, depDomain uint8) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, werrors.NewIo(fmt.Sprintf("resolution canceled: %v", err))
	}
	if f.denied {
		return nil, werrors.NewDependencyDenied("fuzz policy denies dependency")
	}
	return []byte("lying!!"), nil
}

type selfFrameSource struct {
	frame []byte
}

func (s selfFrameSource) GetFrame(ctx context.Context, h hash.Hash32) ([]byte, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, werrors.NewIo(fmt.Sprintf("frame source canceled: %v", err))
	}
	return s.frame, true, nil
}

type rng struct {
	state    uint64
	buffer   [8]byte
	position int
}

func newRng() *rng {
	return &rng{state: prngSeed, position: 8}
}

func (r *rng) nextU64() uint64 {
	x := r.state
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	r.state = x
	return x * multiplier
}

func (r *rng) below(bound uint64) uint64 {
	if bound == 0 {
		panic("below(0)")
	}
	return r.nextU64() % bound
}

func (r *rng) nextByte() byte {
	if r.position == len(r.buffer) {
		binary.LittleEndian.PutUint64(r.buffer[:], r.nextU64())
		r.position = 0
	}
	b := r.buffer[r.position]
	r.position++
	return b
}

type frameLayout struct {
	depCount   uint32
	chunkCount uint32
	bodyStart  int
	bodyLen    int
}

// frameLayout locates the body and tables using the normative WFT1 offsets.
// The body length is taken from the actual frame length so an insert/delete
// mutation is repaired rather than misread.
func frameLayoutOf(data []byte) (frameLayout, bool) {
	if len(data) < headerLen+trailerLen || string(data[0:4]) != "WFT1" {
		return frameLayout{}, false
	}
	depCount := binary.LittleEndian.Uint32(data[28:32])
	chunkCount := binary.LittleEndian.Uint32(data[32:36])
	tableLen := binary.LittleEndian.Uint64(data[76:84])
	expected := uint64(depCount)*uint64(depEntryLen) + uint64(chunkCount)*uint64(chunkEntryLen)
	if tableLen != expected || tableLen > uint64(len(data)) {
		return frameLayout{}, false
	}
	bodyStart := headerLen + int(tableLen)
	if bodyStart+trailerLen > len(data) {
		return frameLayout{}, false
	}
	return frameLayout{
		depCount:   depCount,
		chunkCount: chunkCount,
		bodyStart:  bodyStart,
		bodyLen:    len(data) - trailerLen - bodyStart,
	}, true
}

// repairFrame recomputes body_len, the header CRC32C, and the trailing CRC32C.
func repairFrame(data []byte) bool {
	layout, ok := frameLayoutOf(data)
	if !ok {
		return false
	}
	bodyLen := len(data) - trailerLen - layout.bodyStart
	binary.LittleEndian.PutUint64(data[20:28], uint64(bodyLen))

	crcInput := make([]byte, 0, 92)
	crcInput = append(crcInput, data[0:72]...)
	crcInput = append(crcInput, data[76:96]...)
	binary.LittleEndian.PutUint32(data[72:76], crc32.Checksum(crcInput, castagnoli))

	trailerAt := len(data) - trailerLen
	binary.LittleEndian.PutUint32(data[trailerAt:], crc32.Checksum(data[:trailerAt], castagnoli))
	return true
}

func cloneBytes(b []byte) []byte {
	out := make([]byte, len(b))
	copy(out, b)
	return out
}

// legacyMutate is the un-repaired mutation used by branch 1.
func legacyMutate(r *rng, data []byte) []byte {
	switch r.below(5) {
	case 0:
		if len(data) > 0 {
			pos := int(r.below(uint64(len(data))))
			bit := uint(r.below(8))
			data[pos] ^= 1 << bit
		}
	case 1:
		if len(data) > 0 {
			pos := int(r.below(uint64(len(data))))
			data[pos] = byte(r.below(256))
		}
	case 2:
		if len(data) < maxInputLen {
			room := maxInputLen - len(data)
			n := int(1 + r.below(32))
			if n > room {
				n = room
			}
			pos := int(r.below(uint64(len(data)) + 1))
			inserted := make([]byte, n)
			for i := range inserted {
				inserted[i] = r.nextByte()
			}
			out := make([]byte, 0, len(data)+n)
			out = append(out, data[:pos]...)
			out = append(out, inserted...)
			out = append(out, data[pos:]...)
			data = out
		}
	case 3:
		if len(data) > 0 {
			start := int(r.below(uint64(len(data))))
			remaining := len(data) - start
			n := int(1 + r.below(uint64(remaining)))
			data = append(data[:start], data[start+n:]...)
		}
	case 4:
		if len(data) > 0 {
			newLen := int(r.below(uint64(len(data)) + 1))
			data = data[:newLen]
		}
	default:
		panic("below(5) out of range")
	}
	return data
}

func randomBytes(r *rng) []byte {
	length := int(r.below(maxRandomLen + 1))
	data := make([]byte, length)
	for i := range data {
		data[i] = r.nextByte()
	}
	return data
}

func repairedInput(r *rng, seeds [][]byte) ([]byte, inputKind) {
	index := int(r.below(uint64(len(seeds))))
	data := cloneBytes(seeds[index])
	layout, ok := frameLayoutOf(data)
	if !ok {
		return data, kindStandard
	}
	count := 1 + r.below(4)
	for i := uint64(0); i < count; i++ {
		switch r.below(4) {
		case 0:
			if layout.bodyLen > 0 {
				pos := layout.bodyStart + int(r.below(uint64(layout.bodyLen)))
				data[pos] ^= 1 << uint(r.below(8))
			}
		case 1:
			if layout.bodyLen > 0 {
				pos := layout.bodyStart + int(r.below(uint64(layout.bodyLen)))
				data[pos] = byte(r.below(256))
			}
		case 2:
			if r.below(2) == 0 {
				if len(data) < maxInputLen {
					room := maxInputLen - len(data)
					n := int(1 + r.below(16))
					if n > room {
						n = room
					}
					pos := layout.bodyStart + int(r.below(uint64(layout.bodyLen)+1))
					inserted := make([]byte, n)
					for j := range inserted {
						inserted[j] = r.nextByte()
					}
					out := make([]byte, 0, len(data)+n)
					out = append(out, data[:pos]...)
					out = append(out, inserted...)
					out = append(out, data[pos:]...)
					data = out
				}
			} else if layout.bodyLen > 0 {
				start := layout.bodyStart + int(r.below(uint64(layout.bodyLen)))
				remaining := layout.bodyLen - (start - layout.bodyStart)
				n := int(1 + r.below(uint64(remaining)))
				data = append(data[:start], data[start+n:]...)
			}
			if updated, ok := frameLayoutOf(data); ok {
				layout = updated
			}
		case 3:
			entries := uint64(layout.depCount) + uint64(layout.chunkCount)
			if entries > 0 {
				tableLen := layout.bodyStart - headerLen
				off := headerLen + int(r.below(uint64(tableLen)))
				if r.below(2) == 0 {
					data[off] ^= 1 << uint(r.below(8))
				} else {
					data[off] = byte(r.below(256))
				}
			}
		default:
			panic("below(4) out of range")
		}
	}
	repairFrame(data)
	return data, kindStandard
}

func corruptContentHash(r *rng, seeds [][]byte) ([]byte, inputKind) {
	index := int(r.below(uint64(len(seeds))))
	data := cloneBytes(seeds[index])
	data[40] ^= 0x01
	repairFrame(data)
	return data, kindStandard
}

// makeCycle builds a structurally valid REFERENCE frame whose dependency 0
// hash equals its own content hash.
func makeCycle(seed []byte) ([]byte, inputKind) {
	data := make([]byte, headerLen+depEntryLen+trailerLen)
	copy(data[0:4], []byte("WFT1"))
	data[4] = 1
	data[5] = 0
	data[6] = 0
	data[7] = 2 // REFERENCE
	decodedLen := binary.LittleEndian.Uint64(seed[12:20])
	binary.LittleEndian.PutUint64(data[12:20], decodedLen)
	binary.LittleEndian.PutUint32(data[28:32], 1) // dep_count
	binary.LittleEndian.PutUint16(data[36:38], 1) // DEPENDENCIES
	copy(data[40:72], seed[40:72])                // content_hash
	binary.LittleEndian.PutUint64(data[76:84], uint64(depEntryLen))
	copy(data[96:128], seed[40:72]) // self-referential dependency hash
	binary.LittleEndian.PutUint64(data[128:136], decodedLen)
	data[136] = 0 // payload domain
	data[137] = 1 // REQUIRED
	repairFrame(data)
	return data, kindCycle
}

func semanticInput(r *rng, seeds [][]byte, depSeeds, chunkSeeds, structuredSeeds []int) ([]byte, inputKind) {
	fault := r.below(10)
	switch fault {
	case 0, 1, 2:
		if len(depSeeds) == 0 {
			return corruptContentHash(r, seeds)
		}
		index := depSeeds[int(r.below(uint64(len(depSeeds))))]
		data := cloneBytes(seeds[index])
		for i := 96; i < 128; i++ {
			data[i] = 0xFF
		}
		repairFrame(data)
		kind := kindStandard
		if fault == 1 {
			kind = kindLying
		} else if fault == 2 {
			kind = kindDenied
		}
		return data, kind
	case 3:
		if len(depSeeds) == 0 {
			index := int(r.below(uint64(len(seeds))))
			return makeCycle(seeds[index])
		}
		index := depSeeds[int(r.below(uint64(len(depSeeds))))]
		return makeCycle(seeds[index])
	case 4:
		index := int(r.below(uint64(len(seeds))))
		data := cloneBytes(seeds[index])
		values := []byte{9, 127, 128, 255}
		data[7] = values[int(r.below(4))]
		repairFrame(data)
		return data, kindStandard
	case 5:
		index := int(r.below(uint64(len(seeds))))
		data := cloneBytes(seeds[index])
		binary.LittleEndian.PutUint64(data[12:20], ^uint64(0))
		repairFrame(data)
		return data, kindStandard
	case 6:
		if len(chunkSeeds) == 0 {
			return corruptContentHash(r, seeds)
		}
		index := chunkSeeds[int(r.below(uint64(len(chunkSeeds))))]
		data := cloneBytes(seeds[index])
		layout, ok := frameLayoutOf(data)
		if !ok {
			return data, kindStandard
		}
		chunkStart := headerLen + int(layout.depCount)*depEntryLen
		badOffset := uint64(layout.bodyLen) + 8
		binary.LittleEndian.PutUint64(data[chunkStart:chunkStart+8], badOffset)
		repairFrame(data)
		return data, kindStandard
	case 7:
		return corruptContentHash(r, seeds)
	case 8:
		index := int(r.below(uint64(len(seeds))))
		data := cloneBytes(seeds[index])
		flags := binary.LittleEndian.Uint16(data[8:10]) | 0x0010
		binary.LittleEndian.PutUint16(data[8:10], flags)
		repairFrame(data)
		return data, kindStandard
	case 9:
		if len(structuredSeeds) == 0 {
			return corruptContentHash(r, seeds)
		}
		index := structuredSeeds[int(r.below(uint64(len(structuredSeeds))))]
		data := cloneBytes(seeds[index])
		layout, ok := frameLayoutOf(data)
		if !ok {
			return data, kindStandard
		}
		insertAt := layout.bodyStart + layout.bodyLen
		out := make([]byte, 0, len(data)+1)
		out = append(out, data[:insertAt]...)
		out = append(out, 0x00)
		out = append(out, data[insertAt:]...)
		data = out
		decodedLen := binary.LittleEndian.Uint64(data[12:20]) + 1
		binary.LittleEndian.PutUint64(data[12:20], decodedLen)
		repairFrame(data)
		return data, kindStandard
	default:
		panic("below(10) out of range")
	}
}

func generate(r *rng, seeds [][]byte, depSeeds, chunkSeeds, structuredSeeds []int) ([]byte, inputKind) {
	switch r.below(8) {
	case 0:
		return randomBytes(r), kindStandard
	case 1:
		index := int(r.below(uint64(len(seeds))))
		data := cloneBytes(seeds[index])
		mutations := 1 + r.below(8)
		for i := uint64(0); i < mutations; i++ {
			data = legacyMutate(r, data)
		}
		return data, kindStandard
	case 2, 3, 4, 5:
		return repairedInput(r, seeds)
	case 6:
		return semanticInput(r, seeds, depSeeds, chunkSeeds, structuredSeeds)
	case 7:
		index := int(r.below(uint64(len(seeds))))
		return cloneBytes(seeds[index]), kindStandard
	default:
		panic("below(8) out of range")
	}
}

func seedClassLists(seeds [][]byte) (depSeeds, chunkSeeds, structuredSeeds []int) {
	for index, seed := range seeds {
		if layout, ok := frameLayoutOf(seed); ok {
			if layout.depCount > 0 {
				depSeeds = append(depSeeds, index)
			}
			if layout.chunkCount > 0 {
				chunkSeeds = append(chunkSeeds, index)
			}
		}
		if len(seed) > 7 && seed[7] == 5 {
			structuredSeeds = append(structuredSeeds, index)
		}
	}
	return depSeeds, chunkSeeds, structuredSeeds
}

func loadSeeds(path string) ([][]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("cannot open seeds %q: %w", path, err)
	}
	defer file.Close()

	var seeds [][]byte
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			continue
		}
		if len(text)%2 != 0 {
			return nil, fmt.Errorf("seed line %d has odd hex length", lineNumber)
		}
		decoded, err := hex.DecodeString(text)
		if err != nil {
			return nil, fmt.Errorf("seed line %d is not hex: %w", lineNumber, err)
		}
		if len(decoded) == 0 {
			return nil, fmt.Errorf("seed line %d is empty", lineNumber)
		}
		if len(decoded) > maxInputLen {
			return nil, fmt.Errorf(
				"seed line %d is %d bytes, above the %d-byte mutation bound",
				lineNumber, len(decoded), maxInputLen)
		}
		seeds = append(seeds, decoded)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("cannot read seeds %q: %w", path, err)
	}
	if len(seeds) == 0 {
		return nil, fmt.Errorf("seed file %q contains no seeds", path)
	}
	return seeds, nil
}

func peakRSSKiB() uint64 {
	status, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(status), "\n") {
		if !strings.HasPrefix(line, "VmHWM:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0
		}
		value, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			return 0
		}
		return value
	}
	return 0
}

// classify decodes one input. kind is "ok", "err", or "panic"; payload is the
// content root hex, the normative class name, or the recovered panic message.
func classify(
	ctx context.Context,
	input []byte,
	kind inputKind,
	resolver *dependency.MapResolver,
	lying fixedResolver,
	denied fixedResolver,
) (kindName, payload string, panicked bool) {
	defer func() {
		if recovered := recover(); recovered != nil {
			kindName = "panic"
			payload = strings.NewReplacer("\n", " ", "\r", " ", "\t", " ").Replace(fmt.Sprint(recovered))
			panicked = true
		}
	}()
	var plaintext []byte
	var err error
	switch kind {
	case kindLying:
		plaintext, err = modes.DecodeBytes(ctx, input, lying, limits.Default())
	case kindDenied:
		plaintext, err = modes.DecodeBytes(ctx, input, denied, limits.Default())
	case kindCycle:
		plaintext, err = modes.DecodeDAG(ctx, selfFrameSource{frame: input}, input, limits.Default())
	default:
		plaintext, err = modes.DecodeBytes(ctx, input, resolver, limits.Default())
	}
	if err != nil {
		class := string(werrors.ClassOf(err))
		if class == "" {
			// A non-*werrors.Error would be a taxonomy finding, not a
			// classification we may silently invent.
			class = "NoClass"
		}
		return "err", class, false
	}
	return "ok", hash.Domain(hash.DomainChunk, plaintext).Hex(), false
}

func main() {
	// The differential court guards peak RSS at 256 MiB. Go's GC is lazy by
	// default and can hold well above the live heap; pin a soft limit so the
	// driver actually honours the bound instead of relying on GC timing. This
	// is a harness policy, not a codec claim: the codec itself bounds every
	// allocation it makes.
	debug.SetMemoryLimit(192 << 20)
	var seedsPath string
	n := uint64(1_000_000)
	var dumpIndex uint64
	haveDumpIndex := false
	args := os.Args[1:]
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--seeds":
			i++
			if i >= len(args) {
				usage()
			}
			seedsPath = args[i]
		case "--n":
			i++
			if i >= len(args) {
				usage()
			}
			value, err := strconv.ParseUint(args[i], 10, 64)
			if err != nil {
				usage()
			}
			n = value
		case "--dump-index":
			i++
			if i >= len(args) {
				usage()
			}
			value, err := strconv.ParseUint(args[i], 10, 64)
			if err != nil {
				usage()
			}
			dumpIndex = value
			haveDumpIndex = true
		case "-h", "--help":
			usage()
		default:
			fmt.Fprintf(os.Stderr, "unknown argument %q\n", args[i])
			usage()
		}
	}
	if seedsPath == "" {
		usage()
	}

	seeds, err := loadSeeds(seedsPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fuzzdiff: %v\n", err)
		os.Exit(2)
	}
	resolver := buildResolver()
	lying := fixedResolver{}
	denied := fixedResolver{denied: true}
	depSeeds, chunkSeeds, structuredSeeds := seedClassLists(seeds)
	ctx := context.Background()

	if haveDumpIndex {
		r := newRng()
		var selected []byte
		for index := uint64(0); index <= dumpIndex; index++ {
			input, _ := generate(r, seeds, depSeeds, chunkSeeds, structuredSeeds)
			if index == dumpIndex {
				selected = input
			}
		}
		fmt.Println(hex.EncodeToString(selected))
		return
	}

	started := time.Now()
	out := bufio.NewWriterSize(os.Stdout, 1<<20)
	hasher := blake3.New(32, nil)
	r := newRng()
	panickedAny := false
	var lengthFrame [8]byte

	for index := uint64(0); index < n; index++ {
		input, kind := generate(r, seeds, depSeeds, chunkSeeds, structuredSeeds)
		binary.LittleEndian.PutUint64(lengthFrame[:], uint64(len(input)))
		_, _ = hasher.Write(lengthFrame[:])
		_, _ = hasher.Write(input)

		kindName, payload, panicked := classify(ctx, input, kind, resolver, lying, denied)
		if panicked {
			panickedAny = true
		}
		if _, err := fmt.Fprintf(out, "%d\t%s\t%s\n", index, kindName, payload); err != nil {
			fmt.Fprintf(os.Stderr, "fuzzdiff: cannot write output: %v\n", err)
			os.Exit(2)
		}
	}

	corpusHash := hex.EncodeToString(hasher.Sum(nil))
	elapsed := time.Since(started).Milliseconds()
	rss := peakRSSKiB()
	if _, err := fmt.Fprintf(out, "corpus_hash\t%s\ninputs\t%d\nelapsed_ms\t%d\npeak_rss_kib\t%d\n",
		corpusHash, n, elapsed, rss); err != nil {
		fmt.Fprintf(os.Stderr, "fuzzdiff: cannot write metadata: %v\n", err)
		os.Exit(2)
	}
	if err := out.Flush(); err != nil {
		fmt.Fprintf(os.Stderr, "fuzzdiff: cannot flush output: %v\n", err)
		os.Exit(2)
	}
	if panickedAny {
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: fuzzdiff --seeds <path> [--n <count>] [--dump-index <index>]")
	os.Exit(64)
}
