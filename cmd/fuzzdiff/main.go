// Command fuzzdiff is the Go half of the deterministic cross-language
// differential fuzz court (scripts/fuzz-diff.sh in the WeftFS parent repo).
//
// It reads the shared seed corpus, generates exactly the same N inputs as
// WeftCodec/examples/fuzz_differential.rs, decodes each one through
// modes.DecodeBytes with limits.Default(), and prints one tab-separated line
// per input:
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
// For input index i in [0, N):
//
//   - Draw choice = next_u64(). If choice&1 == 0 (probability 1/2): choose
//     seed_index = below(len(seeds)), clone that seed, then apply 1+below(8)
//     mutations in order.
//   - Otherwise emit len = below(2049) pure random bytes, each from
//     next_byte().
//
// Mutation operation below(5):
//
//   - 0 bit flip: if non-empty, pos = below(len), bit = below(8),
//     data[pos] ^= 1 << bit.
//   - 1 byte replace: if non-empty, pos = below(len), value = below(256),
//     data[pos] = value.
//   - 2 insert: if len(data) < 4096, room = 4096 - len(data),
//     n = min(1+below(32), room), pos = below(len(data)+1), insert n bytes from
//     next_byte() at pos.
//   - 3 delete range: if non-empty, start = below(len),
//     n = 1+below(len-start), remove [start, start+n).
//   - 4 truncate: if non-empty, new_len = below(len+1), truncate to it.
//
// The corpus hash is BLAKE3-256 over each generated input in order, framed
// unambiguously as LE64(len(input)) || input. The two runners compare this
// hash before any per-input comparison.
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
	"os"
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

func mutate(r *rng, data []byte) []byte {
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

func generate(r *rng, seeds [][]byte) []byte {
	if r.nextU64()&1 == 0 {
		index := int(r.below(uint64(len(seeds))))
		data := append([]byte(nil), seeds[index]...)
		mutations := 1 + r.below(8)
		for i := uint64(0); i < mutations; i++ {
			data = mutate(r, data)
		}
		return data
	}
	length := int(r.below(maxRandomLen + 1))
	data := make([]byte, length)
	for i := range data {
		data[i] = r.nextByte()
	}
	return data
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
func classify(ctx context.Context, input []byte, resolver *dependency.MapResolver) (kind, payload string, panicked bool) {
	defer func() {
		if recovered := recover(); recovered != nil {
			kind = "panic"
			payload = strings.NewReplacer("\n", " ", "\r", " ", "\t", " ").Replace(fmt.Sprint(recovered))
			panicked = true
		}
	}()
	plaintext, err := modes.DecodeBytes(ctx, input, resolver, limits.Default())
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
	var seedsPath string
	n := uint64(1_000_000)
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
	ctx := context.Background()

	started := time.Now()
	out := bufio.NewWriterSize(os.Stdout, 1<<20)
	hasher := blake3.New(32, nil)
	r := newRng()
	panickedAny := false
	var lengthFrame [8]byte

	for index := uint64(0); index < n; index++ {
		input := generate(r, seeds)
		binary.LittleEndian.PutUint64(lengthFrame[:], uint64(len(input)))
		_, _ = hasher.Write(lengthFrame[:])
		_, _ = hasher.Write(input)

		kind, payload, panicked := classify(ctx, input, resolver)
		if panicked {
			panickedAny = true
		}
		if _, err := fmt.Fprintf(out, "%d\t%s\t%s\n", index, kind, payload); err != nil {
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
	fmt.Fprintln(os.Stderr, "usage: fuzzdiff --seeds <path> [--n <count>]")
	os.Exit(64)
}
