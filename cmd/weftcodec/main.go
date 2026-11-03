// Command weftcodec is the WeftCodec v1 command-line interface.
//
// Verbs: encode, decode, inspect, verify, archive create/decode,
// selector id/canonicalize. Errors print their normative class and exit 1.
package main

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"unicode/utf8"

	"github.com/infinityabundance/WeftCodec-Go/archive"
	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/modes"
	"github.com/infinityabundance/WeftCodec-Go/selector"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	ctx := context.Background()
	var err error
	switch os.Args[1] {
	case "encode":
		err = runEncode(ctx, os.Args[2:])
	case "decode":
		err = runDecode(ctx, os.Args[2:])
	case "inspect":
		err = runInspect(os.Args[2:])
	case "verify":
		err = runVerify(ctx, os.Args[2:])
	case "archive":
		err = runArchive(ctx, os.Args[2:])
	case "selector":
		err = runSelector(os.Args[2:])
	case "-h", "--help", "help":
		usage()
		return
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n", os.Args[1])
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `weftcodec — WeftCodec v1 encoder/decoder

usage:
  weftcodec encode --input FILE --output FILE [--mode MODE] [--base FILE] [--dict FILE] [--level N] [--chunk-size N]
  weftcodec decode --input FILE --output FILE [--deps DIR]
  weftcodec inspect FILE [--json]
  weftcodec verify FILE [--deps DIR]
  weftcodec archive create --output FILE FRAME...
  weftcodec archive decode --input FILE --out-dir DIR
  weftcodec selector id --input FILE [--json]
  weftcodec selector canonicalize --input FILE`)
}

func runEncode(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("encode", flag.ContinueOnError)
	input := fs.String("input", "", "input path or - for stdin")
	output := fs.String("output", "", "output path or - for stdout")
	modeName := fs.String("mode", "auto", "raw|zstd|reference|delta|dictionary|structured|auto")
	basePath := fs.String("base", "", "base bytes for reference/delta/auto")
	dictPath := fs.String("dict", "", "raw-content zstd dictionary for dictionary/auto")
	level := fs.Int("level", 3, "zstd level")
	chunkSize := fs.Int("chunk-size", 0, "chunk size in bytes (RAW and ZSTD only)")
	if err := fs.Parse(args); err != nil {
		return werrors.NewParse(err.Error())
	}
	if *input == "" || *output == "" {
		return werrors.NewParse("encode requires --input and --output")
	}
	lim := limits.Default()
	data, err := readLimited(*input, lim.MaxDecodedBytes)
	if err != nil {
		return err
	}
	var base, dict []byte
	if *basePath != "" {
		base, err = readLimited(*basePath, lim.MaxDecodedBytes)
		if err != nil {
			return err
		}
	}
	if *dictPath != "" {
		dict, err = readLimited(*dictPath, lim.MaxDictionaryBytes)
		if err != nil {
			return err
		}
	}

	var encoded []byte
	if *modeName == "auto" {
		encoded, err = encodeAuto(data, base, dict, *level, *chunkSize, lim)
	} else {
		mode, modeErr := frame.ModeFromName(*modeName)
		if modeErr != nil {
			return modeErr
		}
		encoded, err = encodeMode(mode, data, base, dict, *level, *chunkSize, lim)
	}
	if err != nil {
		return err
	}
	return writeOutput(*output, encoded)
}

func encodeMode(mode frame.Mode, data, base, dict []byte, level, chunkSize int, lim limits.Limits) ([]byte, error) {
	switch mode {
	case frame.ModeRaw:
		return modes.EncodeRaw(data, chunkSize, lim)
	case frame.ModeZstd:
		return modes.EncodeZstd(data, level, chunkSize, lim)
	case frame.ModeReference:
		if base == nil {
			return nil, werrors.NewMalformed("REFERENCE encoding requires --base")
		}
		return modes.EncodeReference(data, base, lim)
	case frame.ModeDelta:
		if base == nil {
			return nil, werrors.NewMalformed("DELTA encoding requires --base")
		}
		return modes.EncodeDelta(data, base, lim)
	case frame.ModeDictionary:
		if dict == nil {
			return nil, werrors.NewMalformed("DICTIONARY encoding requires --dict")
		}
		return modes.EncodeDictionary(data, dict, level, lim)
	case frame.ModeStructured:
		value, err := wcb.Decode(data, lim)
		if err != nil {
			return nil, err
		}
		return modes.EncodeStructured(value, lim)
	default:
		return nil, werrors.UnsupportedModef("unknown mode %s", mode)
	}
}

type candidate struct {
	mode  uint8
	frame []byte
}

func encodeAuto(data, base, dict []byte, level, chunkSize int, lim limits.Limits) ([]byte, error) {
	candidates := make([]candidate, 0, 4)
	raw, err := modes.EncodeRaw(data, chunkSize, lim)
	if err != nil {
		return nil, err
	}
	candidates = append(candidates, candidate{mode: uint8(frame.ModeRaw), frame: raw})
	zstd, err := modes.EncodeZstd(data, level, chunkSize, lim)
	if err != nil {
		return nil, err
	}
	candidates = append(candidates, candidate{mode: uint8(frame.ModeZstd), frame: zstd})
	if base != nil {
		if bytesEqual(base, data) {
			ref, err := modes.EncodeReference(data, base, lim)
			if err != nil {
				return nil, err
			}
			candidates = append(candidates, candidate{mode: uint8(frame.ModeReference), frame: ref})
		}
		delta, err := modes.EncodeDelta(data, base, lim)
		if err != nil {
			return nil, err
		}
		candidates = append(candidates, candidate{mode: uint8(frame.ModeDelta), frame: delta})
	}
	if dict != nil {
		dictionary, err := modes.EncodeDictionary(data, dict, level, lim)
		if err != nil {
			return nil, err
		}
		candidates = append(candidates, candidate{mode: uint8(frame.ModeDictionary), frame: dictionary})
	}
	best := candidates[0]
	for _, c := range candidates[1:] {
		if len(c.frame) < len(best.frame) || (len(c.frame) == len(best.frame) && c.mode < best.mode) {
			best = c
		}
	}
	return best.frame, nil
}

func runDecode(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("decode", flag.ContinueOnError)
	input := fs.String("input", "", "frame path or - for stdin")
	output := fs.String("output", "", "plaintext path or - for stdout")
	deps := fs.String("deps", "", "directory of dependency files named by lowercase hex digest")
	if err := fs.Parse(args); err != nil {
		return werrors.NewParse(err.Error())
	}
	if *input == "" || *output == "" {
		return werrors.NewParse("decode requires --input and --output")
	}
	lim := limits.Default()
	frameBytes, err := readLimited(*input, lim.MaxFrameBytes)
	if err != nil {
		return err
	}
	resolver := resolverFor(*deps)
	plaintext, err := modes.DecodeBytes(ctx, frameBytes, resolver, lim)
	if err != nil {
		return err
	}
	return writeOutput(*output, plaintext)
}

func runVerify(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("verify", flag.ContinueOnError)
	deps := fs.String("deps", "", "directory of dependency files named by lowercase hex digest")
	if err := fs.Parse(args); err != nil {
		return werrors.NewParse(err.Error())
	}
	if fs.NArg() < 1 {
		return werrors.NewParse("verify requires a frame file")
	}
	lim := limits.Default()
	frameBytes, err := readLimited(fs.Arg(0), lim.MaxFrameBytes)
	if err != nil {
		return err
	}
	resolver := resolverFor(*deps)
	plaintext, err := modes.DecodeBytes(ctx, frameBytes, resolver, lim)
	if err != nil {
		return err
	}
	f, err := frame.Parse(frameBytes, lim)
	if err != nil {
		return err
	}
	fmt.Printf("OK mode=%s decoded_len=%d content_hash=%s chunks=%d dependencies=%d\n",
		f.Header.Mode, len(plaintext), f.Header.ContentHash,
		f.Header.ChunkCount, f.Header.DepCount)
	return nil
}

func runInspect(args []string) error {
	fs := flag.NewFlagSet("inspect", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	inputFlag := fs.String("input", "", "frame file")
	if err := fs.Parse(args); err != nil {
		return werrors.NewParse(err.Error())
	}
	file := *inputFlag
	if file == "" {
		if fs.NArg() < 1 {
			return werrors.NewParse("inspect requires a frame file")
		}
		file = fs.Arg(0)
		for _, extra := range fs.Args()[1:] {
			if extra == "--json" || extra == "-json" {
				*asJSON = true
			}
		}
	}
	lim := limits.Default()
	frameBytes, err := readLimited(file, lim.MaxFrameBytes)
	if err != nil {
		return err
	}
	f, err := frame.Parse(frameBytes, lim)
	if err != nil {
		return err
	}
	h := f.Header
	if *asJSON {
		deps := make([]map[string]any, 0, len(f.Dependencies))
		for _, dep := range f.Dependencies {
			deps = append(deps, map[string]any{
				"hash":        dep.Hash.Hex(),
				"decoded_len": dep.DecodedLen,
				"dep_domain":  dep.DepDomain,
				"dep_flags":   dep.DepFlags,
				"codec_hint":  dep.CodecHint,
			})
		}
		chunks := make([]map[string]any, 0, len(f.Chunks))
		for _, chunk := range f.Chunks {
			chunks = append(chunks, map[string]any{
				"encoded_off": chunk.EncodedOff,
				"encoded_len": chunk.EncodedLen,
				"decoded_len": chunk.DecodedLen,
				"hash":        chunk.Hash.Hex(),
			})
		}
		domainName := "payload"
		if h.Domain == frame.DomainStructured {
			domainName = "structured"
		}
		output := map[string]any{
			"magic":         "WFT1",
			"format_major":  h.FormatMajor,
			"format_minor":  h.FormatMinor,
			"domain":        h.Domain,
			"domain_name":   domainName,
			"mode":          h.Mode.String(),
			"mode_value":    uint8(h.Mode),
			"flags":         h.Flags,
			"flag_names":    h.FlagNames(),
			"features":      h.Features,
			"feature_names": h.FeatureNames(),
			"level":         h.Level,
			"decoded_len":   h.DecodedLen,
			"body_len":      h.BodyLen,
			"table_len":     h.TableLen,
			"frame_len":     len(frameBytes),
			"dep_count":     h.DepCount,
			"chunk_count":   h.ChunkCount,
			"content_hash":  h.ContentHash.Hex(),
			"dependencies":  deps,
			"chunks":        chunks,
		}
		encoded, err := json.MarshalIndent(output, "", "  ")
		if err != nil {
			return werrors.NewIo("cannot serialize inspection: " + err.Error())
		}
		fmt.Println(string(encoded))
		return nil
	}
	domainName := "payload"
	if h.Domain == frame.DomainStructured {
		domainName = "structured"
	}
	fmt.Printf("format: WFT1 %d.%d\n", h.FormatMajor, h.FormatMinor)
	fmt.Printf("domain: %d (%s)\n", h.Domain, domainName)
	fmt.Printf("mode: %s (%d)\n", h.Mode, uint8(h.Mode))
	fmt.Printf("flags: 0x%04x %v\n", h.Flags, h.FlagNames())
	fmt.Printf("features: 0x%04x %v\n", h.Features, h.FeatureNames())
	fmt.Printf("level: %d\n", h.Level)
	fmt.Printf("decoded_len: %d\n", h.DecodedLen)
	fmt.Printf("body_len: %d\n", h.BodyLen)
	fmt.Printf("table_len: %d\n", h.TableLen)
	fmt.Printf("frame_len: %d\n", len(frameBytes))
	fmt.Printf("content_hash: %s\n", h.ContentHash)
	fmt.Printf("dependencies: %d\n", h.DepCount)
	for i, dep := range f.Dependencies {
		fmt.Printf("  [%d] hash=%s decoded_len=%d domain=%d hint=%d\n",
			i, dep.Hash, dep.DecodedLen, dep.DepDomain, dep.CodecHint)
	}
	fmt.Printf("chunks: %d\n", h.ChunkCount)
	for i, chunk := range f.Chunks {
		fmt.Printf("  [%d] off=%d encoded_len=%d decoded_len=%d hash=%s\n",
			i, chunk.EncodedOff, chunk.EncodedLen, chunk.DecodedLen, chunk.Hash)
	}
	return nil
}

func runArchive(ctx context.Context, args []string) error {
	if len(args) < 1 {
		return werrors.NewParse("archive requires create or decode")
	}
	switch args[0] {
	case "create":
		fs := flag.NewFlagSet("archive create", flag.ContinueOnError)
		output := fs.String("output", "", "archive output path")
		if err := fs.Parse(args[1:]); err != nil {
			return werrors.NewParse(err.Error())
		}
		if *output == "" || fs.NArg() < 1 {
			return werrors.NewParse("archive create requires --output and at least one frame")
		}
		lim := limits.Default()
		frames := make([][]byte, 0, fs.NArg())
		for _, path := range fs.Args() {
			b, err := readLimited(path, lim.MaxFrameBytes)
			if err != nil {
				return err
			}
			frames = append(frames, b)
		}
		encoded, err := archive.Create(frames, lim)
		if err != nil {
			return err
		}
		return writeOutput(*output, encoded)
	case "decode":
		fs := flag.NewFlagSet("archive decode", flag.ContinueOnError)
		input := fs.String("input", "", "archive input path")
		outDir := fs.String("out-dir", "", "directory for decoded plaintext files")
		if err := fs.Parse(args[1:]); err != nil {
			return werrors.NewParse(err.Error())
		}
		if *input == "" || *outDir == "" {
			return werrors.NewParse("archive decode requires --input and --out-dir")
		}
		lim := limits.Default()
		b, err := readLimited(*input, lim.MaxTotalFetchedBytes)
		if err != nil {
			return err
		}
		entries, err := archive.Decode(b, lim)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(*outDir, 0o755); err != nil {
			return werrors.NewIo("cannot create " + *outDir + ": " + err.Error())
		}
		for _, entry := range entries {
			path := filepath.Join(*outDir, entry.ContentHash.Hex())
			if err := os.WriteFile(path, entry.Bytes, 0o644); err != nil {
				return werrors.NewIo("cannot write " + path + ": " + err.Error())
			}
			fmt.Printf("%s %d\n", entry.ContentHash, len(entry.Bytes))
		}
		return nil
	default:
		return werrors.NewParse("archive requires create or decode")
	}
}

func runSelector(args []string) error {
	if len(args) < 1 {
		return werrors.NewParse("selector requires id or canonicalize")
	}
	switch args[0] {
	case "id":
		fs := flag.NewFlagSet("selector id", flag.ContinueOnError)
		input := fs.String("input", "", "selector source file")
		asJSON := fs.Bool("json", false, "emit JSON")
		if err := fs.Parse(args[1:]); err != nil {
			return werrors.NewParse(err.Error())
		}
		if *input == "" {
			return werrors.NewParse("selector id requires --input")
		}
		lim := limits.Default()
		b, err := readLimited(*input, lim.MaxSelectorBytes)
		if err != nil {
			return err
		}
		if !utf8.Valid(b) {
			return werrors.NewParse("selector input is not valid UTF-8")
		}
		s, err := selector.ParseWith(string(b), selector.DefaultNamespace, lim)
		if err != nil {
			return err
		}
		canonical, err := s.CanonicalBytes(lim)
		if err != nil {
			return err
		}
		id := hash.Domain(hash.DomainSelector, canonical)
		if *asJSON {
			output := map[string]any{
				"schema":        selector.Schema,
				"namespace":     s.Namespace,
				"selector_id":   id.Hex(),
				"canonical_hex": hex.EncodeToString(canonical),
				"canonical_len": len(canonical),
			}
			encoded, err := json.MarshalIndent(output, "", "  ")
			if err != nil {
				return werrors.NewIo("cannot serialize selector id: " + err.Error())
			}
			fmt.Println(string(encoded))
		} else {
			fmt.Printf("%x\n", id)
		}
		return nil
	case "canonicalize":
		fs := flag.NewFlagSet("selector canonicalize", flag.ContinueOnError)
		input := fs.String("input", "", "selector source file")
		if err := fs.Parse(args[1:]); err != nil {
			return werrors.NewParse(err.Error())
		}
		if *input == "" {
			return werrors.NewParse("selector canonicalize requires --input")
		}
		lim := limits.Default()
		b, err := readLimited(*input, lim.MaxSelectorBytes)
		if err != nil {
			return err
		}
		if !utf8.Valid(b) {
			return werrors.NewParse("selector input is not valid UTF-8")
		}
		s, err := selector.ParseWith(string(b), selector.DefaultNamespace, lim)
		if err != nil {
			return err
		}
		canonical, err := s.CanonicalBytes(lim)
		if err != nil {
			return err
		}
		fmt.Println(hex.EncodeToString(canonical))
		return nil
	default:
		return werrors.NewParse("selector requires id or canonicalize")
	}
}

func resolverFor(dir string) dependency.Resolver {
	if dir != "" {
		return dependency.NewDirectoryResolver(dir)
	}
	return dependency.NewMapResolver()
}

func readLimited(path string, max uint64) ([]byte, error) {
	if path == "-" {
		return readLimitedReader(os.Stdin, max)
	}
	info, err := os.Stat(path)
	if err != nil {
		return nil, werrors.NewIo(fmt.Sprintf("cannot stat %s: %v", path, err))
	}
	if uint64(info.Size()) > max {
		return nil, werrors.LimitExceededf(
			"%s has %d bytes, limit %d", path, info.Size(), max)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, werrors.NewIo(fmt.Sprintf("cannot read %s: %v", path, err))
	}
	if uint64(len(b)) > max {
		return nil, werrors.LimitExceededf(
			"%s grew to %d bytes while reading, limit %d", path, len(b), max)
	}
	return b, nil
}

func readLimitedReader(r io.Reader, max uint64) ([]byte, error) {
	limit := int64(max)
	if max >= uint64(^uint64(0)>>1) {
		limit = int64(^uint64(0) >> 1)
	}
	b, err := io.ReadAll(io.LimitReader(r, limit+1))
	if err != nil {
		return nil, werrors.NewIo("cannot read input: " + err.Error())
	}
	if uint64(len(b)) > max {
		return nil, werrors.LimitExceededf("input exceeds limit %d bytes", max)
	}
	return b, nil
}

func writeOutput(path string, b []byte) error {
	if path == "-" {
		if _, err := os.Stdout.Write(b); err != nil {
			return werrors.NewIo("cannot write stdout: " + err.Error())
		}
		return nil
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		return werrors.NewIo(fmt.Sprintf("cannot write %s: %v", path, err))
	}
	return nil
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
