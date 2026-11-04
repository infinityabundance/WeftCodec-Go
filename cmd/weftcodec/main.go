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
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/infinityabundance/WeftCodec-Go/archive"
	weftcontext "github.com/infinityabundance/WeftCodec-Go/context"
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
	case "context":
		err = runContext(os.Args[2:])
	case "causal":
		err = runCausal(os.Args[2:])
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
  weftcodec encode --mode tensor --input PAYLOAD --dtype D --shape 2,3 | --metadata FILE
  weftcodec encode --mode quantized --input PAYLOAD --dtype i8|u8 --shape 2,3 --scale F [--zero-point N]
  weftcodec encode --mode float-tensor --input PAYLOAD --dtype D --shape 2,3 | --metadata FILE
  weftcodec encode --mode tokens --input TOKENS.json
  weftcodec decode --input FILE --output FILE [--deps DIR]
  weftcodec inspect FILE [--json]
  weftcodec verify FILE [--deps DIR]
  weftcodec archive create --output FILE FRAME...
  weftcodec archive decode --input FILE --out-dir DIR
  weftcodec selector id --input FILE [--json]
  weftcodec selector canonicalize --input FILE
  weftcodec context id --selector FILE|- --namespace N --snapshot HEX [--causal HEX,...] [--resolver R] [--capability-hash HEX] [--budget K=V,...] [--json]
  weftcodec context uri --selector FILE|- --namespace N --snapshot HEX [--descriptor] [--json]
  weftcodec context parse weft://... [--json]
  weftcodec causal id --parents HEX,... [--json]`)
}

func runEncode(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("encode", flag.ContinueOnError)
	input := fs.String("input", "", "input path or - for stdin")
	output := fs.String("output", "", "output path or - for stdout")
	modeName := fs.String("mode", "auto", "raw|zstd|reference|delta|dictionary|structured|tensor|tokens|quantized|float-tensor|auto")
	basePath := fs.String("base", "", "base bytes for reference/delta/auto")
	dictPath := fs.String("dict", "", "raw-content zstd dictionary for dictionary/auto")
	level := fs.Int("level", 3, "zstd level")
	chunkSize := fs.Int("chunk-size", 0, "chunk size in bytes (RAW and ZSTD only)")
	dtype := fs.String("dtype", "", "TENSOR dtype (f16|bf16|f32|f64|i8|u8|i16|i32|i64)")
	shape := fs.String("shape", "", "TENSOR shape as comma-separated u64 extents")
	metadata := fs.String("metadata", "", "TENSOR metadata JSON file")
	scale := fs.Float64("scale", 0, "QUANTIZED_TENSOR affine scale (finite, > 0)")
	zeroPoint := fs.Int64("zero-point", 0, "QUANTIZED_TENSOR zero point")
	if err := fs.Parse(args); err != nil {
		return werrors.NewParse(err.Error())
	}
	scaleSet := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "scale" {
			scaleSet = true
		}
	})
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
	switch *modeName {
	case "tensor":
		encoded, err = encodeTensor(data, *dtype, *shape, *metadata, lim)
	case "quantized", "quantized_tensor":
		encoded, err = encodeQuantizedTensor(data, *dtype, *shape, *metadata,
			*scale, scaleSet, *zeroPoint, lim)
	case "float-tensor", "float_tensor":
		encoded, err = encodeFloatTensor(data, *dtype, *shape, *metadata, lim)
	case "tokens":
		encoded, err = encodeTokens(data, lim)
	case "auto":
		encoded, err = encodeAuto(data, base, dict, *level, *chunkSize, lim)
	default:
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

type tensorMetaInput struct {
	Dtype      string   `json:"dtype"`
	Shape      []uint64 `json:"shape"`
	Layout     string   `json:"layout"`
	Endianness string   `json:"endianness"`
	Scale      *float64 `json:"scale"`
	ZeroPoint  *int64   `json:"zero_point"`
}

type tokensInput struct {
	VocabSchema string   `json:"vocab_schema"`
	TokenIDs    []uint64 `json:"token_ids"`
}

func encodeTensor(payload []byte, dtype, shapeText, metadataPath string, lim limits.Limits) ([]byte, error) {
	shape := []uint64{}
	if metadataPath != "" {
		if dtype != "" || shapeText != "" {
			return nil, werrors.NewMalformed(
				"--metadata cannot be combined with --dtype or --shape")
		}
		raw, err := readLimited(metadataPath, lim.MaxStringBytes)
		if err != nil {
			return nil, err
		}
		var input tensorMetaInput
		if err := json.Unmarshal(raw, &input); err != nil {
			return nil, werrors.Malformedf(
				"tensor metadata must be JSON {\"dtype\":string,\"shape\":[u64,...]}: %v", err)
		}
		if input.Layout != "" && input.Layout != "row_major" {
			return nil, werrors.Malformedf(
				"tensor layout %q is specified but not implemented in v1 (only row_major)",
				input.Layout)
		}
		if input.Endianness != "" && input.Endianness != "little" {
			return nil, werrors.Malformedf(
				"tensor endianness %q is specified but not implemented in v1 (only little)",
				input.Endianness)
		}
		dtype = input.Dtype
		shape = input.Shape
	} else {
		if dtype == "" {
			return nil, werrors.NewMalformed(
				"TENSOR encoding requires --dtype or --metadata")
		}
		if shapeText != "" {
			for _, part := range strings.Split(shapeText, ",") {
				value, err := strconv.ParseUint(strings.TrimSpace(part), 10, 64)
				if err != nil {
					return nil, werrors.Malformedf("bad --shape extent %q: %v", part, err)
				}
				shape = append(shape, value)
			}
		}
	}
	return modes.EncodeTensor(payload, dtype, shape, lim)
}

func encodeTokens(data []byte, lim limits.Limits) ([]byte, error) {
	var input tokensInput
	if err := json.Unmarshal(data, &input); err != nil {
		return nil, werrors.Malformedf(
			"tokens input must be JSON {\"vocab_schema\":string,\"token_ids\":[u64,...]}: %v", err)
	}
	return modes.EncodeTokens(input.TokenIDs, input.VocabSchema, lim)
}

func encodeQuantizedTensor(
	payload []byte, dtype, shapeText, metadataPath string,
	scale float64, scaleSet bool, zeroPoint int64, lim limits.Limits,
) ([]byte, error) {
	shape := []uint64{}
	if metadataPath != "" {
		if dtype != "" || shapeText != "" || scaleSet {
			return nil, werrors.NewMalformed(
				"--metadata cannot be combined with --dtype/--shape/--scale")
		}
		b, err := os.ReadFile(metadataPath)
		if err != nil {
			return nil, err
		}
		var input tensorMetaInput
		if err := json.Unmarshal(b, &input); err != nil {
			return nil, werrors.Malformedf(
				"quantized tensor metadata must be JSON {\"dtype\":string,\"shape\":[u64,...],\"scale\":number,\"zero_point\":integer}: %v",
				err)
		}
		if input.Layout != "" && input.Layout != "row_major" {
			return nil, werrors.Malformedf(
				"quantized tensor layout %q is specified but not implemented in v1 (only row_major)",
				input.Layout)
		}
		if input.Scale == nil {
			return nil, werrors.NewMalformed(
				"quantized tensor metadata requires a `scale` number")
		}
		scale = *input.Scale
		shape = input.Shape
		if input.ZeroPoint != nil {
			zeroPoint = *input.ZeroPoint
		}
		return modes.EncodeQuantizedTensor(payload, input.Dtype, shape, scale, zeroPoint, lim)
	}
	if dtype == "" {
		return nil, werrors.NewMalformed(
			"QUANTIZED_TENSOR encoding requires --dtype or --metadata")
	}
	if !scaleSet {
		return nil, werrors.NewMalformed("QUANTIZED_TENSOR encoding requires --scale")
	}
	if shapeText != "" {
		for _, part := range strings.Split(shapeText, ",") {
			value, err := strconv.ParseUint(strings.TrimSpace(part), 10, 64)
			if err != nil {
				return nil, werrors.Malformedf("bad --shape extent %q: %v", part, err)
			}
			shape = append(shape, value)
		}
	}
	return modes.EncodeQuantizedTensor(payload, dtype, shape, scale, zeroPoint, lim)
}

func encodeFloatTensor(
	payload []byte, dtype, shapeText, metadataPath string, lim limits.Limits,
) ([]byte, error) {
	shape := []uint64{}
	if metadataPath != "" {
		if dtype != "" || shapeText != "" {
			return nil, werrors.NewMalformed(
				"--metadata cannot be combined with --dtype or --shape")
		}
		raw, err := readLimited(metadataPath, lim.MaxStringBytes)
		if err != nil {
			return nil, err
		}
		var input tensorMetaInput
		if err := json.Unmarshal(raw, &input); err != nil {
			return nil, werrors.Malformedf(
				"float tensor metadata must be JSON {\"dtype\":string,\"shape\":[u64,...]}: %v", err)
		}
		dtype = input.Dtype
		shape = input.Shape
	} else {
		if dtype == "" {
			return nil, werrors.NewMalformed(
				"FLOAT_TENSOR encoding requires --dtype or --metadata")
		}
		if shapeText != "" {
			for _, part := range strings.Split(shapeText, ",") {
				value, err := strconv.ParseUint(strings.TrimSpace(part), 10, 64)
				if err != nil {
					return nil, werrors.Malformedf("bad --shape extent %q: %v", part, err)
				}
				shape = append(shape, value)
			}
		}
	}
	parsed, err := modes.ParseFloatDtype(dtype)
	if err != nil {
		return nil, err
	}
	return modes.EncodeFloatTensor(payload, parsed, shape, lim)
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
	case frame.ModeTensor:
		return nil, werrors.NewMalformed(
			"TENSOR encoding requires --dtype/--shape or --metadata")
	case frame.ModeTokenSequence:
		return nil, werrors.NewMalformed(
			"TOKEN_SEQUENCE encoding requires a JSON token input")
	case frame.ModeQuantizedTensor:
		return nil, werrors.NewMalformed(
			"QUANTIZED_TENSOR encoding requires the dedicated CLI path with --dtype/--shape/--scale/--zero-point")
	case frame.ModeFloatTensor:
		return nil, werrors.NewMalformed(
			"FLOAT_TENSOR encoding requires the dedicated CLI path with --dtype/--shape/--metadata")
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

// addLimitFlags registers the decoder resource bounds. Defaults are the
// normative v1 values; each flag can only lower its bound.
func addLimitFlags(fs *flag.FlagSet, lim *limits.Limits) {
	fs.Uint64Var(&lim.MaxFrameBytes, "max-frame-bytes", lim.MaxFrameBytes, "maximum encoded frame bytes")
	fs.Uint64Var(&lim.MaxDecodedBytes, "max-decoded-bytes", lim.MaxDecodedBytes, "maximum decoded bytes")
	fs.Uint64Var(&lim.MaxDependencyDepth, "max-dependency-depth", lim.MaxDependencyDepth, "maximum dependency DAG depth")
	fs.Uint64Var(&lim.MaxDependencyCount, "max-dependency-count", lim.MaxDependencyCount, "maximum dependency table entries")
	fs.Uint64Var(&lim.MaxTotalFetchedBytes, "max-total-fetched-bytes", lim.MaxTotalFetchedBytes, "maximum fetched dependency frame bytes")
	fs.Uint64Var(&lim.MaxChunks, "max-chunks", lim.MaxChunks, "maximum chunk table entries")
	fs.Uint64Var(&lim.MaxDictionaryBytes, "max-dictionary-bytes", lim.MaxDictionaryBytes, "maximum dictionary bytes")
	fs.Uint64Var(&lim.MaxDeltaOps, "max-delta-ops", lim.MaxDeltaOps, "maximum delta operations")
	fs.Uint64Var(&lim.MaxStructuredDepth, "max-structured-depth", lim.MaxStructuredDepth, "maximum WCB nesting depth")
	fs.Uint64Var(&lim.MaxStructuredItems, "max-structured-items", lim.MaxStructuredItems, "maximum WCB item count")
	fs.Uint64Var(&lim.MaxSelectorBytes, "max-selector-bytes", lim.MaxSelectorBytes, "maximum selector source bytes")
	fs.Uint64Var(&lim.MaxSelectorDepth, "max-selector-depth", lim.MaxSelectorDepth, "maximum selector AST depth")
	fs.Uint64Var(&lim.MaxSelectorOperands, "max-selector-operands", lim.MaxSelectorOperands, "maximum selector AST operands")
	fs.Uint64Var(&lim.MaxArchiveEntries, "max-archive-entries", lim.MaxArchiveEntries, "maximum archive entries")
}

func runDecode(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("decode", flag.ContinueOnError)
	input := fs.String("input", "", "frame path or - for stdin")
	output := fs.String("output", "", "plaintext path or - for stdout")
	deps := fs.String("deps", "", "directory of dependency plaintext files named by lowercase hex digest")
	frameSource := fs.String("frame-source", "", "directory of dependency frames named by content hash (enables DAG decode)")
	lim := limits.Default()
	addLimitFlags(fs, &lim)
	if err := fs.Parse(args); err != nil {
		return werrors.NewParse(err.Error())
	}
	if *input == "" || *output == "" {
		return werrors.NewParse("decode requires --input and --output")
	}
	frameBytes, err := readLimited(*input, lim.MaxFrameBytes)
	if err != nil {
		return err
	}
	var plaintext []byte
	if *frameSource != "" {
		plaintext, err = modes.DecodeDAG(ctx, dependency.NewDirectoryFrameSource(*frameSource, lim.MaxFrameBytes), frameBytes, lim)
	} else {
		plaintext, err = modes.DecodeBytes(ctx, frameBytes, resolverFor(*deps), lim)
	}
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
		if h.Mode == frame.ModeFloatTensor {
			output["exponent_coder"] = modes.HuffmanCoderID
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
	if h.Mode == frame.ModeFloatTensor {
		fmt.Printf("exponent_coder: %s\n", modes.HuffmanCoderID)
	}
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
		lim := limits.Default()
		addLimitFlags(fs, &lim)
		if err := fs.Parse(args[1:]); err != nil {
			return werrors.NewParse(err.Error())
		}
		if *output == "" || fs.NArg() < 1 {
			return werrors.NewParse("archive create requires --output and at least one frame")
		}
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
		lim := limits.Default()
		addLimitFlags(fs, &lim)
		if err := fs.Parse(args[1:]); err != nil {
			return werrors.NewParse(err.Error())
		}
		if *input == "" || *outDir == "" {
			return werrors.NewParse("archive decode requires --input and --out-dir")
		}
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
		lim := limits.Default()
		addLimitFlags(fs, &lim)
		if err := fs.Parse(args[1:]); err != nil {
			return werrors.NewParse(err.Error())
		}
		if *input == "" {
			return werrors.NewParse("selector id requires --input")
		}
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
		lim := limits.Default()
		addLimitFlags(fs, &lim)
		if err := fs.Parse(args[1:]); err != nil {
			return werrors.NewParse(err.Error())
		}
		if *input == "" {
			return werrors.NewParse("selector canonicalize requires --input")
		}
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

func runContext(args []string) error {
	if len(args) < 1 {
		return werrors.NewParse("context requires id, uri, or parse")
	}
	switch args[0] {
	case "id":
		return runContextID(args[1:])
	case "uri":
		return runContextURI(args[1:])
	case "parse":
		return runContextParse(args[1:])
	default:
		return werrors.NewParse("context requires id, uri, or parse")
	}
}

type contextOptions struct {
	selector       string
	namespace      string
	snapshot       string
	causal         string
	resolver       string
	capabilityHash string
	budget         string
	asJSON         bool
}

func parseContextFlags(name string, args []string) (contextOptions, error) {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	opts := contextOptions{}
	fs.StringVar(&opts.selector, "selector", "", "selector source file or - for stdin")
	fs.StringVar(&opts.namespace, "namespace", "", "namespace")
	fs.StringVar(&opts.snapshot, "snapshot", "", "pinned snapshot hash")
	fs.StringVar(&opts.causal, "causal", "", "comma-separated causal entries")
	fs.StringVar(&opts.resolver, "resolver", weftcontext.DefaultResolver, "resolver version")
	fs.StringVar(&opts.capabilityHash, "capability-hash", "", "capability scope hash")
	fs.StringVar(&opts.budget, "budget", "", "budget K=V,...")
	fs.BoolVar(&opts.asJSON, "json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return opts, werrors.NewParse(err.Error())
	}
	if opts.selector == "" {
		return opts, werrors.NewParse("context requires --selector")
	}
	if opts.namespace == "" {
		return opts, werrors.NewParse("context requires --namespace")
	}
	if opts.snapshot == "" {
		return opts, werrors.NewParse("context requires --snapshot")
	}
	return opts, nil
}

func (o contextOptions) load() (*weftcontext.ContextRoot, string, error) {
	b, err := readLimited(o.selector, limits.Default().MaxSelectorBytes)
	if err != nil {
		return nil, "", err
	}
	if !utf8.Valid(b) {
		return nil, "", werrors.NewParse("selector input is not valid UTF-8")
	}
	source := string(b)
	causal, err := parseHashList(o.causal, "causal closure")
	if err != nil {
		return nil, "", err
	}
	capability, err := optionalHash(o.capabilityHash, "capability hash")
	if err != nil {
		return nil, "", err
	}
	var budget *weftcontext.Budget
	if o.budget != "" {
		budget, err = weftcontext.ParseBudget(o.budget)
		if err != nil {
			return nil, "", err
		}
	}
	root, err := weftcontext.BuildContextRoot(
		source, o.namespace, o.snapshot, o.resolver, causal, capability, budget)
	if err != nil {
		return nil, "", err
	}
	return root, source, nil
}

func parseHashList(text, what string) ([]hash.Hash32, error) {
	if text == "" {
		return nil, nil
	}
	var hashes []hash.Hash32
	for _, entry := range strings.Split(text, ",") {
		if entry == "" {
			return nil, werrors.NewParse(what + " contains an empty entry")
		}
		h, err := hash.FromHex(entry)
		if err != nil {
			return nil, werrors.NewParse(what + " entry: " + err.Error())
		}
		if h == hash.Zero {
			return nil, werrors.NewParse(what + " entry must not be all zero")
		}
		hashes = append(hashes, h)
	}
	return hashes, nil
}

func optionalHash(text, what string) (*hash.Hash32, error) {
	if text == "" {
		return nil, nil
	}
	h, err := hash.FromHex(text)
	if err != nil {
		return nil, werrors.NewParse(what + ": " + err.Error())
	}
	if h == hash.Zero {
		return nil, werrors.NewParse(what + " must not be all zero")
	}
	return &h, nil
}

func contextJSON(root *weftcontext.ContextRoot) (map[string]any, error) {
	id, err := root.ID()
	if err != nil {
		return nil, err
	}
	uri, err := root.AuthoritativeURI()
	if err != nil {
		return nil, err
	}
	canonical, err := root.CanonicalHex(limits.Default())
	if err != nil {
		return nil, err
	}
	causal := make([]string, 0, len(root.Causal))
	for _, h := range root.Causal {
		causal = append(causal, h.Hex())
	}
	var capability any
	if root.CapabilityHash != nil {
		capability = root.CapabilityHash.Hex()
	}
	return map[string]any{
		"schema":                 weftcontext.ContextRootSchema,
		"namespace":              root.Namespace,
		"selector_id":            root.SelectorID.Hex(),
		"selector_canonical_hex": hex.EncodeToString(root.SelectorBytes),
		"snapshot":               root.Snapshot.Hex(),
		"resolver":               root.Resolver,
		"causal":                 causal,
		"ordering":               root.Ordering,
		"capability_hash":        capability,
		"budget":                 root.Budget.JSON(),
		"context_root_id":        id.Hex(),
		"uri":                    uri,
		"canonical_hex":          canonical,
	}, nil
}

func printJSON(value any) error {
	encoded, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return werrors.NewIo("cannot serialize JSON: " + err.Error())
	}
	fmt.Println(string(encoded))
	return nil
}

func runContextID(args []string) error {
	opts, err := parseContextFlags("context id", args)
	if err != nil {
		return err
	}
	root, _, err := opts.load()
	if err != nil {
		return err
	}
	if opts.asJSON {
		output, err := contextJSON(root)
		if err != nil {
			return err
		}
		return printJSON(output)
	}
	id, err := root.ID()
	if err != nil {
		return err
	}
	fmt.Println(id.Hex())
	return nil
}

func runContextURI(args []string) error {
	descriptor := false
	filtered := make([]string, 0, len(args))
	for _, arg := range args {
		if arg == "--descriptor" {
			descriptor = true
			continue
		}
		filtered = append(filtered, arg)
	}
	opts, err := parseContextFlags("context uri", filtered)
	if err != nil {
		return err
	}
	root, source, err := opts.load()
	if err != nil {
		return err
	}
	if descriptor {
		uri, err := root.DescriptorURI(source)
		if err != nil {
			return err
		}
		if opts.asJSON {
			canonical, err := root.AuthoritativeURI()
			if err != nil {
				return err
			}
			id, err := root.ID()
			if err != nil {
				return err
			}
			return printJSON(map[string]any{
				"uri":             uri,
				"context_root_id": id.Hex(),
				"descriptor":      uri,
				"canonical_uri":   canonical,
			})
		}
		fmt.Println(uri)
		return nil
	}
	uri, err := root.AuthoritativeURI()
	if err != nil {
		return err
	}
	if opts.asJSON {
		id, err := root.ID()
		if err != nil {
			return err
		}
		return printJSON(map[string]any{
			"uri":             uri,
			"context_root_id": id.Hex(),
		})
	}
	fmt.Println(uri)
	return nil
}

func runContextParse(args []string) error {
	asJSON := false
	positional := make([]string, 0, len(args))
	for _, arg := range args {
		switch arg {
		case "--json":
			asJSON = true
		case "-h", "--help":
			return werrors.NewParse("usage: weftcodec context parse weft://... [--json]")
		default:
			positional = append(positional, arg)
		}
	}
	if len(positional) != 1 {
		return werrors.NewParse("context parse requires exactly one URI")
	}
	lim := limits.Default()
	parsed, err := weftcontext.ParseContextURI(positional[0], lim)
	if err != nil {
		return err
	}
	canonical, err := parsed.CanonicalURI(lim)
	if err != nil {
		return err
	}
	if parsed.Form == weftcontext.FormAuthoritative {
		if asJSON {
			return printJSON(map[string]any{
				"form":            "authoritative",
				"namespace":       parsed.Namespace,
				"context_root_id": parsed.ID.Hex(),
				"uri":             canonical,
			})
		}
		fmt.Println(canonical)
		return nil
	}
	root, err := parsed.ToContextRoot(lim)
	if err != nil {
		return err
	}
	if asJSON {
		output, err := contextJSON(root)
		if err != nil {
			return err
		}
		output["form"] = "descriptor"
		output["namespace"] = parsed.Namespace
		output["selector_source"] = parsed.SelectorSource
		output["descriptor"] = positional[0]
		output["uri"] = canonical
		return printJSON(output)
	}
	fmt.Println(canonical)
	return nil
}

func runCausal(args []string) error {
	if len(args) < 1 || args[0] != "id" {
		return werrors.NewParse("causal requires id")
	}
	fs := flag.NewFlagSet("causal id", flag.ContinueOnError)
	parentsText := fs.String("parents", "", "comma-separated explicit parents")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args[1:]); err != nil {
		return werrors.NewParse(err.Error())
	}
	if *parentsText == "" {
		return werrors.NewParse("causal id requires --parents")
	}
	parents, err := parseHashList(*parentsText, "causal parents")
	if err != nil {
		return err
	}
	closure, err := weftcontext.FromParents(parents)
	if err != nil {
		return err
	}
	id, err := closure.ID()
	if err != nil {
		return err
	}
	if *asJSON {
		parentsHex := make([]string, 0, len(closure.Parents))
		for _, h := range closure.Parents {
			parentsHex = append(parentsHex, h.Hex())
		}
		closureHex := make([]string, 0, len(closure.Closure))
		for _, h := range closure.Closure {
			closureHex = append(closureHex, h.Hex())
		}
		canonical, err := closure.CanonicalBytes(limits.Default())
		if err != nil {
			return err
		}
		return printJSON(map[string]any{
			"schema":        weftcontext.CausalSchema,
			"parents":       parentsHex,
			"closure":       closureHex,
			"closure_hash":  id.Hex(),
			"canonical_hex": hex.EncodeToString(canonical),
		})
	}
	fmt.Println(id.Hex())
	return nil
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
