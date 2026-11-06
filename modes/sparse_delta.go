// Package modes implements mode 16 SPARSE_DELTA (WEFT-AI-CODEC §17).
//
// Mode 16 reconstructs a tensor from the same exact word-wise difference mode
// 11 defines, but stores the delta word array as a zero-run stream instead of
// two to four Huffman planes per element:
//
//	payload  = u32le(run_count) || run_count records
//	record   = varint(zero_run_len) || storage_bytes(dtype) bytes of the word
//
// It is the sparse counterpart of mode 11. Trailing zeros are implicit, so an
// all-zero delta is run_count = 0 and reconstructs the base exactly.
package modes

import (
	"context"
	"encoding/binary"
	"slices"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/varint"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// SparseDeltaSchema is the canonical descriptor schema.
const SparseDeltaSchema = "weft.ai.sparse_delta.v1"

// SparseDeltaOp is the only delta operation implemented in v1.
const SparseDeltaOp = "mod-sub"

// SparseDeltaCodec is the only sparse coder implemented in v1.
const SparseDeltaCodec = "zero-run-v1"

// SparseDeltaMeta is a parsed mode-16 descriptor. It has exactly the nine keys
// fixed by WEFT-AI-CODEC §17.2; run_count lives in the payload, not here.
type SparseDeltaMeta struct {
	Dtype        WeightDtype
	Shape        []uint64
	ElementCount uint64
	StorageBytes uint64
	Op           string
	SparseCodec  string
	Base         hash.Hash32
	DeltaRoot    hash.Hash32
}

func sparseDeltaShape(shape []uint64, lim limits.Limits) (uint64, error) {
	if len(shape) == 0 {
		return 0, werrors.NewMalformed("SPARSE_DELTA shape rank must be at least 1")
	}
	if uint64(len(shape)) > lim.MaxTensorRank {
		return 0, werrors.LimitExceededf(
			"SPARSE_DELTA rank %d exceeds limit %d", len(shape), lim.MaxTensorRank)
	}
	product := uint64(1)
	for _, dim := range shape {
		if dim == 0 {
			return 0, werrors.NewMalformed(
				"SPARSE_DELTA shape dimensions must be greater than zero")
		}
		next, err := limits.MulChecked(product, dim, "SPARSE_DELTA shape product")
		if err != nil {
			return 0, err
		}
		product = next
	}
	if product > lim.MaxTensorElements {
		return 0, werrors.LimitExceededf(
			"SPARSE_DELTA element count %d exceeds limit %d", product, lim.MaxTensorElements)
	}
	return product, nil
}

func sparseDeltaPayloadLen(elementCount uint64, dtype WeightDtype) (uint64, error) {
	return limits.MulChecked(elementCount, dtype.StorageBytes, "SPARSE_DELTA element bytes")
}

// ValidateSparseDeltaMeta validates every descriptor invariant and returns the
// element count.
func ValidateSparseDeltaMeta(meta SparseDeltaMeta, lim limits.Limits) (uint64, error) {
	if meta.Op != SparseDeltaOp {
		return 0, werrors.Malformedf(
			"SPARSE_DELTA op %q is not implemented (only %s)", meta.Op, SparseDeltaOp)
	}
	if meta.SparseCodec != SparseDeltaCodec {
		return 0, werrors.Malformedf(
			"SPARSE_DELTA sparse_codec %q is not implemented (only %s)",
			meta.SparseCodec, SparseDeltaCodec)
	}
	if meta.StorageBytes != meta.Dtype.StorageBytes {
		return 0, werrors.Malformedf(
			"SPARSE_DELTA storage_bytes %d disagrees with dtype %s (%d bytes)",
			meta.StorageBytes, meta.Dtype.Name, meta.Dtype.StorageBytes)
	}
	count, err := sparseDeltaShape(meta.Shape, lim)
	if err != nil {
		return 0, err
	}
	if meta.ElementCount != count {
		return 0, werrors.Malformedf(
			"SPARSE_DELTA element_count %d does not match shape product %d",
			meta.ElementCount, count)
	}
	if _, err := sparseDeltaPayloadLen(count, meta.Dtype); err != nil {
		return 0, err
	}
	return count, nil
}

func sparseDeltaDescriptorValue(meta SparseDeltaMeta) wcb.Value {
	shape := make([]wcb.Value, 0, len(meta.Shape))
	for _, dim := range meta.Shape {
		shape = append(shape, wcb.U64Value(dim))
	}
	return wcb.MapValue(map[string]wcb.Value{
		"schema":            wcb.TextValue(SparseDeltaSchema),
		"dtype":             wcb.TextValue(meta.Dtype.Name),
		"shape":             wcb.ArrayValue(shape),
		"element_count":     wcb.U64Value(meta.ElementCount),
		"storage_bytes":     wcb.U64Value(meta.StorageBytes),
		"op":                wcb.TextValue(meta.Op),
		"sparse_codec":      wcb.TextValue(meta.SparseCodec),
		"base_content_root": wcb.BytesValue(meta.Base.Bytes()),
		"delta_root":        wcb.BytesValue(meta.DeltaRoot.Bytes()),
	})
}

func parseSparseDeltaDescriptor(bytes []byte, lim limits.Limits) (SparseDeltaMeta, error) {
	value, err := wcb.Decode(bytes, lim)
	if err != nil {
		return SparseDeltaMeta{}, err
	}
	if value.Kind != wcb.Map {
		return SparseDeltaMeta{}, werrors.NewMalformed(
			"SPARSE_DELTA metadata must be a WCB map")
	}
	m := value.Map
	keys := []string{
		"schema", "dtype", "shape", "element_count", "storage_bytes",
		"op", "sparse_codec", "base_content_root", "delta_root",
	}
	for key := range m {
		if !containsString(keys, key) {
			return SparseDeltaMeta{}, werrors.Malformedf(
				"SPARSE_DELTA metadata has unknown key %q", key)
		}
	}
	for _, key := range keys {
		if _, ok := m[key]; !ok {
			return SparseDeltaMeta{}, werrors.Malformedf(
				"SPARSE_DELTA metadata is missing key %q", key)
		}
	}
	schema, err := weightDeltaMetaText(m, "schema")
	if err != nil {
		return SparseDeltaMeta{}, err
	}
	if schema != SparseDeltaSchema {
		return SparseDeltaMeta{}, werrors.Malformedf(
			"unsupported SPARSE_DELTA schema %q", schema)
	}
	dtypeText, err := weightDeltaMetaText(m, "dtype")
	if err != nil {
		return SparseDeltaMeta{}, err
	}
	dtype, err := ParseWeightDtype(dtypeText)
	if err != nil {
		return SparseDeltaMeta{}, err
	}
	shape, err := weightDeltaMetaShape(m, "shape")
	if err != nil {
		return SparseDeltaMeta{}, err
	}
	elementCount, err := weightDeltaMetaU64(m, "element_count")
	if err != nil {
		return SparseDeltaMeta{}, err
	}
	storageBytes, err := weightDeltaMetaU64(m, "storage_bytes")
	if err != nil {
		return SparseDeltaMeta{}, err
	}
	op, err := weightDeltaMetaText(m, "op")
	if err != nil {
		return SparseDeltaMeta{}, err
	}
	sparseCodec, err := weightDeltaMetaText(m, "sparse_codec")
	if err != nil {
		return SparseDeltaMeta{}, err
	}
	base, err := weightDeltaMetaHash(m, "base_content_root")
	if err != nil {
		return SparseDeltaMeta{}, err
	}
	deltaRoot, err := weightDeltaMetaHash(m, "delta_root")
	if err != nil {
		return SparseDeltaMeta{}, err
	}
	meta := SparseDeltaMeta{
		Dtype:        dtype,
		Shape:        shape,
		ElementCount: elementCount,
		StorageBytes: storageBytes,
		Op:           op,
		SparseCodec:  sparseCodec,
		Base:         base,
		DeltaRoot:    deltaRoot,
	}
	if _, err := ValidateSparseDeltaMeta(meta, lim); err != nil {
		return SparseDeltaMeta{}, err
	}
	return meta, nil
}

// SparseDeltaMetadata parses and validates a mode-16 descriptor without
// resolving the base.
func SparseDeltaMetadata(f *frame.Frame, lim limits.Limits) (SparseDeltaMeta, error) {
	if f.Header.Mode != frame.ModeSparseDelta {
		return SparseDeltaMeta{}, werrors.Malformedf(
			"expected a SPARSE_DELTA frame, got mode %s", f.Header.Mode)
	}
	descriptor, _, err := splitAIBodyPrefix(f.Body)
	if err != nil {
		return SparseDeltaMeta{}, err
	}
	return parseSparseDeltaDescriptor(descriptor, lim)
}

// ---------------------------------------------------------------------------
// Zero-run stream
// ---------------------------------------------------------------------------

// buildSparseRuns encodes the delta words as a zero-run-v1 stream. Records are
// emitted in element order and trailing zeros are never stored, so the payload
// is canonical and identical in both implementations.
func buildSparseRuns(
	dtype WeightDtype,
	target, base []byte,
	elementCount int,
	lim limits.Limits,
) ([]byte, []byte, error) {
	storage := int(dtype.StorageBytes)
	deltaBytes := make([]byte, 0, elementCount*storage)
	mask := weightMask(dtype)
	runCount := uint64(0)
	for index := 0; index < elementCount; index++ {
		delta := (weightWordAt(dtype, target, index) - weightWordAt(dtype, base, index)) & mask
		deltaBytes = pushWeightWord(dtype, delta, deltaBytes)
		if delta != 0 {
			runCount++
		}
	}
	if err := limits.CheckLen(runCount, lim.MaxSparseRuns, "SPARSE_DELTA non-zero run count"); err != nil {
		return nil, nil, err
	}
	if runCount > uint64(^uint32(0)) {
		return nil, nil, werrors.Newf(werrors.LimitExceeded,
			"SPARSE_DELTA run_count does not fit u32")
	}
	payload := make([]byte, 4, 4+int(runCount)*(storage+1))
	binary.LittleEndian.PutUint32(payload, uint32(runCount))
	zeroRun := uint64(0)
	for index := 0; index < elementCount; index++ {
		start := index * storage
		word := deltaBytes[start : start+storage]
		allZero := true
		for _, b := range word {
			if b != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			zeroRun++
			continue
		}
		payload = varint.Append(payload, zeroRun)
		payload = append(payload, word...)
		zeroRun = 0
	}
	return deltaBytes, payload, nil
}

// decodeSparseRuns rebuilds the canonical delta word bytes from a zero-run-v1
// payload, failing closed on every inconsistency.
func decodeSparseRuns(
	dtype WeightDtype,
	payload []byte,
	elementCount int,
	lim limits.Limits,
) ([]byte, error) {
	if len(payload) < 4 {
		return nil, werrors.NewMalformed(
			"SPARSE_DELTA body is shorter than its 4-byte run_count")
	}
	runCount := uint64(binary.LittleEndian.Uint32(payload[:4]))
	if err := limits.CheckLen(runCount, lim.MaxSparseRuns, "SPARSE_DELTA run_count"); err != nil {
		return nil, err
	}
	if runCount > uint64(elementCount) {
		return nil, werrors.Malformedf(
			"SPARSE_DELTA run_count %d exceeds element_count %d", runCount, elementCount)
	}
	storage := int(dtype.StorageBytes)
	deltaLen, err := limits.MulChecked(uint64(elementCount), dtype.StorageBytes,
		"SPARSE_DELTA delta bytes")
	if err != nil {
		return nil, err
	}
	if err := limits.CheckLen(deltaLen, lim.MaxDecodedBytes, "SPARSE_DELTA delta bytes"); err != nil {
		return nil, err
	}
	deltaBytes := make([]byte, int(deltaLen))
	pos := 4
	emitted := uint64(0)
	for record := uint64(0); record < runCount; record++ {
		zeroRun, next, err := varint.DecodeAt(payload, pos)
		if err != nil {
			return nil, err
		}
		pos = next
		if zeroRun > ^uint64(0)-emitted {
			return nil, werrors.Newf(werrors.LimitExceeded,
				"SPARSE_DELTA zero-run length overflows u64")
		}
		emitted += zeroRun
		if emitted >= uint64(elementCount) {
			return nil, werrors.Malformedf(
				"SPARSE_DELTA run %d skips %d words past element_count %d",
				record, zeroRun, elementCount)
		}
		end := pos + storage
		if end > len(payload) {
			return nil, werrors.Malformedf(
				"SPARSE_DELTA run %d record is truncated", record)
		}
		word := payload[pos:end]
		allZero := true
		for _, b := range word {
			if b != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			return nil, werrors.Corruptf(
				"SPARSE_DELTA run %d stores a zero word (non-canonical: zeros are implicit)",
				record)
		}
		if dtype.Name == "q4_0" && word[0] > 15 {
			return nil, werrors.Corruptf(
				"SPARSE_DELTA q4_0 run %d word is %d, outside 0..=15", record, word[0])
		}
		at := int(emitted) * storage
		copy(deltaBytes[at:at+storage], word)
		pos = end
		emitted++
	}
	if pos != len(payload) {
		return nil, werrors.Corruptf(
			"SPARSE_DELTA payload has %d trailing bytes after the %d declared runs",
			len(payload)-pos, runCount)
	}
	return deltaBytes, nil
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

// EncodeSparseDeltaDescriptorFrame builds a frame from an explicit descriptor
// and stored payload. It is the low-level constructor used by the encoder and
// by conformance-vector generation.
func EncodeSparseDeltaDescriptorFrame(
	meta SparseDeltaMeta,
	payload []byte,
	decodedLen uint64,
	contentHash hash.Hash32,
	baseMode uint32,
	lim limits.Limits,
) ([]byte, error) {
	body, err := buildAIBody(sparseDeltaDescriptorValue(meta), payload, lim)
	if err != nil {
		return nil, err
	}
	parts := frame.Parts{
		FormatMajor: frame.FormatMajor,
		FormatMinor: frame.FormatMinor,
		Domain:      frame.DomainPayload,
		Mode:        frame.ModeSparseDelta,
		Flags:       0,
		Level:       0,
		DecodedLen:  decodedLen,
		ContentHash: contentHash,
		Dependencies: []frame.Dependency{
			frame.NewDependency(meta.Base, decodedLen, frame.DepDomainPayload, baseMode),
		},
		Chunks: nil,
		Body:   body,
	}
	return frame.Encode(parts, lim)
}

// EncodeSparseDelta encodes target as a mode-16 delta against baseFrame.
func EncodeSparseDelta(
	ctx context.Context,
	target []byte,
	dtype WeightDtype,
	shape []uint64,
	baseFrame []byte,
	lim limits.Limits,
) ([]byte, error) {
	return EncodeSparseDeltaWithResolver(
		ctx, target, dtype, shape, baseFrame, dependency.NewMapResolver(), lim)
}

// EncodeSparseDeltaWithResolver encodes a target against a base frame whose own
// dependencies are resolved through resolver.
func EncodeSparseDeltaWithResolver(
	ctx context.Context,
	target []byte,
	dtype WeightDtype,
	shape []uint64,
	baseFrame []byte,
	resolver dependency.Resolver,
	lim limits.Limits,
) ([]byte, error) {
	baseParsed, err := frame.Parse(baseFrame, lim)
	if err != nil {
		return nil, err
	}
	baseDtype, baseShape, err := BaseFrameDescriptor(baseParsed, lim)
	if err != nil {
		return nil, err
	}
	if baseDtype.Name != dtype.Name {
		return nil, werrors.Malformedf(
			"base frame dtype %q does not match SPARSE_DELTA dtype %q",
			baseDtype.Name, dtype.Name)
	}
	if !slices.Equal(baseShape, shape) {
		return nil, werrors.NewMalformed(
			"base frame shape does not match the SPARSE_DELTA shape")
	}
	elementCount, err := sparseDeltaShape(shape, lim)
	if err != nil {
		return nil, err
	}
	expected, err := sparseDeltaPayloadLen(elementCount, dtype)
	if err != nil {
		return nil, err
	}
	if err := limits.CheckLen(expected, lim.MaxDecodedBytes, "SPARSE_DELTA element bytes"); err != nil {
		return nil, err
	}
	if uint64(len(target)) != expected {
		return nil, werrors.Malformedf(
			"SPARSE_DELTA target is %d bytes but shape %s x %d requires %d",
			len(target), dtype.Name, elementCount, expected)
	}
	baseBytes, err := DecodeBytes(ctx, baseFrame, resolver, lim)
	if err != nil {
		return nil, err
	}
	if uint64(len(baseBytes)) != expected {
		return nil, werrors.DependencyMismatchf(
			"base frame decodes to %d bytes but the descriptor requires %d",
			len(baseBytes), expected)
	}
	if dtype.IsQuantized() {
		if err := checkQuantizedWords(dtype, target, "target"); err != nil {
			return nil, err
		}
		if err := checkQuantizedWords(dtype, baseBytes, "base"); err != nil {
			return nil, err
		}
	}
	count, err := limits.ToInt(elementCount, "SPARSE_DELTA element count")
	if err != nil {
		return nil, err
	}
	deltaBytes, payload, err := buildSparseRuns(dtype, target, baseBytes, count, lim)
	if err != nil {
		return nil, err
	}
	meta := SparseDeltaMeta{
		Dtype:        dtype,
		Shape:        append([]uint64(nil), shape...),
		ElementCount: elementCount,
		StorageBytes: dtype.StorageBytes,
		Op:           SparseDeltaOp,
		SparseCodec:  SparseDeltaCodec,
		Base:         baseParsed.Header.ContentHash,
		DeltaRoot:    hash.Domain(tensor.DomainTensorValues, deltaBytes),
	}
	return EncodeSparseDeltaDescriptorFrame(
		meta, payload, uint64(len(target)), hash.Domain(hash.DomainChunk, target),
		uint32(baseParsed.Header.Mode), lim)
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

// DecodeSparseDeltaFrame decodes a mode-16 frame, resolving the base through
// resolver. A resolver that can also return the base frame bytes additionally
// gets its dtype/shape descriptor verified.
func DecodeSparseDeltaFrame(
	ctx context.Context,
	f *frame.Frame,
	resolver dependency.Resolver,
	lim limits.Limits,
) ([]byte, error) {
	return decodeSparseDelta(ctx, f, nil, resolver, lim)
}

// DecodeSparseDeltaFrameWithBase decodes a mode-16 frame with the base frame
// supplied out of band.
func DecodeSparseDeltaFrameWithBase(
	ctx context.Context,
	f *frame.Frame,
	baseFrame []byte,
	resolver dependency.Resolver,
	lim limits.Limits,
) ([]byte, error) {
	return decodeSparseDelta(ctx, f, baseFrame, resolver, lim)
}

func decodeSparseDelta(
	ctx context.Context,
	f *frame.Frame,
	baseFrame []byte,
	resolver dependency.Resolver,
	lim limits.Limits,
) ([]byte, error) {
	if f.Header.Mode != frame.ModeSparseDelta {
		return nil, werrors.Malformedf(
			"expected a SPARSE_DELTA frame, got mode %s", f.Header.Mode)
	}
	descriptor, payload, err := splitAIBodyPrefix(f.Body)
	if err != nil {
		return nil, err
	}
	meta, err := parseSparseDeltaDescriptor(descriptor, lim)
	if err != nil {
		return nil, err
	}
	elementCount, err := ValidateSparseDeltaMeta(meta, lim)
	if err != nil {
		return nil, err
	}
	expected, err := sparseDeltaPayloadLen(elementCount, meta.Dtype)
	if err != nil {
		return nil, err
	}
	if f.Header.DecodedLen != expected {
		return nil, werrors.Corruptf(
			"SPARSE_DELTA decoded_len is %d but the descriptor requires %d",
			f.Header.DecodedLen, expected)
	}
	if err := limits.CheckLen(expected, lim.MaxDecodedBytes, "SPARSE_DELTA element bytes"); err != nil {
		return nil, err
	}
	if f.Header.DepCount != 1 {
		return nil, werrors.NewMalformed(
			"SPARSE_DELTA requires exactly one dependency")
	}
	dep := f.Dependencies[0]
	if dep.Hash != meta.Base {
		return nil, werrors.NewMalformed(
			"SPARSE_DELTA dependency hash must equal the descriptor base_content_root")
	}
	if dep.DecodedLen != f.Header.DecodedLen {
		return nil, werrors.NewMalformed(
			"SPARSE_DELTA dependency decoded_len must equal the frame decoded_len")
	}
	count, err := limits.ToInt(elementCount, "SPARSE_DELTA element count")
	if err != nil {
		return nil, err
	}
	deltaBytes, err := decodeSparseRuns(meta.Dtype, payload, count, lim)
	if err != nil {
		return nil, err
	}
	if hash.Domain(tensor.DomainTensorValues, deltaBytes) != meta.DeltaRoot {
		return nil, werrors.NewCorrupt("SPARSE_DELTA delta_root mismatch")
	}

	var baseBytes []byte
	var suppliedFrame *frame.Frame
	if baseFrame != nil {
		baseBytes, err = DecodeBytes(ctx, baseFrame, resolver, lim)
		if err != nil {
			return nil, err
		}
		suppliedFrame, err = frame.Parse(baseFrame, lim)
		if err != nil {
			return nil, err
		}
	} else {
		var fetched []byte
		var ok bool
		if provider, providerOK := resolver.(weightFrameProvider); providerOK {
			fetched, ok, err = provider.ResolveFrame(ctx, meta.Base)
			if err != nil {
				return nil, err
			}
			if !ok {
				fetched = nil
			}
		}
		baseBytes, err = resolver.Resolve(ctx, meta.Base, f.Header.DecodedLen, frame.DepDomainPayload)
		if err != nil {
			return nil, err
		}
		if fetched != nil {
			suppliedFrame, err = frame.Parse(fetched, lim)
			if err != nil {
				return nil, err
			}
		}
	}
	if suppliedFrame != nil {
		if suppliedFrame.Header.ContentHash != meta.Base {
			return nil, werrors.NewDependencyMismatch(
				"supplied base frame content hash does not equal the descriptor base")
		}
		baseDtype, baseShape, err := BaseFrameDescriptor(suppliedFrame, lim)
		if err != nil {
			return nil, werrors.DependencyMismatchf("%s", err.Error())
		}
		if baseDtype.Name != meta.Dtype.Name {
			return nil, werrors.DependencyMismatchf(
				"base frame dtype %q does not match SPARSE_DELTA dtype %q",
				baseDtype.Name, meta.Dtype.Name)
		}
		if !slices.Equal(baseShape, meta.Shape) {
			return nil, werrors.NewDependencyMismatch(
				"base frame shape does not match the SPARSE_DELTA shape")
		}
	}
	if uint64(len(baseBytes)) != expected {
		return nil, werrors.DependencyMismatchf(
			"base frame decodes to %d bytes but the descriptor requires %d",
			len(baseBytes), expected)
	}
	if meta.Dtype.IsQuantized() {
		if err := checkQuantizedWords(meta.Dtype, baseBytes, "base"); err != nil {
			return nil, err
		}
	}
	mask := weightMask(meta.Dtype)
	target := make([]byte, 0, int(expected))
	for index := 0; index < count; index++ {
		delta := weightWordAt(meta.Dtype, deltaBytes, index)
		word := (delta + weightWordAt(meta.Dtype, baseBytes, index)) & mask
		target = pushWeightWord(meta.Dtype, word, target)
	}
	if uint64(len(target)) != expected {
		return nil, werrors.NewCorrupt(
			"SPARSE_DELTA reconstructed element length is inconsistent")
	}
	if hash.Domain(hash.DomainChunk, target) != f.Header.ContentHash {
		return nil, werrors.NewCorrupt(
			"SPARSE_DELTA content_hash mismatch against the reconstructed tensor")
	}
	return target, nil
}
