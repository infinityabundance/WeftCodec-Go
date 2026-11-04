package modes

// Mode 14 SEMANTIC_DELTA (WEFT-AI-CODEC §15).
//
// The payload is exactly the WEIGHT_DELTA scheme (modular element-wise
// subtraction on raw words, one huffman-canonical-v1 stream per byte plane).
// The descriptor additionally binds the semantic root under which the base was
// selected. Decoding recomputes that root from the resolved base bytes; a
// mismatch fails closed. No lossy reconstruction exists.

import (
	"context"
	"slices"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// SemanticDeltaSchema is the canonical mode-14 descriptor schema.
const SemanticDeltaSchema = "weft.ai.semantic_delta.v1"

// SemanticDeltaMeta is a parsed mode-14 descriptor.
type SemanticDeltaMeta struct {
	Dtype            WeightDtype
	Shape            []uint64
	ElementCount     uint64
	WordBits         uint64
	Base             hash.Hash32
	BaseDtype        WeightDtype
	BaseShape        []uint64
	BaseSemanticRoot hash.Hash32
	Policy           SemanticPolicy
	DeltaRoot        hash.Hash32
	PlaneCodec       string
	PayloadLen       uint64
}

// ValidateSemanticDeltaMeta validates every descriptor invariant and returns
// the element count.
func ValidateSemanticDeltaMeta(meta SemanticDeltaMeta, lim limits.Limits) (uint64, error) {
	if meta.PlaneCodec != WeightDeltaPlaneCodec {
		return 0, werrors.Malformedf(
			"SEMANTIC_DELTA plane_codec %q is not implemented (only %s)",
			meta.PlaneCodec, WeightDeltaPlaneCodec)
	}
	if meta.Dtype.WordBits != meta.WordBits {
		return 0, werrors.Malformedf(
			"SEMANTIC_DELTA word_bits %d disagrees with dtype %s (%d bits)",
			meta.WordBits, meta.Dtype.Name, meta.Dtype.WordBits)
	}
	if meta.BaseDtype != meta.Dtype {
		return 0, werrors.Malformedf(
			"SEMANTIC_DELTA base_dtype %q must equal dtype %q",
			meta.BaseDtype.Name, meta.Dtype.Name)
	}
	if !slices.Equal(meta.BaseShape, meta.Shape) {
		return 0, werrors.NewMalformed("SEMANTIC_DELTA base_shape must equal shape")
	}
	count, err := weightDeltaShape(meta.Shape, lim)
	if err != nil {
		return 0, err
	}
	if meta.ElementCount != count {
		return 0, werrors.Malformedf(
			"SEMANTIC_DELTA element_count %d does not match shape product %d",
			meta.ElementCount, count)
	}
	if _, err := weightDeltaPayloadLen(count, meta.Dtype); err != nil {
		return 0, err
	}
	if err := meta.Policy.Validate(meta.Dtype, count); err != nil {
		return 0, err
	}
	return count, nil
}

func semanticDeltaDescriptorValue(meta SemanticDeltaMeta) wcb.Value {
	shape := make([]wcb.Value, 0, len(meta.Shape))
	for _, dim := range meta.Shape {
		shape = append(shape, wcb.U64Value(dim))
	}
	baseShape := make([]wcb.Value, 0, len(meta.BaseShape))
	for _, dim := range meta.BaseShape {
		baseShape = append(baseShape, wcb.U64Value(dim))
	}
	return wcb.MapValue(map[string]wcb.Value{
		"schema":             wcb.TextValue(SemanticDeltaSchema),
		"dtype":              wcb.TextValue(meta.Dtype.Name),
		"shape":              wcb.ArrayValue(shape),
		"element_count":      wcb.U64Value(meta.ElementCount),
		"word_bits":          wcb.U64Value(meta.WordBits),
		"base":               wcb.BytesValue(meta.Base.Bytes()),
		"base_dtype":         wcb.TextValue(meta.BaseDtype.Name),
		"base_shape":         wcb.ArrayValue(baseShape),
		"base_semantic_root": wcb.BytesValue(meta.BaseSemanticRoot.Bytes()),
		"policy":             meta.Policy.WCBValue(),
		"delta_root":         wcb.BytesValue(meta.DeltaRoot.Bytes()),
		"plane_codec":        wcb.TextValue(meta.PlaneCodec),
		"payload_len":        wcb.U64Value(meta.PayloadLen),
	})
}

func parseSemanticDeltaDescriptor(data []byte, lim limits.Limits) (SemanticDeltaMeta, error) {
	value, err := wcb.Decode(data, lim)
	if err != nil {
		return SemanticDeltaMeta{}, err
	}
	if value.Kind != wcb.Map {
		return SemanticDeltaMeta{}, werrors.NewMalformed(
			"SEMANTIC_DELTA metadata must be a WCB map")
	}
	allowed := map[string]bool{
		"schema": true, "dtype": true, "shape": true, "element_count": true,
		"word_bits": true, "base": true, "base_dtype": true, "base_shape": true,
		"base_semantic_root": true, "policy": true, "delta_root": true,
		"plane_codec": true, "payload_len": true,
	}
	for key := range value.Map {
		if !allowed[key] {
			return SemanticDeltaMeta{}, werrors.Malformedf(
				"SEMANTIC_DELTA metadata has unknown key %q", key)
		}
	}
	for key := range allowed {
		if _, ok := value.Map[key]; !ok {
			return SemanticDeltaMeta{}, werrors.Malformedf(
				"SEMANTIC_DELTA metadata is missing key %q", key)
		}
	}
	schemaValue := value.Map["schema"]
	if schemaValue.Kind != wcb.Text || schemaValue.Str != SemanticDeltaSchema {
		return SemanticDeltaMeta{}, werrors.Malformedf(
			"unsupported SEMANTIC_DELTA schema %q", schemaValue.Str)
	}
	dtypeText, err := weightDeltaMetaText(value.Map, "dtype")
	if err != nil {
		return SemanticDeltaMeta{}, err
	}
	dtype, err := ParseWeightDtype(dtypeText)
	if err != nil {
		return SemanticDeltaMeta{}, err
	}
	shape, err := weightDeltaMetaShape(value.Map, "shape")
	if err != nil {
		return SemanticDeltaMeta{}, err
	}
	elementCount, err := weightDeltaMetaU64(value.Map, "element_count")
	if err != nil {
		return SemanticDeltaMeta{}, err
	}
	wordBits, err := weightDeltaMetaU64(value.Map, "word_bits")
	if err != nil {
		return SemanticDeltaMeta{}, err
	}
	base, err := weightDeltaMetaHash(value.Map, "base")
	if err != nil {
		return SemanticDeltaMeta{}, err
	}
	baseDtypeText, err := weightDeltaMetaText(value.Map, "base_dtype")
	if err != nil {
		return SemanticDeltaMeta{}, err
	}
	baseDtype, err := ParseWeightDtype(baseDtypeText)
	if err != nil {
		return SemanticDeltaMeta{}, err
	}
	baseShape, err := weightDeltaMetaShape(value.Map, "base_shape")
	if err != nil {
		return SemanticDeltaMeta{}, err
	}
	baseRoot, err := weightDeltaMetaHash(value.Map, "base_semantic_root")
	if err != nil {
		return SemanticDeltaMeta{}, err
	}
	policy, err := ParseSemanticPolicy(value.Map["policy"], dtype, elementCount)
	if err != nil {
		return SemanticDeltaMeta{}, err
	}
	deltaRoot, err := weightDeltaMetaHash(value.Map, "delta_root")
	if err != nil {
		return SemanticDeltaMeta{}, err
	}
	planeCodec, err := weightDeltaMetaText(value.Map, "plane_codec")
	if err != nil {
		return SemanticDeltaMeta{}, err
	}
	payloadLen, err := weightDeltaMetaU64(value.Map, "payload_len")
	if err != nil {
		return SemanticDeltaMeta{}, err
	}
	meta := SemanticDeltaMeta{
		Dtype:            dtype,
		Shape:            shape,
		ElementCount:     elementCount,
		WordBits:         wordBits,
		Base:             base,
		BaseDtype:        baseDtype,
		BaseShape:        baseShape,
		BaseSemanticRoot: baseRoot,
		Policy:           policy,
		DeltaRoot:        deltaRoot,
		PlaneCodec:       planeCodec,
		PayloadLen:       payloadLen,
	}
	if _, err := ValidateSemanticDeltaMeta(meta, lim); err != nil {
		return SemanticDeltaMeta{}, err
	}
	return meta, nil
}

// SemanticDeltaMetadata parses a mode-14 descriptor (no base resolution).
func SemanticDeltaMetadata(f *frame.Frame, lim limits.Limits) (SemanticDeltaMeta, error) {
	if f.Header.Mode != frame.ModeSemanticDelta {
		return SemanticDeltaMeta{}, werrors.Malformedf(
			"expected a SEMANTIC_DELTA frame, got mode %s", f.Header.Mode)
	}
	descriptor, _, err := splitAIBodyPrefix(f.Body)
	if err != nil {
		return SemanticDeltaMeta{}, err
	}
	return parseSemanticDeltaDescriptor(descriptor, lim)
}

// EncodeSemanticDeltaDescriptorFrame builds a mode-14 frame from an explicit
// descriptor and payload. It does not re-check descriptor semantics; the frame
// is re-parsed, so frame-level mode rules still apply.
func EncodeSemanticDeltaDescriptorFrame(
	meta SemanticDeltaMeta,
	payload []byte,
	decodedLen uint64,
	contentHash hash.Hash32,
	baseMode uint32,
	lim limits.Limits,
) ([]byte, error) {
	metaBytes, err := wcb.Encode(semanticDeltaDescriptorValue(meta), lim)
	if err != nil {
		return nil, err
	}
	if uint64(len(metaBytes)) > uint64(^uint32(0)) {
		return nil, werrors.NewLimitExceeded("SEMANTIC_DELTA metadata does not fit u32")
	}
	body := make([]byte, 0, 4+len(metaBytes)+len(payload))
	var prefix [4]byte
	prefix[0] = byte(len(metaBytes))
	prefix[1] = byte(len(metaBytes) >> 8)
	prefix[2] = byte(len(metaBytes) >> 16)
	prefix[3] = byte(len(metaBytes) >> 24)
	body = append(body, prefix[:]...)
	body = append(body, metaBytes...)
	body = append(body, payload...)

	dep := frame.Dependency{
		Hash:       meta.Base,
		DecodedLen: decodedLen,
		DepDomain:  frame.DepDomainPayload,
		DepFlags:   frame.DepFlagRequired,
		CodecHint:  baseMode,
	}
	parts := frame.PayloadParts(
		frame.ModeSemanticDelta, 0, 0, decodedLen, contentHash, body)
	parts.Dependencies = []frame.Dependency{dep}
	return frame.Encode(parts, lim)
}

// EncodeSemanticDelta encodes target against baseFrame under policy.
func EncodeSemanticDelta(
	ctx context.Context, target []byte, dtype WeightDtype, shape []uint64, baseFrame []byte,
	policy SemanticPolicy, lim limits.Limits,
) ([]byte, error) {
	baseParsed, err := frame.Parse(baseFrame, lim)
	if err != nil {
		return nil, err
	}
	baseDtype, baseShape, err := BaseFrameDescriptor(baseParsed, lim)
	if err != nil {
		return nil, err
	}
	if baseDtype != dtype {
		return nil, werrors.Malformedf(
			"base frame dtype %q does not match SEMANTIC_DELTA dtype %q",
			baseDtype.Name, dtype.Name)
	}
	if !slices.Equal(baseShape, shape) {
		return nil, werrors.NewMalformed(
			"base frame shape does not match the SEMANTIC_DELTA shape")
	}
	elementCount, err := weightDeltaShape(shape, lim)
	if err != nil {
		return nil, err
	}
	expected, err := weightDeltaPayloadLen(elementCount, dtype)
	if err != nil {
		return nil, err
	}
	if err := limits.CheckLen(expected, lim.MaxDecodedBytes, "SEMANTIC_DELTA element bytes"); err != nil {
		return nil, err
	}
	if uint64(len(target)) != expected {
		return nil, werrors.Malformedf(
			"SEMANTIC_DELTA target is %d bytes but shape %v x %s requires %d",
			len(target), shape, dtype.Name, expected)
	}
	if err := policy.Validate(dtype, elementCount); err != nil {
		return nil, err
	}
	baseBytes, err := DecodeBytes(ctx, baseFrame, dependency.NewMapResolver(), lim)
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
	baseRecord, err := NewSemanticRecord(dtype, shape, baseBytes, policy, lim)
	if err != nil {
		return nil, err
	}
	baseRoot, err := baseRecord.Root(lim)
	if err != nil {
		return nil, err
	}
	count, err := limits.ToInt(elementCount, "SEMANTIC_DELTA element count")
	if err != nil {
		return nil, err
	}
	deltaBytes, planes := buildWeightPlanes(dtype, target, baseBytes, count)
	deltaRoot := hash.Domain(tensor.DomainTensorValues, deltaBytes)
	var payload []byte
	for _, plane := range planes {
		payload = append(payload, HuffmanEncode(plane)...)
	}
	meta := SemanticDeltaMeta{
		Dtype:            dtype,
		Shape:            append([]uint64(nil), shape...),
		ElementCount:     elementCount,
		WordBits:         dtype.WordBits,
		Base:             baseParsed.Header.ContentHash,
		BaseDtype:        dtype,
		BaseShape:        append([]uint64(nil), shape...),
		BaseSemanticRoot: baseRoot,
		Policy:           policy,
		DeltaRoot:        deltaRoot,
		PlaneCodec:       WeightDeltaPlaneCodec,
		PayloadLen:       uint64(len(payload)),
	}
	return EncodeSemanticDeltaDescriptorFrame(
		meta, payload, uint64(len(target)), hash.Domain(hash.DomainChunk, target),
		uint32(baseParsed.Header.Mode), lim)
}

// DecodeSemanticDeltaFrame decodes a mode-14 frame, resolving the base.
func DecodeSemanticDeltaFrame(
	ctx context.Context, f *frame.Frame, resolver dependency.Resolver, lim limits.Limits,
) ([]byte, error) {
	return decodeSemanticDelta(ctx, f, nil, resolver, lim)
}

// DecodeSemanticDeltaFrameWithBase decodes a mode-14 frame with the base frame
// supplied out of band.
func DecodeSemanticDeltaFrameWithBase(
	ctx context.Context, f *frame.Frame, baseFrame []byte,
	resolver dependency.Resolver, lim limits.Limits,
) ([]byte, error) {
	return decodeSemanticDelta(ctx, f, baseFrame, resolver, lim)
}

func decodeSemanticDelta(
	ctx context.Context, f *frame.Frame, baseFrame []byte,
	resolver dependency.Resolver, lim limits.Limits,
) ([]byte, error) {
	if f.Header.Mode != frame.ModeSemanticDelta {
		return nil, werrors.Malformedf(
			"expected a SEMANTIC_DELTA frame, got mode %s", f.Header.Mode)
	}
	descriptor, payload, err := splitAIBodyPrefix(f.Body)
	if err != nil {
		return nil, err
	}
	meta, err := parseSemanticDeltaDescriptor(descriptor, lim)
	if err != nil {
		return nil, err
	}
	elementCount, err := ValidateSemanticDeltaMeta(meta, lim)
	if err != nil {
		return nil, err
	}
	expected, err := weightDeltaPayloadLen(elementCount, meta.Dtype)
	if err != nil {
		return nil, err
	}
	if f.Header.DecodedLen != expected {
		return nil, werrors.Corruptf(
			"SEMANTIC_DELTA decoded_len is %d but the descriptor requires %d",
			f.Header.DecodedLen, expected)
	}
	if err := limits.CheckLen(expected, lim.MaxDecodedBytes, "SEMANTIC_DELTA element bytes"); err != nil {
		return nil, err
	}
	if meta.PayloadLen != uint64(len(payload)) {
		return nil, werrors.Malformedf(
			"SEMANTIC_DELTA payload is %d bytes but payload_len declares %d",
			len(payload), meta.PayloadLen)
	}
	if f.Header.DepCount != 1 {
		return nil, werrors.NewMalformed(
			"SEMANTIC_DELTA requires exactly one dependency")
	}
	dep := f.Dependencies[0]
	if dep.Hash != meta.Base {
		return nil, werrors.NewMalformed(
			"SEMANTIC_DELTA dependency hash must equal the descriptor base")
	}
	if dep.DecodedLen != f.Header.DecodedLen {
		return nil, werrors.NewMalformed(
			"SEMANTIC_DELTA dependency decoded_len must equal the frame decoded_len")
	}
	count, err := limits.ToInt(elementCount, "SEMANTIC_DELTA element count")
	if err != nil {
		return nil, err
	}
	planes, err := decodeWeightPlanes(meta.Dtype, payload, count)
	if err != nil {
		return nil, err
	}
	deltaBytes := make([]byte, 0, count*int(meta.Dtype.StorageBytes))
	for index := 0; index < count; index++ {
		delta, err := weightDeltaFromPlanes(meta.Dtype, planes, index)
		if err != nil {
			return nil, err
		}
		deltaBytes = pushWeightWord(meta.Dtype, delta, deltaBytes)
	}
	if hash.Domain(tensor.DomainTensorValues, deltaBytes) != meta.DeltaRoot {
		return nil, werrors.NewCorrupt("SEMANTIC_DELTA delta_root mismatch")
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
				"base frame dtype %q does not match SEMANTIC_DELTA dtype %q",
				baseDtype.Name, meta.Dtype.Name)
		}
		if !slices.Equal(baseShape, meta.Shape) {
			return nil, werrors.NewDependencyMismatch(
				"base frame shape does not match the SEMANTIC_DELTA shape")
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

	recomputedRecord, err := NewSemanticRecord(meta.Dtype, meta.Shape, baseBytes, meta.Policy, lim)
	if err != nil {
		return nil, werrors.DependencyMismatchf("%s", err.Error())
	}
	recomputed, err := recomputedRecord.Root(lim)
	if err != nil {
		return nil, werrors.DependencyMismatchf("%s", err.Error())
	}
	if recomputed != meta.BaseSemanticRoot {
		return nil, werrors.DependencyMismatchf(
			"base semantic root %s under policy %s does not equal the descriptor %s (no approximate reconstruction exists)",
			recomputed.Hex(), meta.Policy.String(), meta.BaseSemanticRoot.Hex())
	}

	mask := weightMask(meta.Dtype)
	target := make([]byte, 0, count*int(meta.Dtype.StorageBytes))
	for index := 0; index < count; index++ {
		delta, err := weightDeltaFromPlanes(meta.Dtype, planes, index)
		if err != nil {
			return nil, err
		}
		word := (delta + weightWordAt(meta.Dtype, baseBytes, index)) & mask
		target = pushWeightWord(meta.Dtype, word, target)
	}
	if uint64(len(target)) != expected {
		return nil, werrors.NewCorrupt(
			"SEMANTIC_DELTA reconstructed element length is inconsistent")
	}
	if hash.Domain(hash.DomainChunk, target) != f.Header.ContentHash {
		return nil, werrors.NewCorrupt(
			"SEMANTIC_DELTA content_hash mismatch against the reconstructed tensor")
	}
	return target, nil
}
