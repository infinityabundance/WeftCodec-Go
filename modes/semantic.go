// Package modes implements semantic tensor roots (WEFT-AI-CODEC §15).
//
// A semantic root is metadata for grouping near-duplicate tensors. It is
// never an identity for exact bytes and never replaces the logical tensor root
// or a frame content hash. `SelectSemanticBase` uses it only to propose a
// candidate base; mode 14 re-verifies that base under the declared policy.
package modes

import (
	"context"

	"github.com/infinityabundance/WeftCodec-Go/dependency"
	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/tensor"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// SemanticSchema is the canonical semantic-root record schema.
const SemanticSchema = "weft.tensor.semantic.v1"

// DomainTensorSemantic is the semantic-root identity domain.
const DomainTensorSemantic = "weft/tensor/semantic/v1"

// DomainTensorSemanticValues is the domain for policy-transformed values.
const DomainTensorSemanticValues = "weft/tensor/semantic-values/v1"

// MaxBlockLevels bounds block_quantize.levels so integer arithmetic cannot
// overflow.
const MaxBlockLevels = 1 << 24

// SemanticPolicy is one canonical semantic policy.
type SemanticPolicy struct {
	Scheme    string
	Bits      uint32
	BlockSize uint64
	Levels    uint64
}

// NonePolicy is the identity policy.
func NonePolicy() SemanticPolicy { return SemanticPolicy{Scheme: "none"} }

// MantissaTruncatePolicy clears the low `bits` mantissa bits.
func MantissaTruncatePolicy(bits uint32) SemanticPolicy {
	return SemanticPolicy{Scheme: "mantissa_truncate", Bits: bits}
}

// BlockQuantizePolicy is symmetric per-block quantization.
func BlockQuantizePolicy(blockSize, levels uint64) SemanticPolicy {
	return SemanticPolicy{Scheme: "block_quantize", BlockSize: blockSize, Levels: levels}
}

// String is the canonical CLI/inspection spelling.
func (p SemanticPolicy) String() string {
	switch p.Scheme {
	case "none":
		return "none"
	case "mantissa_truncate":
		return "mantissa_truncate:" + u32Text(p.Bits)
	case "block_quantize":
		return "block_quantize:" + u64Text(p.BlockSize) + ":" + u64Text(p.Levels)
	default:
		return p.Scheme
	}
}

func u32Text(v uint32) string {
	if v == 0 {
		return "0"
	}
	var buf [10]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}

func u64Text(v uint64) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}

func semanticMantissaBits(dtype WeightDtype) (uint32, bool) {
	switch dtype.Name {
	case "bf16":
		return 7, true
	case "fp16":
		return 10, true
	case "fp32":
		return 23, true
	default:
		return 0, false
	}
}

// ParseSemanticPolicy parses a policy value, validating it against dtype and
// elementCount.
func ParseSemanticPolicy(value wcb.Value, dtype WeightDtype, elementCount uint64) (SemanticPolicy, error) {
	if value.Kind != wcb.Map {
		return SemanticPolicy{}, werrors.NewMalformed("semantic policy must be a WCB map")
	}
	schemeValue, ok := value.Map["scheme"]
	if !ok || schemeValue.Kind != wcb.Text {
		return SemanticPolicy{}, werrors.NewMalformed("semantic policy `scheme` must be text")
	}
	scheme := schemeValue.Str
	switch scheme {
	case "none":
		if err := checkPolicyKeys(value.Map, "none", []string{"scheme"}); err != nil {
			return SemanticPolicy{}, err
		}
		return NonePolicy(), nil
	case "mantissa_truncate":
		if err := checkPolicyKeys(value.Map, "mantissa_truncate", []string{"bits", "scheme"}); err != nil {
			return SemanticPolicy{}, err
		}
		bitsValue, ok := value.Map["bits"]
		if !ok || bitsValue.Kind != wcb.U64 {
			return SemanticPolicy{}, werrors.NewMalformed(
				"semantic policy `mantissa_truncate` key `bits` must be u64")
		}
		maxBits, ok := semanticMantissaBits(dtype)
		if !ok {
			return SemanticPolicy{}, werrors.Malformedf(
				"mantissa_truncate policy is not defined for dtype %s", dtype.Name)
		}
		if bitsValue.U == 0 || bitsValue.U > uint64(maxBits) {
			return SemanticPolicy{}, werrors.Malformedf(
				"mantissa_truncate bits %d is outside 1..=%d for %s", bitsValue.U, maxBits, dtype.Name)
		}
		return MantissaTruncatePolicy(uint32(bitsValue.U)), nil
	case "block_quantize":
		if err := checkPolicyKeys(value.Map, "block_quantize",
			[]string{"block_size", "levels", "scheme"}); err != nil {
			return SemanticPolicy{}, err
		}
		blockValue, ok := value.Map["block_size"]
		if !ok || blockValue.Kind != wcb.U64 {
			return SemanticPolicy{}, werrors.NewMalformed(
				"semantic policy `block_quantize` key `block_size` must be u64")
		}
		levelsValue, ok := value.Map["levels"]
		if !ok || levelsValue.Kind != wcb.U64 {
			return SemanticPolicy{}, werrors.NewMalformed(
				"semantic policy `block_quantize` key `levels` must be u64")
		}
		policy := BlockQuantizePolicy(blockValue.U, levelsValue.U)
		if err := policy.Validate(dtype, elementCount); err != nil {
			return SemanticPolicy{}, err
		}
		return policy, nil
	default:
		return SemanticPolicy{}, werrors.Malformedf(
			"unknown semantic policy scheme %q (v1 allows none, mantissa_truncate, block_quantize)",
			scheme)
	}
}

func checkPolicyKeys(m map[string]wcb.Value, scheme string, allowed []string) error {
	for key := range m {
		found := false
		for _, name := range allowed {
			if key == name {
				found = true
				break
			}
		}
		if !found {
			return werrors.Malformedf(
				"semantic policy %q has unknown key %q", scheme, key)
		}
	}
	for _, name := range allowed {
		if _, ok := m[name]; !ok {
			return werrors.Malformedf(
				"semantic policy %q is missing key %q", scheme, name)
		}
	}
	return nil
}

// WCBValue renders the canonical policy value.
func (p SemanticPolicy) WCBValue() wcb.Value {
	switch p.Scheme {
	case "mantissa_truncate":
		return wcb.MapValue(map[string]wcb.Value{
			"bits":   wcb.U64Value(uint64(p.Bits)),
			"scheme": wcb.TextValue("mantissa_truncate"),
		})
	case "block_quantize":
		return wcb.MapValue(map[string]wcb.Value{
			"block_size": wcb.U64Value(p.BlockSize),
			"levels":     wcb.U64Value(p.Levels),
			"scheme":     wcb.TextValue("block_quantize"),
		})
	default:
		return wcb.MapValue(map[string]wcb.Value{
			"scheme": wcb.TextValue("none"),
		})
	}
}

// Validate checks the policy against a dtype and element count.
func (p SemanticPolicy) Validate(dtype WeightDtype, elementCount uint64) error {
	switch p.Scheme {
	case "none":
		return nil
	case "mantissa_truncate":
		maxBits, ok := semanticMantissaBits(dtype)
		if !ok {
			return werrors.Malformedf(
				"mantissa_truncate policy is not defined for dtype %s", dtype.Name)
		}
		if p.Bits == 0 || p.Bits > maxBits {
			return werrors.Malformedf(
				"mantissa_truncate bits %d is outside 1..=%d for %s", p.Bits, maxBits, dtype.Name)
		}
		return nil
	case "block_quantize":
		if p.BlockSize == 0 {
			return werrors.NewMalformed(
				"block_quantize block_size must be greater than zero")
		}
		if p.Levels == 0 || p.Levels > MaxBlockLevels {
			return werrors.Malformedf(
				"block_quantize levels %d is outside 1..=%d", p.Levels, uint64(MaxBlockLevels))
		}
		if elementCount%p.BlockSize != 0 {
			return werrors.Malformedf(
				"block_quantize block_size %d does not divide element_count %d",
				p.BlockSize, elementCount)
		}
		return nil
	default:
		return werrors.Malformedf("unknown semantic policy scheme %q", p.Scheme)
	}
}

// SemanticRecord is the canonical semantic-root record.
type SemanticRecord struct {
	Dtype               WeightDtype
	Shape               []uint64
	ElementCount        uint64
	Policy              SemanticPolicy
	QuantizedValuesHash hash.Hash32
}

func semanticWordMask(dtype WeightDtype) uint32 {
	switch dtype.WordBits {
	case 32:
		return ^uint32(0)
	case 16:
		return 0xFFFF
	case 8:
		return 0xFF
	case 4:
		return 0x0F
	default:
		return 0
	}
}

func semanticExponentAllOnes(dtype WeightDtype, word uint32) bool {
	switch dtype.Name {
	case "bf16", "fp16":
		return word&0x7F80 == 0x7F80
	case "fp32":
		return word&0x7F800000 == 0x7F800000
	default:
		return false
	}
}

func semanticSignedWord(dtype WeightDtype, word uint32) int64 {
	bits := dtype.WordBits
	sign := uint64(1) << (bits - 1)
	modulus := uint64(1) << bits
	raw := uint64(word)
	if raw >= sign {
		return int64(raw) - int64(modulus)
	}
	return int64(raw)
}

func semanticRawWord(dtype WeightDtype, signed int64) uint32 {
	return uint32(uint64(signed) & uint64(semanticWordMask(dtype)))
}

func semanticTransform(dtype WeightDtype, data []byte, elementCount int, policy SemanticPolicy) []byte {
	switch policy.Scheme {
	case "mantissa_truncate":
		retain := ^uint32((uint64(1) << policy.Bits) - 1)
		out := make([]byte, 0, len(data))
		for index := 0; index < elementCount; index++ {
			word := weightWordAt(dtype, data, index)
			if !semanticExponentAllOnes(dtype, word) {
				word &= retain
			}
			out = pushWeightWord(dtype, word, out)
		}
		return out
	case "block_quantize":
		block := int(policy.BlockSize)
		levels := policy.Levels
		out := make([]byte, 0, len(data))
		for start := 0; start < elementCount; start += block {
			end := start + block
			if end > elementCount {
				end = elementCount
			}
			var amax int64
			for index := start; index < end; index++ {
				value := semanticSignedWord(dtype, weightWordAt(dtype, data, index))
				magnitude := value
				if magnitude < 0 {
					magnitude = -magnitude
				}
				if magnitude > amax {
					amax = magnitude
				}
			}
			if amax == 0 || levels == 1 {
				for index := start; index < end; index++ {
					out = pushWeightWord(dtype, semanticRawWord(dtype, 0), out)
				}
				continue
			}
			span := uint64(2 * amax)
			for index := start; index < end; index++ {
				value := semanticSignedWord(dtype, weightWordAt(dtype, data, index))
				v := uint64(value + amax)
				k := (v*(levels-1) + uint64(amax)) / span
				if k >= levels {
					k = levels - 1
				}
				level := (span*k + (levels-1)/2) / (levels - 1)
				out = pushWeightWord(dtype, semanticRawWord(dtype, int64(level)-amax), out)
			}
		}
		return out
	default:
		out := make([]byte, len(data))
		copy(out, data)
		return out
	}
}

// NewSemanticRecord builds a semantic-root record from canonical row-major
// element bytes.
func NewSemanticRecord(
	dtype WeightDtype, shape []uint64, data []byte, policy SemanticPolicy, lim limits.Limits,
) (SemanticRecord, error) {
	elementCount, err := weightDeltaShape(shape, lim)
	if err != nil {
		return SemanticRecord{}, err
	}
	expected, err := weightDeltaPayloadLen(elementCount, dtype)
	if err != nil {
		return SemanticRecord{}, err
	}
	if err := limits.CheckLen(expected, lim.MaxDecodedBytes, "semantic tensor bytes"); err != nil {
		return SemanticRecord{}, err
	}
	if expected != uint64(len(data)) {
		return SemanticRecord{}, werrors.Malformedf(
			"semantic tensor is %d bytes but shape %v x %s requires %d",
			len(data), shape, dtype.Name, expected)
	}
	if err := policy.Validate(dtype, elementCount); err != nil {
		return SemanticRecord{}, err
	}
	if dtype.IsQuantized() {
		if err := checkQuantizedWords(dtype, data, "semantic tensor"); err != nil {
			return SemanticRecord{}, err
		}
	}
	count, err := limits.ToInt(elementCount, "semantic element count")
	if err != nil {
		return SemanticRecord{}, err
	}
	var valuesHash hash.Hash32
	if policy.Scheme == "none" {
		valuesHash = hash.Domain(tensor.DomainTensorValues, data)
	} else {
		transformed := semanticTransform(dtype, data, count, policy)
		valuesHash = hash.Domain(DomainTensorSemanticValues, transformed)
	}
	shapeCopy := append([]uint64(nil), shape...)
	return SemanticRecord{
		Dtype:               dtype,
		Shape:               shapeCopy,
		ElementCount:        elementCount,
		Policy:              policy,
		QuantizedValuesHash: valuesHash,
	}, nil
}

// Validate checks every structural invariant.
func (r SemanticRecord) Validate(lim limits.Limits) error {
	count, err := weightDeltaShape(r.Shape, lim)
	if err != nil {
		return err
	}
	if r.ElementCount != count {
		return werrors.Malformedf(
			"semantic element_count %d does not match shape product %d", r.ElementCount, count)
	}
	if _, err := weightDeltaPayloadLen(count, r.Dtype); err != nil {
		return err
	}
	return r.Policy.Validate(r.Dtype, count)
}

// Equal reports whether two records describe the same candidate group.
func (r SemanticRecord) Equal(other SemanticRecord) bool {
	if r.Dtype != other.Dtype || r.ElementCount != other.ElementCount || r.Policy != other.Policy {
		return false
	}
	if len(r.Shape) != len(other.Shape) {
		return false
	}
	for i := range r.Shape {
		if r.Shape[i] != other.Shape[i] {
			return false
		}
	}
	return r.QuantizedValuesHash == other.QuantizedValuesHash
}

// WCBValue renders the canonical record.
func (r SemanticRecord) WCBValue() wcb.Value {
	shape := make([]wcb.Value, 0, len(r.Shape))
	for _, dim := range r.Shape {
		shape = append(shape, wcb.U64Value(dim))
	}
	return wcb.MapValue(map[string]wcb.Value{
		"schema":                wcb.TextValue(SemanticSchema),
		"dtype":                 wcb.TextValue(r.Dtype.Name),
		"shape":                 wcb.ArrayValue(shape),
		"element_count":         wcb.U64Value(r.ElementCount),
		"policy":                r.Policy.WCBValue(),
		"quantized_values_hash": wcb.BytesValue(r.QuantizedValuesHash.Bytes()),
	})
}

// CanonicalBytes renders the canonical WCB bytes.
func (r SemanticRecord) CanonicalBytes(lim limits.Limits) ([]byte, error) {
	if err := r.Validate(lim); err != nil {
		return nil, err
	}
	return wcb.Encode(r.WCBValue(), lim)
}

// Root is H("weft/tensor/semantic/v1", canonical bytes).
func (r SemanticRecord) Root(lim limits.Limits) (hash.Hash32, error) {
	canonical, err := r.CanonicalBytes(lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(DomainTensorSemantic, canonical), nil
}

// ParseSemanticRecord decodes a canonical semantic-root record.
func ParseSemanticRecord(data []byte, lim limits.Limits) (SemanticRecord, error) {
	value, err := wcb.Decode(data, lim)
	if err != nil {
		return SemanticRecord{}, err
	}
	if value.Kind != wcb.Map {
		return SemanticRecord{}, werrors.NewMalformed("semantic root record must be a WCB map")
	}
	allowed := map[string]bool{
		"schema": true, "dtype": true, "shape": true, "element_count": true,
		"policy": true, "quantized_values_hash": true,
	}
	for key := range value.Map {
		if !allowed[key] {
			return SemanticRecord{}, werrors.Malformedf(
				"semantic root record has unknown key %q", key)
		}
	}
	for key := range allowed {
		if _, ok := value.Map[key]; !ok {
			return SemanticRecord{}, werrors.Malformedf(
				"semantic root record is missing key %q", key)
		}
	}
	schemaValue := value.Map["schema"]
	if schemaValue.Kind != wcb.Text || schemaValue.Str != SemanticSchema {
		return SemanticRecord{}, werrors.Malformedf("unsupported semantic root schema %q", schemaValue.Str)
	}
	dtypeValue := value.Map["dtype"]
	if dtypeValue.Kind != wcb.Text {
		return SemanticRecord{}, werrors.NewMalformed("semantic root `dtype` must be text")
	}
	dtype, err := ParseWeightDtype(dtypeValue.Str)
	if err != nil {
		return SemanticRecord{}, err
	}
	shapeValue := value.Map["shape"]
	if shapeValue.Kind != wcb.Array {
		return SemanticRecord{}, werrors.NewMalformed("semantic root `shape` must be an array")
	}
	shape := make([]uint64, 0, len(shapeValue.Array))
	for _, item := range shapeValue.Array {
		if item.Kind != wcb.U64 {
			return SemanticRecord{}, werrors.NewMalformed("semantic root `shape` elements must be u64")
		}
		shape = append(shape, item.U)
	}
	countValue := value.Map["element_count"]
	if countValue.Kind != wcb.U64 {
		return SemanticRecord{}, werrors.NewMalformed("semantic root `element_count` must be u64")
	}
	policy, err := ParseSemanticPolicy(value.Map["policy"], dtype, countValue.U)
	if err != nil {
		return SemanticRecord{}, err
	}
	hashValue := value.Map["quantized_values_hash"]
	if hashValue.Kind != wcb.Bytes || len(hashValue.Bytes) != hash.HashLen {
		return SemanticRecord{}, werrors.Malformedf(
			"semantic root `quantized_values_hash` must be %d bytes", hash.HashLen)
	}
	valuesHash, err := hash.FromSlice(hashValue.Bytes)
	if err != nil {
		return SemanticRecord{}, err
	}
	record := SemanticRecord{
		Dtype:               dtype,
		Shape:               shape,
		ElementCount:        countValue.U,
		Policy:              policy,
		QuantizedValuesHash: valuesHash,
	}
	if err := record.Validate(lim); err != nil {
		return SemanticRecord{}, err
	}
	reencoded, err := wcb.Encode(record.WCBValue(), lim)
	if err != nil {
		return SemanticRecord{}, err
	}
	if !bytesEqual(reencoded, data) {
		return SemanticRecord{}, werrors.NewNonCanonical(
			"semantic root bytes are not in canonical form")
	}
	return record, nil
}

// SemanticRoot computes the root with default limits.
func SemanticRoot(dtype WeightDtype, shape []uint64, data []byte, policy SemanticPolicy) (hash.Hash32, error) {
	return SemanticRootWithLimits(dtype, shape, data, policy, limits.Default())
}

// SemanticRootWithLimits computes the root with explicit limits.
func SemanticRootWithLimits(
	dtype WeightDtype, shape []uint64, data []byte, policy SemanticPolicy, lim limits.Limits,
) (hash.Hash32, error) {
	record, err := NewSemanticRecord(dtype, shape, data, policy, lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	return record.Root(lim)
}

// SemanticMatches reports whether two records describe the same candidate group.
func SemanticMatches(a, b SemanticRecord) bool { return a.Equal(b) }

// SelectSemanticBase chooses the cheapest exact base among semantic candidates.
//
// Both the target and every candidate must be present as bytes: the predicted
// cost is the exact per-plane huffman-canonical-v1 size of the modular word
// difference. Candidates whose dtype, shape, or semantic root under policy
// differ are never offered. Ties are broken by the candidate frame content
// hash, so the result is deterministic for a fixed candidate order. It returns
// the chosen index and ok=false when no candidate matches.
func SelectSemanticBase(
	target []byte, dtype WeightDtype, shape []uint64, policy SemanticPolicy,
	candidates [][]byte, lim limits.Limits,
) (int, bool, error) {
	targetRecord, err := NewSemanticRecord(dtype, shape, target, policy, lim)
	if err != nil {
		return 0, false, err
	}
	targetRoot, err := targetRecord.Root(lim)
	if err != nil {
		return 0, false, err
	}
	count, err := limits.ToInt(targetRecord.ElementCount, "semantic element count")
	if err != nil {
		return 0, false, err
	}
	expected, err := weightDeltaPayloadLen(targetRecord.ElementCount, dtype)
	if err != nil {
		return 0, false, err
	}
	bestIndex := -1
	var bestCost uint64
	var bestHash hash.Hash32
	for index, frameBytes := range candidates {
		parsed, err := frame.Parse(frameBytes, lim)
		if err != nil {
			continue
		}
		candidateDtype, candidateShape, err := BaseFrameDescriptor(parsed, lim)
		if err != nil {
			continue
		}
		if candidateDtype != dtype || !semanticShapeEqual(candidateShape, shape) {
			continue
		}
		candidateBytes, err := DecodeBytes(context.Background(), frameBytes, dependency.NewMapResolver(), lim)
		if err != nil {
			continue
		}
		if uint64(len(candidateBytes)) != expected {
			continue
		}
		candidateRecord, err := NewSemanticRecord(dtype, shape, candidateBytes, policy, lim)
		if err != nil {
			continue
		}
		candidateRoot, err := candidateRecord.Root(lim)
		if err != nil || candidateRoot != targetRoot {
			continue
		}
		_, planes := buildWeightPlanes(dtype, target, candidateBytes, count)
		var cost uint64
		for _, plane := range planes {
			cost += uint64(len(HuffmanEncode(plane)))
		}
		key := parsed.Header.ContentHash
		if bestIndex < 0 || cost < bestCost || (cost == bestCost && key.Less(bestHash)) {
			bestIndex = index
			bestCost = cost
			bestHash = key
		}
	}
	if bestIndex < 0 {
		return 0, false, nil
	}
	return bestIndex, true, nil
}

// semanticShapeEqual compares two shape vectors.
func semanticShapeEqual(a, b []uint64) bool {
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
