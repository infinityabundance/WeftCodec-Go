package modes

import (
	"encoding/binary"
	"math"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/varint"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// TokenEncoding is the canonical v1 token payload encoding.
const TokenEncoding = "u64-zigzag-delta-leb128"

// TensorMeta is a parsed TENSOR descriptor.
type TensorMeta struct {
	Dtype        string
	Shape        []uint64
	ElementCount uint64
	PayloadBytes uint64
}

// TokenMeta is a parsed TOKEN_SEQUENCE descriptor.
type TokenMeta struct {
	VocabSchema  string
	TokenCount   uint64
	PayloadBytes uint64
}

func tensorElementSize(dtype string) (uint64, error) {
	switch dtype {
	case "f16", "bf16", "i16":
		return 2, nil
	case "f32", "i32":
		return 4, nil
	case "f64", "i64":
		return 8, nil
	case "i8", "u8":
		return 1, nil
	default:
		return 0, werrors.Malformedf(
			"unknown tensor dtype %q (v1 allows f16, bf16, f32, f64, i8, u8, i16, i32, i64)",
			dtype)
	}
}

func tensorProduct(shape []uint64, lim limits.Limits) (uint64, error) {
	if uint64(len(shape)) > lim.MaxTensorRank {
		return 0, werrors.LimitExceededf(
			"tensor rank %d exceeds limit %d", len(shape), lim.MaxTensorRank)
	}
	product := uint64(1)
	for _, dim := range shape {
		if dim != 0 && product > math.MaxUint64/dim {
			return 0, werrors.NewLimitExceeded("tensor shape product overflows u64")
		}
		product *= dim
	}
	if product > lim.MaxTensorElements {
		return 0, werrors.LimitExceededf(
			"tensor element count %d exceeds limit %d", product, lim.MaxTensorElements)
	}
	return product, nil
}

func buildAIBody(meta wcb.Value, payload []byte, lim limits.Limits) ([]byte, error) {
	metaBytes, err := wcb.Encode(meta, lim)
	if err != nil {
		return nil, err
	}
	if uint64(len(metaBytes)) > math.MaxUint32 {
		return nil, werrors.NewLimitExceeded("AI metadata does not fit u32")
	}
	body := make([]byte, 0, 4+len(metaBytes)+len(payload))
	var prefix [4]byte
	binary.LittleEndian.PutUint32(prefix[:], uint32(len(metaBytes)))
	body = append(body, prefix[:]...)
	body = append(body, metaBytes...)
	body = append(body, payload...)
	return body, nil
}

func splitAIBody(body []byte, decodedLen uint64) ([]byte, []byte, error) {
	if len(body) < 4 {
		return nil, nil, werrors.NewMalformed(
			"AI body is shorter than its 4-byte metadata length")
	}
	metaLen := uint64(binary.LittleEndian.Uint32(body[0:4]))
	metaEnd := uint64(4) + metaLen
	if metaEnd > uint64(len(body)) {
		return nil, nil, werrors.NewMalformed(
			"AI metadata extends past the frame body")
	}
	payload := body[metaEnd:]
	if uint64(len(payload)) != decodedLen {
		return nil, nil, werrors.Malformedf(
			"AI payload length %d differs from decoded_len %d", len(payload), decodedLen)
	}
	return body[4:metaEnd], payload, nil
}

func metaText(m map[string]wcb.Value, key string) (string, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.Text {
		return "", werrors.Malformedf("metadata %q must be text", key)
	}
	return v.Str, nil
}

func metaU64(m map[string]wcb.Value, key string) (uint64, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.U64 {
		return 0, werrors.Malformedf("metadata %q must be u64", key)
	}
	return v.U, nil
}

func metaBytes(m map[string]wcb.Value, key string) ([]byte, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.Bytes {
		return nil, werrors.Malformedf("metadata %q must be bytes", key)
	}
	return v.Bytes, nil
}

func metaShape(m map[string]wcb.Value) ([]uint64, error) {
	v, ok := m["shape"]
	if !ok || v.Kind != wcb.Array {
		return nil, werrors.NewMalformed("metadata \"shape\" must be an array")
	}
	shape := make([]uint64, 0, len(v.Array))
	for _, item := range v.Array {
		if item.Kind != wcb.U64 {
			return nil, werrors.NewMalformed("metadata \"shape\" elements must be u64")
		}
		shape = append(shape, item.U)
	}
	return shape, nil
}

func checkExactKeys(m map[string]wcb.Value, expected []string, what string) error {
	allowed := make(map[string]bool, len(expected))
	for _, key := range expected {
		allowed[key] = true
	}
	for key := range m {
		if !allowed[key] {
			return werrors.Malformedf("%s metadata has unknown key %q", what, key)
		}
	}
	for _, key := range expected {
		if _, ok := m[key]; !ok {
			return werrors.Malformedf("%s metadata is missing key %q", what, key)
		}
	}
	return nil
}

// EncodeTensor encodes a canonical tensor payload (WEFT-AI-CODEC §2).
func EncodeTensor(payload []byte, dtype string, shape []uint64, lim limits.Limits) ([]byte, error) {
	if err := limits.CheckLen(uint64(len(payload)), lim.MaxDecodedBytes, "tensor payload"); err != nil {
		return nil, err
	}
	size, err := tensorElementSize(dtype)
	if err != nil {
		return nil, err
	}
	elementCount, err := tensorProduct(shape, lim)
	if err != nil {
		return nil, err
	}
	if elementCount != 0 && size > math.MaxUint64/elementCount {
		return nil, werrors.NewLimitExceeded("tensor payload byte count overflows u64")
	}
	payloadBytes := elementCount * size
	if payloadBytes != uint64(len(payload)) {
		return nil, werrors.Malformedf(
			"tensor payload is %d bytes but shape %v x %s is %d bytes",
			len(payload), shape, dtype, payloadBytes)
	}
	shapeItems := make([]wcb.Value, 0, len(shape))
	for _, dim := range shape {
		shapeItems = append(shapeItems, wcb.U64Value(dim))
	}
	meta := wcb.MapValue(map[string]wcb.Value{
		"schema":        wcb.TextValue("weft.tensor.v1"),
		"dtype":         wcb.TextValue(dtype),
		"shape":         wcb.ArrayValue(shapeItems),
		"layout":        wcb.TextValue("row_major"),
		"endianness":    wcb.TextValue("little"),
		"element_count": wcb.U64Value(elementCount),
		"payload_bytes": wcb.U64Value(payloadBytes),
		"payload_hash":  wcb.BytesValue(hash.Domain(hash.DomainTensorPayload, payload).Bytes()),
	})
	body, err := buildAIBody(meta, payload, lim)
	if err != nil {
		return nil, err
	}
	parts := frame.PayloadParts(
		frame.ModeTensor, 0, 0, uint64(len(payload)),
		hash.Domain(hash.DomainChunk, payload), body)
	return frame.Encode(parts, lim)
}

func parseTensorBody(f *frame.Frame, lim limits.Limits) (TensorMeta, []byte, error) {
	descriptorBytes, payload, err := splitAIBody(f.Body, f.Header.DecodedLen)
	if err != nil {
		return TensorMeta{}, nil, err
	}
	value, err := wcb.Decode(descriptorBytes, lim)
	if err != nil {
		return TensorMeta{}, nil, err
	}
	if value.Kind != wcb.Map {
		return TensorMeta{}, nil, werrors.NewMalformed("tensor metadata must be a WCB map")
	}
	m := value.Map
	if err := checkExactKeys(m, []string{
		"schema", "dtype", "shape", "layout", "endianness",
		"element_count", "payload_bytes", "payload_hash",
	}, "tensor"); err != nil {
		return TensorMeta{}, nil, err
	}
	schema, err := metaText(m, "schema")
	if err != nil {
		return TensorMeta{}, nil, err
	}
	if schema != "weft.tensor.v1" {
		return TensorMeta{}, nil, werrors.Malformedf("unsupported tensor schema %q", schema)
	}
	dtype, err := metaText(m, "dtype")
	if err != nil {
		return TensorMeta{}, nil, err
	}
	size, err := tensorElementSize(dtype)
	if err != nil {
		return TensorMeta{}, nil, err
	}
	layout, err := metaText(m, "layout")
	if err != nil {
		return TensorMeta{}, nil, err
	}
	if layout != "row_major" {
		return TensorMeta{}, nil, werrors.Malformedf(
			"tensor layout %q is specified but not implemented in v1 (only row_major)", layout)
	}
	endianness, err := metaText(m, "endianness")
	if err != nil {
		return TensorMeta{}, nil, err
	}
	if endianness != "little" {
		return TensorMeta{}, nil, werrors.Malformedf(
			"tensor endianness %q is specified but not implemented in v1 (only little)", endianness)
	}
	shape, err := metaShape(m)
	if err != nil {
		return TensorMeta{}, nil, err
	}
	declaredCount, err := metaU64(m, "element_count")
	if err != nil {
		return TensorMeta{}, nil, err
	}
	declaredBytes, err := metaU64(m, "payload_bytes")
	if err != nil {
		return TensorMeta{}, nil, err
	}
	expectedCount, err := tensorProduct(shape, lim)
	if err != nil {
		return TensorMeta{}, nil, err
	}
	if declaredCount != expectedCount {
		return TensorMeta{}, nil, werrors.Malformedf(
			"tensor element_count %d does not match shape product %d",
			declaredCount, expectedCount)
	}
	if declaredCount != 0 && size > math.MaxUint64/declaredCount {
		return TensorMeta{}, nil, werrors.NewLimitExceeded(
			"tensor payload byte count overflows u64")
	}
	expectedBytes := declaredCount * size
	if declaredBytes != expectedBytes {
		return TensorMeta{}, nil, werrors.Malformedf(
			"tensor payload_bytes %d does not match element_count x element_size %d",
			declaredBytes, expectedBytes)
	}
	if declaredBytes != uint64(len(payload)) {
		return TensorMeta{}, nil, werrors.Malformedf(
			"tensor payload is %d bytes but metadata declares %d", len(payload), declaredBytes)
	}
	declaredHash, err := metaBytes(m, "payload_hash")
	if err != nil {
		return TensorMeta{}, nil, err
	}
	if len(declaredHash) != hash.HashLen {
		return TensorMeta{}, nil, werrors.Malformedf(
			"tensor payload_hash must be %d bytes, got %d", hash.HashLen, len(declaredHash))
	}
	actual := hash.Domain(hash.DomainTensorPayload, payload)
	if !bytesEqual(declaredHash, actual.Bytes()) {
		return TensorMeta{}, nil, werrors.NewCorrupt("tensor payload_hash mismatch")
	}
	return TensorMeta{
		Dtype:        dtype,
		Shape:        shape,
		ElementCount: declaredCount,
		PayloadBytes: declaredBytes,
	}, payload, nil
}

// TensorMetadata validates and returns a TENSOR descriptor.
func TensorMetadata(f *frame.Frame, lim limits.Limits) (TensorMeta, error) {
	meta, _, err := parseTensorBody(f, lim)
	return meta, err
}

// DecodeTensorFrame validates a TENSOR frame and returns its exact payload.
func DecodeTensorFrame(f *frame.Frame, lim limits.Limits) ([]byte, error) {
	_, payload, err := parseTensorBody(f, lim)
	if err != nil {
		return nil, err
	}
	out := make([]byte, len(payload))
	copy(out, payload)
	return out, nil
}

// EncodeTokenPayload encodes token ids to the canonical delta payload.
//
// The first id is an absolute minimal LEB128 u64. Every later id is
// token[i]-token[i-1] in wrapping 64-bit arithmetic, zigzag-encoded and
// written as a minimal LEB128 u64.
func EncodeTokenPayload(ids []uint64, lim limits.Limits) ([]byte, error) {
	out := make([]byte, 0, len(ids))
	var previous uint64
	for i, token := range ids {
		if i == 0 {
			out = varint.Append(out, token)
		} else {
			delta := token - previous
			out = varint.Append(out, varint.ZigzagEncode(int64(delta)))
		}
		previous = token
		if err := limits.CheckLen(uint64(len(out)), lim.MaxDecodedBytes, "token payload"); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// DecodeTokenPayload decodes a canonical token payload into ids.
func DecodeTokenPayload(payload []byte, tokenCount uint64, lim limits.Limits) ([]uint64, error) {
	if err := validateTokenCount(tokenCount, payload, lim); err != nil {
		return nil, err
	}
	count, err := limits.ToInt(tokenCount, "token count")
	if err != nil {
		return nil, err
	}
	ids := make([]uint64, 0, count)
	pos := 0
	var previous uint64
	for i := 0; i < count; i++ {
		raw, next, err := varint.DecodeAt(payload, pos)
		if err != nil {
			return nil, err
		}
		pos = next
		var token uint64
		if i == 0 {
			token = raw
		} else {
			token = previous + uint64(varint.ZigzagDecode(raw))
		}
		ids = append(ids, token)
		previous = token
	}
	if pos != len(payload) {
		return nil, werrors.Malformedf(
			"token payload has %d trailing bytes", len(payload)-pos)
	}
	return ids, nil
}

// ValidateTokenPayload validates a payload without materializing the id slice.
func ValidateTokenPayload(payload []byte, tokenCount uint64, lim limits.Limits) error {
	if err := validateTokenCount(tokenCount, payload, lim); err != nil {
		return err
	}
	count, err := limits.ToInt(tokenCount, "token count")
	if err != nil {
		return err
	}
	pos := 0
	var previous uint64
	for i := 0; i < count; i++ {
		raw, next, err := varint.DecodeAt(payload, pos)
		if err != nil {
			return err
		}
		pos = next
		if i == 0 {
			previous = raw
		} else {
			previous = previous + uint64(varint.ZigzagDecode(raw))
		}
	}
	if pos != len(payload) {
		return werrors.Malformedf(
			"token payload has %d trailing bytes", len(payload)-pos)
	}
	return nil
}

func validateTokenCount(tokenCount uint64, payload []byte, lim limits.Limits) error {
	if tokenCount > lim.MaxTokens {
		return werrors.LimitExceededf(
			"token count %d exceeds limit %d", tokenCount, lim.MaxTokens)
	}
	if tokenCount > uint64(len(payload)) {
		return werrors.Malformedf(
			"token count %d cannot fit in a %d-byte payload", tokenCount, len(payload))
	}
	return nil
}

// EncodeTokens encodes a token sequence (WEFT-AI-CODEC §3).
func EncodeTokens(ids []uint64, vocabSchema string, lim limits.Limits) ([]byte, error) {
	if vocabSchema == "" {
		return nil, werrors.NewMalformed("vocab_schema must not be empty")
	}
	if uint64(len(ids)) > lim.MaxTokens {
		return nil, werrors.LimitExceededf(
			"token count %d exceeds limit %d", len(ids), lim.MaxTokens)
	}
	payload, err := EncodeTokenPayload(ids, lim)
	if err != nil {
		return nil, err
	}
	if err := limits.CheckLen(uint64(len(payload)), lim.MaxDecodedBytes, "token payload"); err != nil {
		return nil, err
	}
	meta := wcb.MapValue(map[string]wcb.Value{
		"schema":        wcb.TextValue("weft.tokens.v1"),
		"vocab_schema":  wcb.TextValue(vocabSchema),
		"encoding":      wcb.TextValue(TokenEncoding),
		"token_count":   wcb.U64Value(uint64(len(ids))),
		"payload_bytes": wcb.U64Value(uint64(len(payload))),
		"payload_hash":  wcb.BytesValue(hash.Domain(hash.DomainTokensPayload, payload).Bytes()),
	})
	body, err := buildAIBody(meta, payload, lim)
	if err != nil {
		return nil, err
	}
	parts := frame.PayloadParts(
		frame.ModeTokenSequence, 0, 0, uint64(len(payload)),
		hash.Domain(hash.DomainChunk, payload), body)
	return frame.Encode(parts, lim)
}

func parseTokenBody(f *frame.Frame, lim limits.Limits) (TokenMeta, []byte, error) {
	descriptorBytes, payload, err := splitAIBody(f.Body, f.Header.DecodedLen)
	if err != nil {
		return TokenMeta{}, nil, err
	}
	value, err := wcb.Decode(descriptorBytes, lim)
	if err != nil {
		return TokenMeta{}, nil, err
	}
	if value.Kind != wcb.Map {
		return TokenMeta{}, nil, werrors.NewMalformed("token metadata must be a WCB map")
	}
	m := value.Map
	if err := checkExactKeys(m, []string{
		"schema", "vocab_schema", "encoding", "token_count",
		"payload_bytes", "payload_hash",
	}, "token"); err != nil {
		return TokenMeta{}, nil, err
	}
	schema, err := metaText(m, "schema")
	if err != nil {
		return TokenMeta{}, nil, err
	}
	if schema != "weft.tokens.v1" {
		return TokenMeta{}, nil, werrors.Malformedf("unsupported token schema %q", schema)
	}
	vocabSchema, err := metaText(m, "vocab_schema")
	if err != nil {
		return TokenMeta{}, nil, err
	}
	if vocabSchema == "" {
		return TokenMeta{}, nil, werrors.NewMalformed("token vocab_schema must not be empty")
	}
	encoding, err := metaText(m, "encoding")
	if err != nil {
		return TokenMeta{}, nil, err
	}
	if encoding != TokenEncoding {
		return TokenMeta{}, nil, werrors.Malformedf(
			"token encoding %q is specified but not implemented in v1 (only %s)",
			encoding, TokenEncoding)
	}
	tokenCount, err := metaU64(m, "token_count")
	if err != nil {
		return TokenMeta{}, nil, err
	}
	if tokenCount > lim.MaxTokens {
		return TokenMeta{}, nil, werrors.LimitExceededf(
			"token count %d exceeds limit %d", tokenCount, lim.MaxTokens)
	}
	declaredBytes, err := metaU64(m, "payload_bytes")
	if err != nil {
		return TokenMeta{}, nil, err
	}
	if declaredBytes != uint64(len(payload)) {
		return TokenMeta{}, nil, werrors.Malformedf(
			"token payload is %d bytes but metadata declares %d", len(payload), declaredBytes)
	}
	declaredHash, err := metaBytes(m, "payload_hash")
	if err != nil {
		return TokenMeta{}, nil, err
	}
	if len(declaredHash) != hash.HashLen {
		return TokenMeta{}, nil, werrors.Malformedf(
			"token payload_hash must be %d bytes, got %d", hash.HashLen, len(declaredHash))
	}
	actual := hash.Domain(hash.DomainTokensPayload, payload)
	if !bytesEqual(declaredHash, actual.Bytes()) {
		return TokenMeta{}, nil, werrors.NewCorrupt("token payload_hash mismatch")
	}
	if err := ValidateTokenPayload(payload, tokenCount, lim); err != nil {
		return TokenMeta{}, nil, err
	}
	return TokenMeta{
		VocabSchema:  vocabSchema,
		TokenCount:   tokenCount,
		PayloadBytes: declaredBytes,
	}, payload, nil
}

// TokenMetadata validates and returns a TOKEN_SEQUENCE descriptor.
func TokenMetadata(f *frame.Frame, lim limits.Limits) (TokenMeta, error) {
	meta, _, err := parseTokenBody(f, lim)
	return meta, err
}

// DecodeTokensFrame validates a TOKEN_SEQUENCE frame and returns its payload.
func DecodeTokensFrame(f *frame.Frame, lim limits.Limits) ([]byte, error) {
	_, payload, err := parseTokenBody(f, lim)
	if err != nil {
		return nil, err
	}
	out := make([]byte, len(payload))
	copy(out, payload)
	return out, nil
}
