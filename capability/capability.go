// Package capability implements authenticated capability envelopes
// (WEFT-CAPABILITIES §9).
//
// An envelope is a canonical WCB record (weft.capability.envelope.v1) signed
// with Ed25519. The envelope is public; only the grantor's private key is
// secret. Verification is offline: the caller supplies the revocation set and
// the clock. Revocation coordination is explicitly external.
//
//	EnvelopeId = H("weft/capability/envelope/v1", WCB(envelope))
//	sign_input = H("weft/capability/sign/v1",     WCB(envelope))
//	BlobId     = H("weft/capability/blob/v1",     WCB({envelope, signature}))
package capability

import (
	"bytes"
	"crypto/ed25519"
	"encoding/json"
	"fmt"

	"github.com/infinityabundance/WeftCodec-Go/context"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// EnvelopeSchema is the canonical schema string.
const EnvelopeSchema = "weft.capability.envelope.v1"

// Domain separation strings.
const (
	DomainEnvelope = "weft/capability/envelope/v1"
	DomainSign     = "weft/capability/sign/v1"
	DomainBlob     = "weft/capability/blob/v1"
)

// SignatureLen is the Ed25519 signature length.
const SignatureLen = 64

// MaxTenantBytes bounds a tenant label.
const MaxTenantBytes = 256

// Rejection is a typed envelope rejection. The strings are the interoperable
// names and match the Rust implementation exactly.
type Rejection string

const (
	RejectionMalformed     Rejection = "Malformed"
	RejectionNonCanonical  Rejection = "NonCanonical"
	RejectionUnknownSchema Rejection = "UnknownSchema"
	RejectionBadSignature  Rejection = "BadSignature"
	RejectionWrongGrantee  Rejection = "WrongGrantee"
	RejectionExpired       Rejection = "Expired"
	RejectionRevoked       Rejection = "Revoked"
)

// Class maps a rejection to a WeftCodec error class.
func (r Rejection) Class() werrors.Class {
	switch r {
	case RejectionMalformed:
		return werrors.Malformed
	case RejectionNonCanonical:
		return werrors.NonCanonical
	case RejectionUnknownSchema:
		return werrors.UnknownSchema
	case RejectionBadSignature:
		return werrors.Corrupt
	case RejectionWrongGrantee, RejectionExpired, RejectionRevoked:
		return werrors.DependencyDenied
	default:
		return werrors.Malformed
	}
}

// EnvelopeError is a rejection with a human-readable message.
type EnvelopeError struct {
	Rejection Rejection
	Message   string
}

func (e *EnvelopeError) Error() string { return fmt.Sprintf("%s: %s", e.Rejection, e.Message) }

// Class returns the normative codec class for this rejection.
func (e *EnvelopeError) Class() werrors.Class { return e.Rejection.Class() }

func rejectionFrom(err error) Rejection {
	switch werrors.ClassOf(err) {
	case werrors.NonCanonical:
		return RejectionNonCanonical
	case werrors.UnknownSchema:
		return RejectionUnknownSchema
	default:
		return RejectionMalformed
	}
}

func envelopeError(err error) *EnvelopeError {
	return &EnvelopeError{Rejection: rejectionFrom(err), Message: err.Error()}
}

func newRejection(rejection Rejection, message string) *EnvelopeError {
	return &EnvelopeError{Rejection: rejection, Message: message}
}

// Envelope is an authenticated capability grant. Every field is public.
type Envelope struct {
	Grantor      hash.Hash32
	Grantee      hash.Hash32
	Tenant       string
	Roots        []hash.Hash32
	Namespaces   []string
	Predicates   []string
	ExpiresAtNS  *int64
	Budget       *context.Budget
	RevocationID hash.Hash32
	IssuedAtNS   int64
	Nonce        hash.Hash32
}

func envelopeLimits() limits.Limits {
	lim := limits.Default()
	lim.MaxStructuredItems = 65536
	lim.MaxStructuredDepth = 16
	lim.MaxStringBytes = 64 * 1024
	return lim
}

// New builds and validates a canonical envelope.
func New(
	grantor, grantee hash.Hash32,
	tenant string,
	roots []hash.Hash32,
	namespaces, predicates []string,
	expiresAtNS *int64,
	budget *context.Budget,
	revocationID hash.Hash32,
	issuedAtNS int64,
	nonce hash.Hash32,
) (*Envelope, error) {
	e := &Envelope{
		Grantor:      grantor,
		Grantee:      grantee,
		Tenant:       tenant,
		Roots:        append([]hash.Hash32(nil), roots...),
		Namespaces:   append([]string(nil), namespaces...),
		Predicates:   append([]string(nil), predicates...),
		ExpiresAtNS:  expiresAtNS,
		Budget:       budget,
		RevocationID: revocationID,
		IssuedAtNS:   issuedAtNS,
		Nonce:        nonce,
	}
	if err := e.Validate(envelopeLimits()); err != nil {
		return nil, err
	}
	return e, nil
}

// Validate enforces every field and cross-field rule.
func (e *Envelope) Validate(lim limits.Limits) error {
	if e.Grantor == hash.Zero {
		return werrors.NewNonCanonical("capability envelope grantor must not be zero")
	}
	if e.Grantee == hash.Zero {
		return werrors.NewNonCanonical("capability envelope grantee must not be zero")
	}
	if e.RevocationID == hash.Zero {
		return werrors.NewNonCanonical("capability envelope revocation_id must not be zero")
	}
	if e.Nonce == hash.Zero {
		return werrors.NewNonCanonical("capability envelope nonce must not be zero")
	}
	if len(e.Tenant) == 0 || len(e.Tenant) > MaxTenantBytes {
		return werrors.NewNonCanonical(
			fmt.Sprintf("capability envelope tenant must be 1..=%d bytes", MaxTenantBytes))
	}
	// Reuse the frozen capability-scope rules for grants, budgets, and
	// sorted/unique arrays. NewCapabilityScope normalizes, so a mismatch means
	// the input was not already canonical.
	scope, err := context.NewCapabilityScope(
		e.Tenant, append([]string(nil), e.Namespaces...), append([]hash.Hash32(nil), e.Roots...),
		append([]string(nil), e.Predicates...), e.Budget, e.ExpiresAtNS)
	if err != nil {
		return err
	}
	if !equalStrings(scope.Namespaces, e.Namespaces) ||
		!equalHashes(scope.Roots, e.Roots) ||
		!equalStrings(scope.Predicates, e.Predicates) {
		return werrors.NewNonCanonical(
			"capability envelope grant lists must be sorted and unique")
	}
	if e.ExpiresAtNS != nil && *e.ExpiresAtNS < e.IssuedAtNS {
		return werrors.NewNonCanonical(
			"capability envelope expires_at_ns precedes issued_at_ns")
	}
	encoded, err := e.encodeWithLimits(lim)
	if err != nil {
		return err
	}
	if uint64(len(encoded)) > lim.MaxStringBytes {
		return werrors.LimitExceededf(
			"capability envelope is %d bytes, above the %d-byte bound", len(encoded), lim.MaxStringBytes)
	}
	return nil
}

// ToWCB returns the canonical WCB value. Every key is always present.
func (e *Envelope) ToWCB() wcb.Value {
	roots := make([]wcb.Value, 0, len(e.Roots))
	for _, r := range e.Roots {
		roots = append(roots, wcb.BytesValue(r.Bytes()))
	}
	namespaces := make([]wcb.Value, 0, len(e.Namespaces))
	for _, n := range e.Namespaces {
		namespaces = append(namespaces, wcb.TextValue(n))
	}
	predicates := make([]wcb.Value, 0, len(e.Predicates))
	for _, p := range e.Predicates {
		predicates = append(predicates, wcb.TextValue(p))
	}
	m := map[string]wcb.Value{
		"schema":        wcb.TextValue(EnvelopeSchema),
		"grantor":       wcb.BytesValue(e.Grantor.Bytes()),
		"grantee":       wcb.BytesValue(e.Grantee.Bytes()),
		"tenant":        wcb.TextValue(e.Tenant),
		"roots":         wcb.ArrayValue(roots),
		"namespaces":    wcb.ArrayValue(namespaces),
		"predicates":    wcb.ArrayValue(predicates),
		"revocation_id": wcb.BytesValue(e.RevocationID.Bytes()),
		"issued_at_ns":  wcb.TimestampValue(e.IssuedAtNS),
		"nonce":         wcb.BytesValue(e.Nonce.Bytes()),
	}
	if e.ExpiresAtNS != nil {
		m["expires_at_ns"] = wcb.TimestampValue(*e.ExpiresAtNS)
	} else {
		m["expires_at_ns"] = wcb.NullValue()
	}
	if e.Budget != nil {
		m["budget"] = e.Budget.ToWCB()
	} else {
		m["budget"] = wcb.NullValue()
	}
	return wcb.MapValue(m)
}

func (e *Envelope) encodeWithLimits(lim limits.Limits) ([]byte, error) {
	return wcb.Encode(e.ToWCB(), lim)
}

// CanonicalBytes encodes the record canonically.
func (e *Envelope) CanonicalBytes(lim limits.Limits) ([]byte, error) {
	if err := e.Validate(lim); err != nil {
		return nil, err
	}
	return e.encodeWithLimits(lim)
}

// ID returns H("weft/capability/envelope/v1", canonical bytes).
func (e *Envelope) ID() (hash.Hash32, error) {
	b, err := e.CanonicalBytes(envelopeLimits())
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(DomainEnvelope, b), nil
}

// SignInput returns H("weft/capability/sign/v1", canonical bytes).
func (e *Envelope) SignInput() (hash.Hash32, error) {
	b, err := e.CanonicalBytes(envelopeLimits())
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(DomainSign, b), nil
}

// EnvelopeFromCanonicalBytes decodes strict canonical bytes.
func EnvelopeFromCanonicalBytes(b []byte, lim limits.Limits) (*Envelope, error) {
	value, err := wcb.Decode(b, lim)
	if err != nil {
		return nil, err
	}
	if value.Kind != wcb.Map {
		return nil, werrors.NewNonCanonical("capability envelope must be a WCB map")
	}
	m := value.Map
	for key := range m {
		switch key {
		case "schema", "grantor", "grantee", "tenant", "roots", "namespaces",
			"predicates", "expires_at_ns", "budget", "revocation_id",
			"issued_at_ns", "nonce":
		default:
			return nil, werrors.NonCanonicalf("unknown capability envelope key `%s`", key)
		}
	}
	for _, key := range []string{"schema", "grantor", "grantee", "tenant", "roots",
		"namespaces", "predicates", "expires_at_ns", "budget", "revocation_id",
		"issued_at_ns", "nonce"} {
		if _, ok := m[key]; !ok {
			return nil, werrors.Malformedf("capability envelope is missing `%s`", key)
		}
	}
	schema, err := requiredText(m, "schema")
	if err != nil {
		return nil, err
	}
	if schema != EnvelopeSchema {
		return nil, werrors.UnknownSchemaf("unknown capability envelope schema `%s`", schema)
	}
	grantor, err := requiredHash(m, "grantor")
	if err != nil {
		return nil, err
	}
	grantee, err := requiredHash(m, "grantee")
	if err != nil {
		return nil, err
	}
	tenant, err := requiredText(m, "tenant")
	if err != nil {
		return nil, err
	}
	roots, err := requiredHashArray(m, "roots")
	if err != nil {
		return nil, err
	}
	namespaces, err := requiredTextArray(m, "namespaces")
	if err != nil {
		return nil, err
	}
	predicates, err := requiredTextArray(m, "predicates")
	if err != nil {
		return nil, err
	}
	var expiresAt *int64
	switch v := m["expires_at_ns"]; v.Kind {
	case wcb.Null:
	case wcb.Timestamp, wcb.I64:
		expires := v.I
		expiresAt = &expires
	default:
		return nil, werrors.NewNonCanonical(
			"capability envelope expires_at_ns must be a timestamp or null")
	}
	var budget *context.Budget
	switch v := m["budget"]; v.Kind {
	case wcb.Null:
	case wcb.Map:
		budget, err = context.BudgetFromWCB(v)
		if err != nil {
			return nil, err
		}
	default:
		return nil, werrors.NewNonCanonical("capability envelope budget must be a map or null")
	}
	revocationID, err := requiredHash(m, "revocation_id")
	if err != nil {
		return nil, err
	}
	issuedVal := m["issued_at_ns"]
	if issuedVal.Kind != wcb.Timestamp && issuedVal.Kind != wcb.I64 {
		return nil, werrors.NewNonCanonical("capability envelope issued_at_ns must be a timestamp")
	}
	nonce, err := requiredHash(m, "nonce")
	if err != nil {
		return nil, err
	}
	e := &Envelope{
		Grantor:      grantor,
		Grantee:      grantee,
		Tenant:       tenant,
		Roots:        roots,
		Namespaces:   namespaces,
		Predicates:   predicates,
		ExpiresAtNS:  expiresAt,
		Budget:       budget,
		RevocationID: revocationID,
		IssuedAtNS:   issuedVal.I,
		Nonce:        nonce,
	}
	if err := e.Validate(lim); err != nil {
		return nil, err
	}
	reencoded, err := e.encodeWithLimits(lim)
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(reencoded, b) {
		return nil, werrors.NewNonCanonical("capability envelope bytes are not in canonical form")
	}
	return e, nil
}

// CapabilityScope returns the envelope grant as a frozen capability scope.
func (e *Envelope) CapabilityScope() (*context.CapabilityScope, error) {
	return context.NewCapabilityScope(
		e.Tenant, append([]string(nil), e.Namespaces...), append([]hash.Hash32(nil), e.Roots...),
		append([]string(nil), e.Predicates...), e.Budget, e.ExpiresAtNS)
}

// EnvelopeFromJSON parses a JSON envelope document (CLI convenience).
func EnvelopeFromJSON(text []byte) (*Envelope, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(text, &raw); err != nil {
		return nil, werrors.NewParse(fmt.Sprintf("bad capability envelope JSON: %v", err))
	}
	for key := range raw {
		switch key {
		case "schema", "grantor", "grantee", "tenant", "roots", "namespaces",
			"predicates", "expires_at_ns", "budget", "revocation_id",
			"issued_at_ns", "nonce":
		default:
			return nil, werrors.NonCanonicalf("unknown capability envelope key `%s`", key)
		}
	}
	var schema string
	if err := json.Unmarshal(raw["schema"], &schema); err != nil {
		return nil, werrors.NewParse("capability envelope `schema` must be text")
	}
	if schema != EnvelopeSchema {
		return nil, werrors.UnknownSchemaf("unknown capability envelope schema `%s`", schema)
	}
	grantor, err := jsonHash(raw["grantor"], "grantor")
	if err != nil {
		return nil, err
	}
	grantee, err := jsonHash(raw["grantee"], "grantee")
	if err != nil {
		return nil, err
	}
	var tenant string
	if err := json.Unmarshal(raw["tenant"], &tenant); err != nil {
		return nil, werrors.NewParse("capability envelope `tenant` must be text")
	}
	var rootTexts []string
	if len(raw["roots"]) != 0 {
		if err := json.Unmarshal(raw["roots"], &rootTexts); err != nil {
			return nil, werrors.NewParse("capability envelope `roots` must be an array of hex text")
		}
	}
	roots := make([]hash.Hash32, 0, len(rootTexts))
	for _, text := range rootTexts {
		h, err := hash.FromHex(text)
		if err != nil {
			return nil, err
		}
		roots = append(roots, h)
	}
	var namespaces []string
	if len(raw["namespaces"]) != 0 {
		if err := json.Unmarshal(raw["namespaces"], &namespaces); err != nil {
			return nil, werrors.NewParse("capability envelope `namespaces` must be an array of text")
		}
	}
	var predicates []string
	if len(raw["predicates"]) != 0 {
		if err := json.Unmarshal(raw["predicates"], &predicates); err != nil {
			return nil, werrors.NewParse("capability envelope `predicates` must be an array of text")
		}
	}
	var expiresAt *int64
	if len(raw["expires_at_ns"]) != 0 && string(raw["expires_at_ns"]) != "null" {
		var value int64
		if err := json.Unmarshal(raw["expires_at_ns"], &value); err != nil {
			return nil, werrors.NewParse("capability envelope `expires_at_ns` must be an integer")
		}
		expiresAt = &value
	}
	var budget *context.Budget
	if len(raw["budget"]) != 0 && string(raw["budget"]) != "null" {
		budget, err = jsonBudget(raw["budget"])
		if err != nil {
			return nil, err
		}
	}
	revocationID, err := jsonHash(raw["revocation_id"], "revocation_id")
	if err != nil {
		return nil, err
	}
	var issuedAt int64
	if err := json.Unmarshal(raw["issued_at_ns"], &issuedAt); err != nil {
		return nil, werrors.NewParse("capability envelope `issued_at_ns` must be an integer")
	}
	nonce, err := jsonHash(raw["nonce"], "nonce")
	if err != nil {
		return nil, err
	}
	return New(grantor, grantee, tenant, roots, namespaces, predicates, expiresAt, budget,
		revocationID, issuedAt, nonce)
}

// ToJSON returns a JSON object representation.
func (e *Envelope) ToJSON() map[string]any {
	roots := make([]string, 0, len(e.Roots))
	for _, r := range e.Roots {
		roots = append(roots, r.Hex())
	}
	var expires any
	if e.ExpiresAtNS != nil {
		expires = *e.ExpiresAtNS
	}
	var budget any
	if e.Budget != nil {
		budget = budgetJSON(e.Budget)
	}
	return map[string]any{
		"schema":        EnvelopeSchema,
		"grantor":       e.Grantor.Hex(),
		"grantee":       e.Grantee.Hex(),
		"tenant":        e.Tenant,
		"roots":         roots,
		"namespaces":    e.Namespaces,
		"predicates":    e.Predicates,
		"expires_at_ns": expires,
		"budget":        budget,
		"revocation_id": e.RevocationID.Hex(),
		"issued_at_ns":  e.IssuedAtNS,
		"nonce":         e.Nonce.Hex(),
	}
}

func budgetJSON(b *context.Budget) map[string]any {
	out := map[string]any{}
	if b.MaxResults != nil {
		out["max_results"] = *b.MaxResults
	}
	if b.MaxBytes != nil {
		out["max_bytes"] = *b.MaxBytes
	}
	if b.MaxDepth != nil {
		out["max_depth"] = *b.MaxDepth
	}
	if b.MaxWork != nil {
		out["max_work"] = *b.MaxWork
	}
	return out
}

func jsonBudget(raw json.RawMessage) (*context.Budget, error) {
	var fields map[string]uint64
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, werrors.NewParse("capability envelope `budget` must be an object of u64")
	}
	b := &context.Budget{}
	for key, value := range fields {
		v := value
		switch key {
		case "max_results":
			b.MaxResults = &v
		case "max_bytes":
			b.MaxBytes = &v
		case "max_depth":
			b.MaxDepth = &v
		case "max_work":
			b.MaxWork = &v
		default:
			return nil, werrors.NonCanonicalf("unknown budget key `%s`", key)
		}
	}
	if err := b.Validate(); err != nil {
		return nil, err
	}
	return b, nil
}

func jsonHash(raw json.RawMessage, what string) (hash.Hash32, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err != nil {
		return hash.Hash32{}, werrors.NewParse(
			fmt.Sprintf("capability envelope `%s` must be hex text", what))
	}
	return hash.FromHex(text)
}

// Blob is a signed envelope as stored: canonical {envelope, signature}.
type Blob struct {
	Envelope  *Envelope
	Signature []byte
}

// NewBlob validates the signature length.
func NewBlob(envelope *Envelope, signature []byte) (*Blob, error) {
	if len(signature) != SignatureLen {
		return nil, werrors.Malformedf(
			"capability signature must be %d bytes, got %d", SignatureLen, len(signature))
	}
	return &Blob{Envelope: envelope, Signature: append([]byte(nil), signature...)}, nil
}

// Encode returns canonical WCB bytes {envelope: bytes, signature: bytes}.
func (b *Blob) Encode() ([]byte, error) {
	if len(b.Signature) != SignatureLen {
		return nil, werrors.Malformedf(
			"capability signature must be %d bytes, got %d", SignatureLen, len(b.Signature))
	}
	envelope, err := b.Envelope.CanonicalBytes(envelopeLimits())
	if err != nil {
		return nil, err
	}
	return wcb.Encode(wcb.MapValue(map[string]wcb.Value{
		"envelope":  wcb.BytesValue(envelope),
		"signature": wcb.BytesValue(b.Signature),
	}), envelopeLimits())
}

// ID returns H("weft/capability/blob/v1", encoded blob).
func (b *Blob) ID() (hash.Hash32, error) {
	bytes, err := b.Encode()
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(DomainBlob, bytes), nil
}

// DecodeBlob parses canonical blob bytes.
func DecodeBlob(raw []byte) (*Blob, error) {
	value, err := wcb.Decode(raw, envelopeLimits())
	if err != nil {
		return nil, err
	}
	if value.Kind != wcb.Map {
		return nil, werrors.NewNonCanonical("capability blob must be a WCB map")
	}
	for key := range value.Map {
		if key != "envelope" && key != "signature" {
			return nil, werrors.NonCanonicalf("unknown capability blob key `%s`", key)
		}
	}
	envVal, ok := value.Map["envelope"]
	if !ok || envVal.Kind != wcb.Bytes {
		return nil, werrors.NewNonCanonical("capability blob `envelope` must be bytes")
	}
	sigVal, ok := value.Map["signature"]
	if !ok || sigVal.Kind != wcb.Bytes {
		return nil, werrors.NewNonCanonical("capability blob `signature` must be bytes")
	}
	if len(sigVal.Bytes) != SignatureLen {
		return nil, werrors.NewNonCanonical(
			fmt.Sprintf("capability signature must be %d bytes", SignatureLen))
	}
	envelope, err := EnvelopeFromCanonicalBytes(envVal.Bytes, envelopeLimits())
	if err != nil {
		return nil, err
	}
	blob := &Blob{Envelope: envelope, Signature: append([]byte(nil), sigVal.Bytes...)}
	reencoded, err := blob.Encode()
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(reencoded, raw) {
		return nil, werrors.NewNonCanonical("capability blob bytes are not in canonical form")
	}
	return blob, nil
}

// VerifiedEnvelope is an envelope that passed offline verification.
type VerifiedEnvelope struct {
	Envelope   *Envelope
	EnvelopeID hash.Hash32
	BlobID     hash.Hash32
	Signature  []byte
}

// PublicKeyForSeed derives the Ed25519 public key for a 32-byte seed.
func PublicKeyForSeed(seed []byte) (hash.Hash32, error) {
	if len(seed) != ed25519.SeedSize {
		return hash.Hash32{}, werrors.Malformedf(
			"Ed25519 seed must be %d bytes, got %d", ed25519.SeedSize, len(seed))
	}
	key := ed25519.NewKeyFromSeed(seed)
	public := key.Public().(ed25519.PublicKey)
	return hash.FromSlice(public)
}

// Sign signs a canonical envelope with a 32-byte Ed25519 seed.
func Sign(envelope *Envelope, seed []byte) (*Blob, error) {
	if len(seed) != ed25519.SeedSize {
		return nil, werrors.Malformedf(
			"Ed25519 seed must be %d bytes, got %d", ed25519.SeedSize, len(seed))
	}
	key := ed25519.NewKeyFromSeed(seed)
	public, ok := key.Public().(ed25519.PublicKey)
	if !ok || len(public) != 32 {
		return nil, werrors.NewMalformed("cannot derive Ed25519 public key")
	}
	if !bytes.Equal(public, envelope.Grantor.Bytes()) {
		return nil, werrors.NewNonCanonical(
			"capability envelope grantor does not match the signing key")
	}
	input, err := envelope.SignInput()
	if err != nil {
		return nil, err
	}
	signature := ed25519.Sign(key, input.Bytes())
	return NewBlob(envelope, signature)
}

// SignBytes signs an arbitrary message with a 32-byte Ed25519 seed.
//
// It is the same Ed25519 construction used by Sign, exposed so other Weft
// artifacts (for example a signed receipt bundle) can be verified without a
// second crypto path.
func SignBytes(seed, message []byte) ([]byte, error) {
	if len(seed) != ed25519.SeedSize {
		return nil, werrors.Malformedf(
			"Ed25519 seed must be %d bytes, got %d", ed25519.SeedSize, len(seed))
	}
	key := ed25519.NewKeyFromSeed(seed)
	return ed25519.Sign(key, message), nil
}

// VerifyBytes verifies a detached Ed25519 signature over an arbitrary message.
func VerifyBytes(publicKey hash.Hash32, message, signature []byte) error {
	if len(signature) != SignatureLen {
		return werrors.Malformedf(
			"Ed25519 signature must be %d bytes, got %d", SignatureLen, len(signature))
	}
	if !ed25519.Verify(ed25519.PublicKey(publicKey.Bytes()), message, signature) {
		return werrors.NewCorrupt("Ed25519 signature does not verify")
	}
	return nil
}

// Verify verifies a stored blob offline.
//
// Checks, in normative order: canonical form, Ed25519 signature under grantor,
// optional grantee binding, expiry, and revocation-set membership.
func Verify(
	blobBytes []byte,
	revocations []hash.Hash32,
	nowNS int64,
	expectedGrantee *hash.Hash32,
) (*VerifiedEnvelope, error) {
	blob, err := DecodeBlob(blobBytes)
	if err != nil {
		return nil, envelopeError(err)
	}
	envelope := blob.Envelope
	input, err := envelope.SignInput()
	if err != nil {
		return nil, newRejection(RejectionNonCanonical, err.Error())
	}
	if !ed25519.Verify(ed25519.PublicKey(envelope.Grantor.Bytes()), input.Bytes(), blob.Signature) {
		return nil, newRejection(RejectionBadSignature,
			"capability envelope signature does not verify under grantor")
	}
	if expectedGrantee != nil && *expectedGrantee != envelope.Grantee {
		return nil, newRejection(RejectionWrongGrantee,
			"capability envelope is not bound to the expected grantee")
	}
	if envelope.ExpiresAtNS != nil && nowNS > *envelope.ExpiresAtNS {
		return nil, newRejection(RejectionExpired, "capability envelope has expired")
	}
	for _, id := range revocations {
		if id == envelope.RevocationID {
			return nil, newRejection(RejectionRevoked,
				"capability envelope revocation id is revoked")
		}
	}
	envelopeID, err := envelope.ID()
	if err != nil {
		return nil, newRejection(RejectionNonCanonical, err.Error())
	}
	blobID, err := blob.ID()
	if err != nil {
		return nil, newRejection(RejectionNonCanonical, err.Error())
	}
	return &VerifiedEnvelope{
		Envelope:   envelope,
		EnvelopeID: envelopeID,
		BlobID:     blobID,
		Signature:  append([]byte(nil), blob.Signature...),
	}, nil
}

// VerifyWithoutGrantee verifies without an expected grantee.
func VerifyWithoutGrantee(blobBytes []byte, revocations []hash.Hash32, nowNS int64) (*VerifiedEnvelope, error) {
	return Verify(blobBytes, revocations, nowNS, nil)
}

// ---- local helpers (context's are unexported) --------------------------------

func requiredText(m map[string]wcb.Value, key string) (string, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.Text {
		return "", werrors.NewNonCanonical(
			fmt.Sprintf("capability envelope `%s` must be text", key))
	}
	return v.Str, nil
}

func requiredHash(m map[string]wcb.Value, key string) (hash.Hash32, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.Bytes || len(v.Bytes) != hash.HashLen {
		return hash.Hash32{}, werrors.NewNonCanonical(
			fmt.Sprintf("capability envelope `%s` must be bytes(32)", key))
	}
	h, err := hash.FromSlice(v.Bytes)
	if err != nil {
		return hash.Hash32{}, err
	}
	return h, nil
}

func requiredHashArray(m map[string]wcb.Value, key string) ([]hash.Hash32, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.Array {
		return nil, werrors.NewNonCanonical(
			fmt.Sprintf("capability envelope `%s` must be an array", key))
	}
	out := make([]hash.Hash32, 0, len(v.Array))
	for _, item := range v.Array {
		if item.Kind != wcb.Bytes {
			return nil, werrors.NewNonCanonical(
				fmt.Sprintf("capability envelope `%s` entries must be bytes(32)", key))
		}
		h, err := hash.FromSlice(item.Bytes)
		if err != nil {
			return nil, err
		}
		out = append(out, h)
	}
	return out, nil
}

func requiredTextArray(m map[string]wcb.Value, key string) ([]string, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.Array {
		return nil, werrors.NewNonCanonical(
			fmt.Sprintf("capability envelope `%s` must be an array", key))
	}
	out := make([]string, 0, len(v.Array))
	for _, item := range v.Array {
		if item.Kind != wcb.Text {
			return nil, werrors.NewNonCanonical(
				fmt.Sprintf("capability envelope `%s` entries must be text", key))
		}
		out = append(out, item.Str)
	}
	return out, nil
}

func equalStrings(a, b []string) bool {
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

func equalHashes(a, b []hash.Hash32) bool {
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
