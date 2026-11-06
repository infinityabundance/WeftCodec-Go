package capability

import (
	"bytes"
	"crypto/ed25519"
	"encoding/json"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/context"
	"github.com/infinityabundance/WeftCodec-Go/hash"
)

func seedKey(seedByte byte) (hash.Hash32, []byte) {
	seed := bytes.Repeat([]byte{seedByte}, ed25519.SeedSize)
	key := ed25519.NewKeyFromSeed(seed)
	pub := key.Public().(ed25519.PublicKey)
	h, _ := hash.FromSlice(pub)
	return h, seed
}

func hashOf(byteValue byte) hash.Hash32 {
	return hash.Hash32(bytes.Repeat([]byte{byteValue}, hash.HashLen))
}

func sample(t *testing.T) *Envelope {
	t.Helper()
	grantor, _ := seedKey(7)
	expires := int64(2_000_000_000_000_000_000)
	budget := &context.Budget{
		MaxResults: u64ptr(10),
		MaxDepth:   u64ptr(2),
		MaxWork:    u64ptr(100),
	}
	e, err := New(
		grantor,
		hashOf(9),
		"tenant-a",
		[]hash.Hash32{hashOf(0x11)},
		[]string{"default"},
		[]string{"derived_from"},
		&expires,
		budget,
		hashOf(0x22),
		1_700_000_000_000_000_000,
		hashOf(0x33),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return e
}

func u64ptr(v uint64) *uint64 { return &v }

func TestCanonicalRoundTrip(t *testing.T) {
	e := sample(t)
	lim := envelopeLimits()
	b, err := e.CanonicalBytes(lim)
	if err != nil {
		t.Fatalf("canonical: %v", err)
	}
	decoded, err := EnvelopeFromCanonicalBytes(b, lim)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if decoded.Grantor != e.Grantor || decoded.Tenant != e.Tenant {
		t.Fatalf("decode mismatch: %+v", decoded)
	}
	id, err := e.ID()
	if err != nil {
		t.Fatalf("id: %v", err)
	}
	if id != hash.Domain(DomainEnvelope, b) {
		t.Fatalf("envelope id domain mismatch")
	}
	input, err := e.SignInput()
	if err != nil {
		t.Fatalf("sign input: %v", err)
	}
	if input != hash.Domain(DomainSign, b) {
		t.Fatalf("sign input domain mismatch")
	}
}

func TestSignVerifyRoundTrip(t *testing.T) {
	e := sample(t)
	_, seed := seedKey(7)
	blob, err := Sign(e, seed)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	raw, err := blob.Encode()
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	verified, err := Verify(raw, nil, 1_700_000_000_000_000_000, &e.Grantee)
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if verified.EnvelopeID != mustID(t, e) {
		t.Fatalf("envelope id mismatch")
	}
	if verified.BlobID != mustBlobID(t, blob) {
		t.Fatalf("blob id mismatch")
	}
}

func mustID(t *testing.T, e *Envelope) hash.Hash32 {
	t.Helper()
	id, err := e.ID()
	if err != nil {
		t.Fatalf("id: %v", err)
	}
	return id
}

func mustBlobID(t *testing.T, b *Blob) hash.Hash32 {
	t.Helper()
	id, err := b.ID()
	if err != nil {
		t.Fatalf("blob id: %v", err)
	}
	return id
}

func TestRejectMatrix(t *testing.T) {
	e := sample(t)
	_, seed := seedKey(7)
	blob, err := Sign(e, seed)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	raw, err := blob.Encode()
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	bad := *blob
	bad.Signature = append([]byte(nil), blob.Signature...)
	bad.Signature[0] ^= 0xff
	if _, err := Verify(bad.EncodeOrPanic(t), nil, 0, nil); rejectionOf(t, err) != RejectionBadSignature {
		t.Fatalf("bad signature rejection = %v", rejectionOf(t, err))
	}

	wrong := hashOf(0xAA)
	if _, err := Verify(raw, nil, 0, &wrong); rejectionOf(t, err) != RejectionWrongGrantee {
		t.Fatalf("wrong grantee rejection = %v", rejectionOf(t, err))
	}

	if _, err := Verify(raw, nil, 2_000_000_000_000_000_001, nil); rejectionOf(t, err) != RejectionExpired {
		t.Fatalf("expired rejection = %v", rejectionOf(t, err))
	}

	if _, err := Verify(raw, []hash.Hash32{e.RevocationID}, 0, nil); rejectionOf(t, err) != RejectionRevoked {
		t.Fatalf("revoked rejection = %v", rejectionOf(t, err))
	}

	nonCanonical := append(append([]byte(nil), raw...), 0)
	if _, err := Verify(nonCanonical, nil, 0, nil); rejectionOf(t, err) != RejectionNonCanonical {
		t.Fatalf("non-canonical rejection = %v", rejectionOf(t, err))
	}
}

func (b *Blob) EncodeOrPanic(t *testing.T) []byte {
	t.Helper()
	raw, err := b.Encode()
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	return raw
}

func rejectionOf(t *testing.T, err error) Rejection {
	t.Helper()
	if err == nil {
		t.Fatalf("expected an error")
	}
	typed, ok := err.(*EnvelopeError)
	if !ok {
		t.Fatalf("error is not *EnvelopeError: %v", err)
	}
	return typed.Rejection
}

func TestRejectUnsortedGrants(t *testing.T) {
	grantor, _ := seedKey(7)
	_, err := New(
		grantor,
		hashOf(9),
		"tenant-a",
		nil,
		[]string{"b", "a"},
		nil,
		nil,
		nil,
		hashOf(0x22),
		0,
		hashOf(0x33),
	)
	if err == nil {
		t.Fatalf("expected non-canonical error")
	}
}

func TestJSONRoundTrip(t *testing.T) {
	e := sample(t)
	raw, err := e.CanonicalBytes(envelopeLimits())
	if err != nil {
		t.Fatalf("canonical: %v", err)
	}
	// Build JSON from the same fields and decode it back.
	j := e.ToJSON()
	encoded, err := json.Marshal(j)
	if err != nil {
		t.Fatalf("json: %v", err)
	}
	parsed, err := EnvelopeFromJSON(encoded)
	if err != nil {
		t.Fatalf("from json: %v", err)
	}
	again, err := parsed.CanonicalBytes(envelopeLimits())
	if err != nil {
		t.Fatalf("canonical again: %v", err)
	}
	if !bytes.Equal(raw, again) {
		t.Fatalf("json round trip changed canonical bytes")
	}
}

func TestSignVerifyBytes(t *testing.T) {
	seed := bytes.Repeat([]byte{0x11}, ed25519.SeedSize)
	message := bytes.Repeat([]byte{0xab}, 32)
	signature, err := SignBytes(seed, message)
	if err != nil {
		t.Fatalf("sign bytes: %v", err)
	}
	if len(signature) != SignatureLen {
		t.Fatalf("signature length %d", len(signature))
	}
	public, err := PublicKeyForSeed(seed)
	if err != nil {
		t.Fatalf("public key: %v", err)
	}
	if err := VerifyBytes(public, message, signature); err != nil {
		t.Fatalf("verify bytes: %v", err)
	}
	tampered := append([]byte(nil), message...)
	tampered[0] ^= 0xff
	if err := VerifyBytes(public, tampered, signature); err == nil {
		t.Fatalf("expected tampered message to fail")
	}
	otherSeed := bytes.Repeat([]byte{0x22}, ed25519.SeedSize)
	other, err := PublicKeyForSeed(otherSeed)
	if err != nil {
		t.Fatalf("other public key: %v", err)
	}
	if err := VerifyBytes(other, message, signature); err == nil {
		t.Fatalf("expected wrong key to fail")
	}
}
