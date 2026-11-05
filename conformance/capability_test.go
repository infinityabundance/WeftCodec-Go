package conformance_test

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/capability"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
)

// capabilityVector replays the frozen authenticated-envelope vectors. The
// vectors are produced by the Rust generator and copied byte-for-byte here;
// the Go implementation must agree on canonical bytes, EnvelopeId, sign input,
// blob id, and every rejection name.
type capabilityVector struct {
	Name         string          `json:"name"`
	EnvelopeJSON json.RawMessage `json:"envelope_json"`
	CanonicalHex string          `json:"canonical_hex"`
	EnvelopeID   string          `json:"envelope_id"`
	SignInput    string          `json:"sign_input"`
	BlobID       string          `json:"blob_id"`
	BlobHex      string          `json:"blob_hex"`
	Grantee      string          `json:"grantee"`
	NowNS        int64           `json:"now_ns"`
	Revocations  []string        `json:"revocations"`
	ExpectError  string          `json:"expect_error"`
}

func TestCapabilityVectors(t *testing.T) {
	dir := filepath.Join("v1", "capability")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("cannot read %s: %v", dir, err)
	}
	var names []string
	for _, entry := range entries {
		if !entry.IsDir() && filepath.Ext(entry.Name()) == ".json" {
			names = append(names, entry.Name())
		}
	}
	sort.Strings(names)
	if len(names) < 6 {
		t.Fatalf("expected the authenticated capability vector set, found %d", len(names))
	}

	for _, name := range names {
		path := filepath.Join(dir, name)
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		var vector capabilityVector
		if err := json.Unmarshal(raw, &vector); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		blob, err := hex.DecodeString(vector.BlobHex)
		if err != nil {
			t.Fatalf("%s: blob hex: %v", path, err)
		}
		var revocations []hash.Hash32
		for _, text := range vector.Revocations {
			id, err := hash.FromHex(text)
			if err != nil {
				t.Fatalf("%s: revocation hex: %v", path, err)
			}
			revocations = append(revocations, id)
		}
		var grantee *hash.Hash32
		if vector.Grantee != "" && vector.ExpectError != "" {
			id, err := hash.FromHex(vector.Grantee)
			if err != nil {
				t.Fatalf("%s: grantee hex: %v", path, err)
			}
			grantee = &id
		}

		if vector.ExpectError != "" {
			_, err := capability.Verify(blob, revocations, vector.NowNS, grantee)
			if err == nil {
				t.Fatalf("%s: expected %s, verification succeeded", vector.Name, vector.ExpectError)
			}
			envelopeErr, ok := err.(*capability.EnvelopeError)
			if !ok {
				t.Fatalf("%s: non-envelope error %T: %v", vector.Name, err, err)
			}
			if string(envelopeErr.Rejection) != vector.ExpectError {
				t.Fatalf("%s: expected %s, got %s", vector.Name, vector.ExpectError, envelopeErr.Rejection)
			}
			continue
		}

		envelope, err := capability.EnvelopeFromJSON(vector.EnvelopeJSON)
		if err != nil {
			t.Fatalf("%s: envelope JSON: %v", vector.Name, err)
		}
		canonical, err := envelope.CanonicalBytes(limits.Default())
		if err != nil {
			t.Fatalf("%s: canonical bytes: %v", vector.Name, err)
		}
		if hex.EncodeToString(canonical) != vector.CanonicalHex {
			t.Fatalf("%s: canonical bytes differ", vector.Name)
		}
		if id, err := envelope.ID(); err != nil || id.Hex() != vector.EnvelopeID {
			t.Fatalf("%s: envelope id = %v (%v), want %s", vector.Name, id.Hex(), err, vector.EnvelopeID)
		}
		if input, err := envelope.SignInput(); err != nil || input.Hex() != vector.SignInput {
			t.Fatalf("%s: sign input = %v (%v), want %s", vector.Name, input.Hex(), err, vector.SignInput)
		}
		verified, err := capability.Verify(blob, nil, vector.NowNS, nil)
		if err != nil {
			t.Fatalf("%s: verify: %v", vector.Name, err)
		}
		if verified.EnvelopeID.Hex() != vector.EnvelopeID {
			t.Fatalf("%s: verified envelope id differs", vector.Name)
		}
		if verified.BlobID.Hex() != vector.BlobID {
			t.Fatalf("%s: blob id = %s, want %s", vector.Name, verified.BlobID.Hex(), vector.BlobID)
		}
		stored, err := capability.DecodeBlob(blob)
		if err != nil {
			t.Fatalf("%s: blob decode: %v", vector.Name, err)
		}
		encoded, err := stored.Encode()
		if err != nil {
			t.Fatalf("%s: blob encode: %v", vector.Name, err)
		}
		if hex.EncodeToString(encoded) != vector.BlobHex {
			t.Fatalf("%s: canonical blob bytes differ", vector.Name)
		}
	}
}
