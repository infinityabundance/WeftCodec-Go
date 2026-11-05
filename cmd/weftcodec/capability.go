package main

import (
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/infinityabundance/WeftCodec-Go/capability"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
)

func runCapability(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("capability requires sign, verify, or id")
	}
	switch args[0] {
	case "sign":
		return runCapabilitySign(args[1:])
	case "verify":
		return runCapabilityVerify(args[1:])
	case "id":
		return runCapabilityID(args[1:])
	case "key":
		return runCapabilityKey(args[1:])
	default:
		return fmt.Errorf("unknown capability command %q", args[0])
	}
}

func runCapabilityKey(args []string) error {
	fs := flag.NewFlagSet("capability key", flag.ContinueOnError)
	keyPath := fs.String("key", "", "32-byte Ed25519 seed (hex or raw)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *keyPath == "" {
		return fmt.Errorf("capability key requires --key")
	}
	seed, err := readSeedFile(*keyPath)
	if err != nil {
		return err
	}
	public, err := capability.PublicKeyForSeed(seed)
	if err != nil {
		return err
	}
	fmt.Println(public.Hex())
	return nil
}

func readEnvelopeFile(path string) (*capability.Envelope, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	trimmed := strings.TrimSpace(string(raw))
	if strings.HasPrefix(trimmed, "{") {
		return capability.EnvelopeFromJSON(raw)
	}
	bytes, err := hex.DecodeString(trimmed)
	if err != nil {
		return nil, fmt.Errorf("envelope file is neither JSON nor canonical hex")
	}
	return capability.EnvelopeFromCanonicalBytes(bytes, defaultEnvelopeLimits())
}

func defaultEnvelopeLimits() limits.Limits {
	lim := limits.Default()
	lim.MaxStructuredItems = 65536
	lim.MaxStructuredDepth = 16
	lim.MaxStringBytes = 64 * 1024
	return lim
}

func runCapabilitySign(args []string) error {
	fs := flag.NewFlagSet("capability sign", flag.ContinueOnError)
	envelopePath := fs.String("envelope", "", "envelope JSON or canonical hex file")
	keyPath := fs.String("key", "", "32-byte Ed25519 seed (hex or raw)")
	out := fs.String("out", "-", "output blob path, or - for stdout")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *envelopePath == "" || *keyPath == "" {
		return fmt.Errorf("capability sign requires --envelope and --key")
	}
	envelope, err := readEnvelopeFile(*envelopePath)
	if err != nil {
		return err
	}
	seed, err := readSeedFile(*keyPath)
	if err != nil {
		return err
	}
	blob, err := capability.Sign(envelope, seed)
	if err != nil {
		return err
	}
	raw, err := blob.Encode()
	if err != nil {
		return err
	}
	if *out == "-" {
		if _, err := os.Stdout.Write(raw); err != nil {
			return err
		}
	} else if err := os.WriteFile(*out, raw, 0o644); err != nil {
		return err
	}
	envelopeID, err := envelope.ID()
	if err != nil {
		return err
	}
	blobID, err := blob.ID()
	if err != nil {
		return err
	}
	if *asJSON {
		out := map[string]any{
			"envelope_id":   envelopeID.Hex(),
			"blob_id":       blobID.Hex(),
			"tenant":        envelope.Tenant,
			"grantor":       envelope.Grantor.Hex(),
			"grantee":       envelope.Grantee.Hex(),
			"expires_at_ns": envelope.ExpiresAtNS,
			"revocation_id": envelope.RevocationID.Hex(),
			"signature":     hex.EncodeToString(blob.Signature),
			"blob_bytes":    len(raw),
		}
		encoded, _ := json.MarshalIndent(out, "", "  ")
		fmt.Println(string(encoded))
	} else {
		fmt.Printf("envelope_id=%s\nblob_id=%s\n", envelopeID.Hex(), blobID.Hex())
	}
	return nil
}

func readSeedFile(path string) ([]byte, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) == 32 {
		return raw, nil
	}
	decoded, err := hex.DecodeString(strings.TrimSpace(string(raw)))
	if err != nil || len(decoded) != 32 {
		return nil, fmt.Errorf("key file must be 32 raw bytes or 64 hex digits")
	}
	return decoded, nil
}

func readRevocations(path string) ([]hash.Hash32, error) {
	if path == "" {
		return nil, nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	text := strings.TrimSpace(string(raw))
	var out []hash.Hash32
	if strings.HasPrefix(text, "[") {
		var items []string
		if err := json.Unmarshal([]byte(text), &items); err != nil {
			return nil, fmt.Errorf("bad revocation JSON: %v", err)
		}
		for _, item := range items {
			h, err := hash.FromHex(strings.TrimSpace(item))
			if err != nil {
				return nil, err
			}
			out = append(out, h)
		}
		return out, nil
	}
	for _, item := range strings.FieldsFunc(text, func(r rune) bool {
		return r == ',' || r == ' ' || r == '\n' || r == '\t' || r == '\r'
	}) {
		h, err := hash.FromHex(item)
		if err != nil {
			return nil, err
		}
		out = append(out, h)
	}
	return out, nil
}

func runCapabilityVerify(args []string) error {
	fs := flag.NewFlagSet("capability verify", flag.ContinueOnError)
	blobPath := fs.String("blob", "", "canonical blob file")
	revocationsPath := fs.String("revocations", "", "revocation ids file")
	nowNS := fs.Int64("now-ns", time.Now().UnixNano(), "verification clock (UTC ns)")
	granteeText := fs.String("grantee", "", "require this grantee (64 hex digits)")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *blobPath == "" {
		return fmt.Errorf("capability verify requires --blob")
	}
	raw, err := os.ReadFile(*blobPath)
	if err != nil {
		return err
	}
	revocations, err := readRevocations(*revocationsPath)
	if err != nil {
		return err
	}
	var expected *hash.Hash32
	if *granteeText != "" {
		h, err := hash.FromHex(*granteeText)
		if err != nil {
			return err
		}
		expected = &h
	}
	verified, err := capability.Verify(raw, revocations, *nowNS, expected)
	if err != nil {
		return err
	}
	if *asJSON {
		out := map[string]any{
			"valid":         true,
			"envelope_id":   verified.EnvelopeID.Hex(),
			"blob_id":       verified.BlobID.Hex(),
			"tenant":        verified.Envelope.Tenant,
			"grantor":       verified.Envelope.Grantor.Hex(),
			"grantee":       verified.Envelope.Grantee.Hex(),
			"namespaces":    verified.Envelope.Namespaces,
			"predicates":    verified.Envelope.Predicates,
			"expires_at_ns": verified.Envelope.ExpiresAtNS,
			"revocation_id": verified.Envelope.RevocationID.Hex(),
		}
		encoded, _ := json.MarshalIndent(out, "", "  ")
		fmt.Println(string(encoded))
	} else {
		fmt.Printf("valid=true\nenvelope_id=%s\nblob_id=%s\n",
			verified.EnvelopeID.Hex(), verified.BlobID.Hex())
	}
	return nil
}

func runCapabilityID(args []string) error {
	fs := flag.NewFlagSet("capability id", flag.ContinueOnError)
	envelopePath := fs.String("envelope", "", "envelope JSON or canonical hex file")
	asJSON := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *envelopePath == "" {
		return fmt.Errorf("capability id requires --envelope")
	}
	envelope, err := readEnvelopeFile(*envelopePath)
	if err != nil {
		return err
	}
	id, err := envelope.ID()
	if err != nil {
		return err
	}
	signInput, err := envelope.SignInput()
	if err != nil {
		return err
	}
	if *asJSON {
		canonical, err := envelope.CanonicalBytes(limits.Default())
		if err != nil {
			return err
		}
		out := map[string]any{
			"schema":        capability.EnvelopeSchema,
			"envelope_id":   id.Hex(),
			"sign_input":    signInput.Hex(),
			"canonical_hex": hex.EncodeToString(canonical),
			"tenant":        envelope.Tenant,
			"grantor":       envelope.Grantor.Hex(),
			"grantee":       envelope.Grantee.Hex(),
		}
		encoded, _ := json.MarshalIndent(out, "", "  ")
		fmt.Println(string(encoded))
	} else {
		fmt.Println(id.Hex())
	}
	return nil
}
