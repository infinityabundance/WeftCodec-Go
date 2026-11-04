// Package context implements Weft Context Roots, causal closures, capability
// scopes, and weft:// addressing (spec/WEFT-CONTEXT-ROOT.md,
// spec/WEFT-CAUSAL.md, spec/WEFT-CAPABILITIES.md).
//
// Canonical records are WCB and identities are domain separated:
//
//	ContextRootId   = H("weft/context-root/v1", WCB(context-root record))
//	CausalClosureId = H("weft/causal/v1",       WCB(causal record))
//	CapabilityHash  = H("weft/capability/v1",   WCB(capability record))
//
// This package is pure data. It does not resolve selectors, read snapshots, or
// authorize principals; those are store responsibilities.
package context

import (
	"bytes"
	"fmt"
	"sort"
	"strings"

	"github.com/infinityabundance/WeftCodec-Go/generated"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// Schema strings, policy names, and text limits are generated from
// spec/schema/identity-v1.schema. Do not edit these values here.
const (
	ContextRootSchema = generated.ContextRootSchema
	CausalSchema      = generated.CausalSchema
	CapabilitySchema  = generated.CapabilitySchema
)

// OrderingSelectorThenObjectID is the only v1 ordering/tie-break policy.
const OrderingSelectorThenObjectID = generated.OrderingSelectorThenObjectID

// DefaultResolver is the default resolver/policy version.
const DefaultResolver = generated.DefaultResolver

// URIPrefix is the weft:// scheme prefix.
const URIPrefix = generated.URIPrefix

const (
	maxNamespaceBytes = generated.MaxNamespaceBytes
	maxResolverBytes  = generated.MaxResolverBytes
)

func validateNamespace(namespace string) error {
	b := []byte(namespace)
	if len(b) == 0 || len(b) > maxNamespaceBytes {
		return werrors.NewNonCanonical("namespace must be 1..=128 bytes")
	}
	if namespace == "." || namespace == ".." {
		return werrors.NewNonCanonical("namespace `.` or `..` is not allowed")
	}
	first := b[0]
	if !isAlnum(first) && first != '_' {
		return werrors.NewNonCanonical(
			"namespace must begin with an ASCII letter, digit, or `_`")
	}
	for _, c := range b {
		if !isAlnum(c) && c != '_' && c != '.' && c != '-' {
			return werrors.NewNonCanonical(
				"namespace may contain only ASCII letters, digits, `_`, `.`, and `-`")
		}
	}
	return nil
}

func validateResolver(resolver string) error {
	b := []byte(resolver)
	if len(b) == 0 || len(b) > maxResolverBytes {
		return werrors.NewNonCanonical("resolver must be 1..=64 bytes")
	}
	if !isAlpha(b[0]) {
		return werrors.NewNonCanonical("resolver must begin with an ASCII letter")
	}
	for _, c := range b {
		if !isAlnum(c) && c != '.' && c != '-' && c != '_' {
			return werrors.NewNonCanonical(
				"resolver may contain only ASCII letters, digits, `.`, `-`, and `_`")
		}
	}
	return nil
}

func validatePredicate(predicate string) error {
	if len(predicate) == 0 || len(predicate) > 256 {
		return werrors.NewNonCanonical("capability predicate must be 1..=256 bytes")
	}
	for i := 0; i < len(predicate); i++ {
		c := predicate[i]
		if !isAlnum(c) && c != '_' && c != '.' && c != '-' && c != '/' {
			return werrors.NewNonCanonical(
				"capability predicate contains a disallowed byte")
		}
	}
	return nil
}

func isAlnum(c byte) bool {
	return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

func isAlpha(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

func hashLess(a, b hash.Hash32) bool { return bytes.Compare(a[:], b[:]) < 0 }

func bytesEqual(a, b []byte) bool { return bytes.Equal(a, b) }

func sortUniqueHashes(hashes []hash.Hash32, what string) ([]hash.Hash32, error) {
	for _, h := range hashes {
		if h == hash.Zero {
			return nil, werrors.NewNonCanonical(
				what + " must not contain the all-zero hash")
		}
	}
	out := append([]hash.Hash32(nil), hashes...)
	sort.Slice(out, func(i, j int) bool { return hashLess(out[i], out[j]) })
	unique := out[:0]
	for i, h := range out {
		if i == 0 || h != out[i-1] {
			unique = append(unique, h)
		}
	}
	return unique, nil
}

func checkAscendingUnique(hashes []hash.Hash32, what string) error {
	for i := 1; i < len(hashes); i++ {
		if bytes.Compare(hashes[i-1][:], hashes[i][:]) >= 0 {
			return werrors.NewNonCanonical(
				what + " must be strictly ascending and unique")
		}
	}
	for _, h := range hashes {
		if h == hash.Zero {
			return werrors.NewNonCanonical(
				what + " must not contain the all-zero hash")
		}
	}
	return nil
}

func containsHash(hashes []hash.Hash32, needle hash.Hash32) bool {
	for _, h := range hashes {
		if h == needle {
			return true
		}
	}
	return false
}

func strictlySortedStrings(values []string) bool {
	for i := 1; i < len(values); i++ {
		if values[i-1] >= values[i] {
			return false
		}
	}
	return true
}

func checkCausalEntries(count int, lim limits.Limits, what string) error {
	if uint64(count) > lim.MaxCausalEntries {
		return werrors.LimitExceededf(
			"%s has %d entries, limit %d", what, count, lim.MaxCausalEntries)
	}
	return nil
}

func checkCapabilityEntries(count int, lim limits.Limits, what string) error {
	if uint64(count) > lim.MaxCapabilityEntries {
		return werrors.LimitExceededf(
			"%s has %d entries, limit %d", what, count, lim.MaxCapabilityEntries)
	}
	return nil
}

func hashesToValue(hashes []hash.Hash32) wcb.Value {
	items := make([]wcb.Value, 0, len(hashes))
	for _, h := range hashes {
		items = append(items, wcb.BytesValue(hashBytes(h)))
	}
	return wcb.ArrayValue(items)
}

func hashBytes(h hash.Hash32) []byte {
	out := make([]byte, hash.HashLen)
	copy(out, h[:])
	return out
}

func hashFromValue(v wcb.Value, what string) (hash.Hash32, error) {
	if v.Kind != wcb.Bytes || len(v.Bytes) != hash.HashLen {
		return hash.Hash32{}, werrors.NewNonCanonical(what + " must be bytes(32)")
	}
	return hash.FromSlice(v.Bytes)
}

func requiredText(m map[string]wcb.Value, key string) (string, error) {
	v, ok := m[key]
	if !ok {
		return "", werrors.NewNonCanonical("missing `" + key + "`")
	}
	if v.Kind != wcb.Text {
		return "", werrors.NewNonCanonical("`" + key + "` must be text")
	}
	return v.Str, nil
}

func requiredBytes(m map[string]wcb.Value, key string) ([]byte, error) {
	v, ok := m[key]
	if !ok {
		return nil, werrors.NewNonCanonical("missing `" + key + "`")
	}
	if v.Kind != wcb.Bytes {
		return nil, werrors.NewNonCanonical("`" + key + "` must be bytes")
	}
	return append([]byte(nil), v.Bytes...), nil
}

func requiredHash(m map[string]wcb.Value, key, what string) (hash.Hash32, error) {
	v, ok := m[key]
	if !ok {
		return hash.Hash32{}, werrors.NewNonCanonical("missing `" + key + "`")
	}
	h, err := hashFromValue(v, what)
	if err != nil {
		return hash.Hash32{}, err
	}
	if h == hash.Zero {
		return hash.Hash32{}, werrors.NewNonCanonical(what + " must not be all zero")
	}
	return h, nil
}

func requiredHashes(m map[string]wcb.Value, key string, lim limits.Limits, what string) ([]hash.Hash32, error) {
	v, ok := m[key]
	if !ok {
		return nil, werrors.NewNonCanonical("missing `" + key + "`")
	}
	if v.Kind != wcb.Array {
		return nil, werrors.NewNonCanonical("`" + key + "` must be an array of bytes(32)")
	}
	if err := checkCausalEntries(len(v.Array), lim, what); err != nil {
		return nil, err
	}
	out := make([]hash.Hash32, 0, len(v.Array))
	for _, item := range v.Array {
		h, err := hashFromValue(item, what)
		if err != nil {
			return nil, err
		}
		if h == hash.Zero {
			return nil, werrors.NewNonCanonical(
				what + " must not contain the all-zero hash")
		}
		out = append(out, h)
	}
	return out, nil
}

func requiredTextArray(m map[string]wcb.Value, key, what string) ([]string, error) {
	v, ok := m[key]
	if !ok {
		return nil, werrors.NewNonCanonical("missing `" + key + "`")
	}
	return textArray(v, what)
}

func textArray(v wcb.Value, what string) ([]string, error) {
	if v.Kind != wcb.Array {
		return nil, werrors.NewNonCanonical(what + " must be an array")
	}
	out := make([]string, 0, len(v.Array))
	for _, item := range v.Array {
		if item.Kind != wcb.Text {
			return nil, werrors.NewNonCanonical(what + " must contain only text")
		}
		out = append(out, item.Str)
	}
	return out, nil
}

func hashesToHexList(hashes []hash.Hash32) string {
	parts := make([]string, 0, len(hashes))
	for _, h := range hashes {
		parts = append(parts, h.Hex())
	}
	return strings.Join(parts, ",")
}

func parseLowerHex32(text, what string) (hash.Hash32, error) {
	if len(text) != hash.HashLen*2 {
		return hash.Hash32{}, werrors.NewParse(
			fmt.Sprintf("%s must be exactly 64 lowercase hex digits", what))
	}
	for i := 0; i < len(text); i++ {
		c := text[i]
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			return hash.Hash32{}, werrors.NewParse(
				fmt.Sprintf("%s must be lowercase hexadecimal", what))
		}
	}
	h, err := hash.FromHex(text)
	if err != nil {
		return hash.Hash32{}, werrors.NewParse(what + ": " + err.Error())
	}
	if h == hash.Zero {
		return hash.Hash32{}, werrors.NewParse(what + " must not be all zero")
	}
	return h, nil
}

func isUnreserved(c byte) bool {
	return isAlnum(c) || c == '-' || c == '.' || c == '_' || c == '~'
}

// PercentEncode encodes selector text canonically.
func PercentEncode(text string) string {
	var b strings.Builder
	const hexUpper = "0123456789ABCDEF"
	for i := 0; i < len(text); i++ {
		c := text[i]
		if isUnreserved(c) {
			b.WriteByte(c)
			continue
		}
		b.WriteByte('%')
		b.WriteByte(hexUpper[c>>4])
		b.WriteByte(hexUpper[c&0x0f])
	}
	return b.String()
}

// PercentDecode decodes selector text strictly.
func PercentDecode(text string) (string, error) {
	out := make([]byte, 0, len(text))
	for i := 0; i < len(text); {
		c := text[i]
		if c == '%' {
			if i+2 >= len(text) {
				return "", werrors.NewParse("truncated percent escape")
			}
			high, ok := upperHex(text[i+1])
			if !ok {
				return "", werrors.NewParse(
					"percent escape must use uppercase hex digits")
			}
			low, ok := upperHex(text[i+2])
			if !ok {
				return "", werrors.NewParse(
					"percent escape must use uppercase hex digits")
			}
			decoded := high<<4 | low
			if isUnreserved(decoded) {
				return "", werrors.NewParse(
					"percent escape encodes an unreserved character (non-canonical)")
			}
			out = append(out, decoded)
			i += 3
			continue
		}
		if !isUnreserved(c) {
			return "", werrors.Newf(werrors.Parse,
				"raw byte 0x%02X in selector text must be percent-encoded", c)
		}
		out = append(out, c)
		i++
	}
	return string(out), nil
}

func upperHex(c byte) (byte, bool) {
	switch {
	case c >= '0' && c <= '9':
		return c - '0', true
	case c >= 'A' && c <= 'F':
		return c - 'A' + 10, true
	default:
		return 0, false
	}
}
