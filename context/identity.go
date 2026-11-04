package context

import (
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// IdentityMode is the content/causal identity mode (WEFT-CAUSAL §2).
type IdentityMode uint8

const (
	// BytesOnly is the v1 default: VersionId depends on content/version bytes
	// only. Existing v1 identities are frozen under this mode.
	BytesOnly IdentityMode = iota
	// CausalExact includes the causal closure hash in VersionId.
	CausalExact
)

// String returns the stored text form.
func (m IdentityMode) String() string {
	switch m {
	case BytesOnly:
		return "bytes_only"
	case CausalExact:
		return "causal_exact"
	default:
		return "unknown"
	}
}

// ParseIdentityMode parses the stored text form. Unknown names are
// UnsupportedVersion; they are never silently treated as the default.
func ParseIdentityMode(text string) (IdentityMode, error) {
	switch text {
	case "bytes_only":
		return BytesOnly, nil
	case "causal_exact":
		return CausalExact, nil
	default:
		return 0, werrors.UnsupportedVersionf("unknown identity mode `%s`", text)
	}
}
