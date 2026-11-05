// Package hash implements the single v1 hash construction:
//
//	H(domain, data) = BLAKE3-256( u32le(len(domain)) || domain || data )
//
// Domain strings are ASCII, contain no NUL, and are versioned. The length
// prefix makes domain/data splits unambiguous.
package hash

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"strings"

	"lukechampine.com/blake3"

	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// HashLen is the v1 digest length.
const HashLen = 32

// Domain strings (WEFTCODEC-FORMAT §1.1).
const (
	DomainChunk       = "weft/chunk/v1"
	DomainManifest    = "weft/manifest/v1"
	DomainStructured  = "weft/structured/v1"
	DomainSelector    = "weft/selector/v1"
	DomainSnapshot    = "weft/snapshot/v1"
	DomainCommit      = "weft/store/commit/v1"
	DomainRelation    = "weft/relation/v1"
	DomainVersion     = "weft/version/v1"
	DomainObject      = "weft/object/v1"
	DomainContextRoot = "weft/context-root/v1"
	DomainCausal      = "weft/causal/v1"
	DomainCapability  = "weft/capability/v1"
	// Tensor/Token payload bindings (WEFT-AI-CODEC §2-§3).
	DomainTensorPayload = "weft/tensor/payload/v1"
	DomainTokensPayload = "weft/tokens/payload/v1"
	// Quantized-tensor payload binding (WEFT-AI-CODEC §4).
	DomainQuantizedTensorPayload = "weft/quantized-tensor/payload/v1"
	// Float-tensor stored-payload binding (WEFT-AI-CODEC §5).
	DomainFloatTensorPayload = "weft/float-tensor/payload/v1"
	// Sharded quantized tensor stored-payload binding (WEFT-AI-CODEC §11).
	DomainTensorShardedPayload = "weft/tensor-sharded/payload/v1"

	// DomainTensorTile is the independently decodable tile identity domain
	// (WEFT-AI-CODEC §16).
	DomainTensorTile = "weft/tensor/tile/v1"
)

// Hash32 is a 32-byte Weft digest. It is comparable and ordered
// lexicographically by its bytes, matching the Rust `[u8; 32]` ordering.
type Hash32 [HashLen]byte

// Zero is the all-zero digest placeholder.
var Zero Hash32

// FromBytes wraps raw bytes.
func FromBytes(b [HashLen]byte) Hash32 { return Hash32(b) }

// FromSlice copies exactly 32 bytes from b.
func FromSlice(b []byte) (Hash32, error) {
	var h Hash32
	if len(b) != HashLen {
		return h, werrors.Malformedf("hash must be %d bytes, got %d", HashLen, len(b))
	}
	copy(h[:], b)
	return h, nil
}

// FromHex parses exactly 64 hex digits (either case).
func FromHex(s string) (Hash32, error) {
	var h Hash32
	if len(s) != HashLen*2 {
		return h, werrors.Malformedf("hash hex must be %d characters, got %d", HashLen*2, len(s))
	}
	if _, err := hex.Decode(h[:], []byte(s)); err != nil {
		return h, werrors.Malformedf("invalid hash hex: %v", err)
	}
	return h, nil
}

// Bytes returns the raw digest bytes (a copy).
func (h Hash32) Bytes() []byte {
	out := make([]byte, HashLen)
	copy(out, h[:])
	return out
}

// Hex returns lowercase hexadecimal.
func (h Hash32) Hex() string { return hex.EncodeToString(h[:]) }

// String implements fmt.Stringer.
func (h Hash32) String() string { return h.Hex() }

// Less orders digests by unsigned byte value.
func (h Hash32) Less(other Hash32) bool {
	for i := 0; i < HashLen; i++ {
		if h[i] != other[i] {
			return h[i] < other[i]
		}
	}
	return false
}

// Domain hashes data under a domain.
func Domain(domain string, data []byte) Hash32 {
	hasher := blake3.New(HashLen, nil)
	writePrefix(hasher, domain)
	_, _ = hasher.Write(data)
	var out Hash32
	sum := hasher.Sum(out[:0])
	copy(out[:], sum)
	return out
}

// DomainReader hashes a stream under a domain without buffering the input.
func DomainReader(domain string, r io.Reader) (Hash32, error) {
	hasher := blake3.New(HashLen, nil)
	writePrefix(hasher, domain)
	if _, err := io.Copy(hasher, r); err != nil {
		return Hash32{}, werrors.NewIo(fmt.Sprintf("cannot hash stream: %v", err))
	}
	var out Hash32
	sum := hasher.Sum(out[:0])
	copy(out[:], sum)
	return out, nil
}

// TryDomain validates the domain and hashes data.
func TryDomain(domain string, data []byte) (Hash32, error) {
	if !isASCII(domain) || strings.ContainsRune(domain, 0) {
		return Hash32{}, werrors.NewMalformed("hash domain must be ASCII without NUL")
	}
	if uint64(len(domain)) > uint64(^uint32(0)) {
		return Hash32{}, werrors.NewLimitExceeded("hash domain exceeds u32 length")
	}
	return Domain(domain, data), nil
}

func writePrefix(w io.Writer, domain string) {
	var prefix [4]byte
	length := uint64(len(domain))
	if length > uint64(^uint32(0)) {
		length = uint64(^uint32(0))
	}
	binary.LittleEndian.PutUint32(prefix[:], uint32(length))
	_, _ = w.Write(prefix[:])
	_, _ = io.WriteString(w, domain)
}

func isASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] > 0x7f {
			return false
		}
	}
	return true
}

// Chunk is a convenience for the chunk/plaintext domain.
func Chunk(data []byte) Hash32 { return Domain(DomainChunk, data) }

// Selector is a convenience for the selector domain.
func Selector(data []byte) Hash32 { return Domain(DomainSelector, data) }
