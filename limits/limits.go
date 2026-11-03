// Package limits holds the WeftCodec v1 resource bounds
// (WEFTCODEC-FORMAT §10, WEFT-SELECTORS §1).
//
// Every bound is enforced before an allocation that depends on it. All fields
// may be lowered by callers; the defaults are the normative v1 defaults.
package limits

import (
	"math"

	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

const (
	DefaultMaxFrameBytes        uint64 = 1 << 30
	DefaultMaxDecodedBytes      uint64 = 4 << 30
	DefaultMaxDependencyDepth   uint64 = 8
	DefaultMaxDependencyCount   uint64 = 64
	DefaultMaxTotalFetchedBytes uint64 = 8 << 30
	DefaultMaxChunks            uint64 = 1_048_576
	DefaultMaxDictionaryBytes   uint64 = 64 << 20
	DefaultMaxStructuredDepth   uint64 = 64
	DefaultMaxStructuredItems   uint64 = 1_000_000
	DefaultMaxStringBytes       uint64 = 256 << 20
	DefaultMaxDeltaOps          uint64 = 16_777_216
	DefaultMaxSelectorBytes     uint64 = 64 << 10
	DefaultMaxSelectorDepth     uint64 = 64
	DefaultMaxFieldSegments     uint64 = 16
	DefaultMaxSelectorOperands  uint64 = 4_096
	DefaultMaxArchiveEntries    uint64 = 1_000_000
)

// Limits is a value type so it can be passed and copied cheaply.
type Limits struct {
	MaxFrameBytes        uint64
	MaxDecodedBytes      uint64
	MaxDependencyDepth   uint64
	MaxDependencyCount   uint64
	MaxTotalFetchedBytes uint64
	MaxChunks            uint64
	MaxDictionaryBytes   uint64
	MaxStructuredDepth   uint64
	MaxStructuredItems   uint64
	MaxStringBytes       uint64
	MaxDeltaOps          uint64
	MaxSelectorBytes     uint64
	MaxSelectorDepth     uint64
	MaxFieldSegments     uint64
	MaxSelectorOperands  uint64
	MaxArchiveEntries    uint64
}

// Default returns the normative v1 defaults.
func Default() Limits {
	return Limits{
		MaxFrameBytes:        DefaultMaxFrameBytes,
		MaxDecodedBytes:      DefaultMaxDecodedBytes,
		MaxDependencyDepth:   DefaultMaxDependencyDepth,
		MaxDependencyCount:   DefaultMaxDependencyCount,
		MaxTotalFetchedBytes: DefaultMaxTotalFetchedBytes,
		MaxChunks:            DefaultMaxChunks,
		MaxDictionaryBytes:   DefaultMaxDictionaryBytes,
		MaxStructuredDepth:   DefaultMaxStructuredDepth,
		MaxStructuredItems:   DefaultMaxStructuredItems,
		MaxStringBytes:       DefaultMaxStringBytes,
		MaxDeltaOps:          DefaultMaxDeltaOps,
		MaxSelectorBytes:     DefaultMaxSelectorBytes,
		MaxSelectorDepth:     DefaultMaxSelectorDepth,
		MaxFieldSegments:     DefaultMaxFieldSegments,
		MaxSelectorOperands:  DefaultMaxSelectorOperands,
		MaxArchiveEntries:    DefaultMaxArchiveEntries,
	}
}

// CheckLen reports LimitExceeded when n > max.
func CheckLen(n, max uint64, what string) error {
	if n > max {
		return werrors.LimitExceededf("%s length %d exceeds limit %d", what, n, max)
	}
	return nil
}

// ToInt converts a u64 length to int, reporting LimitExceeded when it cannot be
// represented.
func ToInt(v uint64, what string) (int, error) {
	if v > uint64(math.MaxInt) {
		return 0, werrors.LimitExceededf("%s length %d does not fit int", what, v)
	}
	return int(v), nil
}

// AddChecked adds two unsigned lengths, reporting LimitExceeded on overflow.
func AddChecked(a, b uint64, what string) (uint64, error) {
	if a > math.MaxUint64-b {
		return 0, werrors.LimitExceededf("%s length overflow", what)
	}
	return a + b, nil
}

// MulChecked multiplies two unsigned lengths, reporting LimitExceeded on
// overflow.
func MulChecked(a, b uint64, what string) (uint64, error) {
	if a != 0 && b > math.MaxUint64/a {
		return 0, werrors.LimitExceededf("%s length overflow", what)
	}
	return a * b, nil
}
