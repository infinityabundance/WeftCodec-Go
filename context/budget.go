package context

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// Budget is a resource budget (WEFT-CONTEXT-ROOT §4). At least one bound must
// be present; every stated bound is a ceiling.
type Budget struct {
	MaxResults *uint64
	MaxBytes   *uint64
	MaxDepth   *uint64
	MaxWork    *uint64
}

// IsEmpty reports whether no bound is present.
func (b *Budget) IsEmpty() bool {
	return b == nil || (b.MaxResults == nil && b.MaxBytes == nil && b.MaxDepth == nil && b.MaxWork == nil)
}

// Validate requires a non-empty budget with every stated bound ≥ 1.
func (b *Budget) Validate() error {
	if b.IsEmpty() {
		return werrors.NewNonCanonical("budget must state at least one bound")
	}
	for _, item := range []struct {
		name  string
		value *uint64
	}{
		{"max_results", b.MaxResults},
		{"max_bytes", b.MaxBytes},
		{"max_depth", b.MaxDepth},
		{"max_work", b.MaxWork},
	} {
		if item.value != nil && *item.value == 0 {
			return werrors.NonCanonicalf("budget `%s` must be at least 1", item.name)
		}
	}
	return nil
}

// ParseBudget parses `max_results=..,max_bytes=..,max_depth=..,max_work=..`.
func ParseBudget(spec string) (*Budget, error) {
	if spec == "" {
		return nil, werrors.NewParse("budget specification must not be empty")
	}
	b := &Budget{}
	seen := map[string]bool{}
	for _, entry := range strings.Split(spec, ",") {
		if entry == "" {
			return nil, werrors.NewParse("empty budget entry")
		}
		name, value, ok := strings.Cut(entry, "=")
		if !ok {
			return nil, werrors.NewParse(fmt.Sprintf("budget entry `%s` is missing `=`", entry))
		}
		if value == "" {
			return nil, werrors.NewParse(
				fmt.Sprintf("budget value for `%s` must be decimal digits", name))
		}
		for i := 0; i < len(value); i++ {
			if value[i] < '0' || value[i] > '9' {
				return nil, werrors.NewParse(
					fmt.Sprintf("budget value for `%s` must be decimal digits", name))
			}
		}
		if len(value) > 1 && value[0] == '0' {
			return nil, werrors.NewParse(
				fmt.Sprintf("budget value for `%s` is not canonical decimal", name))
		}
		parsed, err := strconv.ParseUint(value, 10, 64)
		if err != nil {
			return nil, werrors.NewParse(
				fmt.Sprintf("budget value for `%s` overflows u64", name))
		}
		if seen[name] {
			return nil, werrors.NewParse(fmt.Sprintf("duplicate budget key `%s`", name))
		}
		seen[name] = true
		switch name {
		case "max_results":
			b.MaxResults = &parsed
		case "max_bytes":
			b.MaxBytes = &parsed
		case "max_depth":
			b.MaxDepth = &parsed
		case "max_work":
			b.MaxWork = &parsed
		default:
			return nil, werrors.NewParse(fmt.Sprintf(
				"unknown budget key `%s` (expected max_results, max_bytes, max_depth, max_work)", name))
		}
	}
	if err := b.Validate(); err != nil {
		return nil, err
	}
	return b, nil
}

// ToWCB returns the canonical WCB map value.
func (b *Budget) ToWCB() wcb.Value {
	m := map[string]wcb.Value{}
	if b != nil && b.MaxResults != nil {
		m["max_results"] = wcb.U64Value(*b.MaxResults)
	}
	if b != nil && b.MaxBytes != nil {
		m["max_bytes"] = wcb.U64Value(*b.MaxBytes)
	}
	if b != nil && b.MaxDepth != nil {
		m["max_depth"] = wcb.U64Value(*b.MaxDepth)
	}
	if b != nil && b.MaxWork != nil {
		m["max_work"] = wcb.U64Value(*b.MaxWork)
	}
	return wcb.MapValue(m)
}

// BudgetFromWCB decodes a budget map, rejecting unknown/zero entries.
func BudgetFromWCB(v wcb.Value) (*Budget, error) {
	if v.Kind != wcb.Map {
		return nil, werrors.NewNonCanonical("budget must be a WCB map")
	}
	b := &Budget{}
	for key, value := range v.Map {
		if value.Kind != wcb.U64 {
			return nil, werrors.NewNonCanonical(
				fmt.Sprintf("budget `%s` must be a u64", key))
		}
		number := value.U
		switch key {
		case "max_results":
			b.MaxResults = &number
		case "max_bytes":
			b.MaxBytes = &number
		case "max_depth":
			b.MaxDepth = &number
		case "max_work":
			b.MaxWork = &number
		default:
			return nil, werrors.NewNonCanonical(
				fmt.Sprintf("unknown budget key `%s`", key))
		}
	}
	if err := b.Validate(); err != nil {
		return nil, err
	}
	return b, nil
}

// QueryValue returns the descriptor query fragment (no leading `budget=`).
func (b *Budget) QueryValue() string {
	var parts []string
	if b.MaxResults != nil {
		parts = append(parts, fmt.Sprintf("max_results=%d", *b.MaxResults))
	}
	if b.MaxBytes != nil {
		parts = append(parts, fmt.Sprintf("max_bytes=%d", *b.MaxBytes))
	}
	if b.MaxDepth != nil {
		parts = append(parts, fmt.Sprintf("max_depth=%d", *b.MaxDepth))
	}
	if b.MaxWork != nil {
		parts = append(parts, fmt.Sprintf("max_work=%d", *b.MaxWork))
	}
	return strings.Join(parts, ",")
}

// Equal reports structural equality.
func (b *Budget) Equal(other *Budget) bool {
	if b == nil || other == nil {
		return b == other
	}
	return equalU64Ptr(b.MaxResults, other.MaxResults) &&
		equalU64Ptr(b.MaxBytes, other.MaxBytes) &&
		equalU64Ptr(b.MaxDepth, other.MaxDepth) &&
		equalU64Ptr(b.MaxWork, other.MaxWork)
}

func equalU64Ptr(a, b *uint64) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

// JSON returns a plain map for CLI output.
func (b *Budget) JSON() map[string]any {
	if b == nil {
		return nil
	}
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
