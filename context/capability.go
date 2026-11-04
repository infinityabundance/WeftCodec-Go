package context

import (
	"sort"

	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// CapabilityScope is a capability grant record (WEFT-CAPABILITIES §1). It is an
// identifier for a grant, not a signature and not authority by itself.
type CapabilityScope struct {
	Principal  string
	Namespaces []string
	Roots      []hash.Hash32
	Predicates []string
	Budget     *Budget
	ExpiresAt  *int64
}

// NewCapabilityScope builds a canonical scope: lists are sorted and deduplicated.
func NewCapabilityScope(
	principal string,
	namespaces []string,
	roots []hash.Hash32,
	predicates []string,
	budget *Budget,
	expiresAt *int64,
) (*CapabilityScope, error) {
	ns := append([]string(nil), namespaces...)
	sort.Strings(ns)
	ns = dedupeStrings(ns)
	for _, namespace := range ns {
		if err := validateNamespace(namespace); err != nil {
			return nil, err
		}
	}
	rs, err := sortUniqueHashes(roots, "capability roots")
	if err != nil {
		return nil, err
	}
	ps := append([]string(nil), predicates...)
	sort.Strings(ps)
	ps = dedupeStrings(ps)
	for _, predicate := range ps {
		if err := validatePredicate(predicate); err != nil {
			return nil, err
		}
	}
	if budget != nil {
		if err := budget.Validate(); err != nil {
			return nil, err
		}
	}
	scope := &CapabilityScope{
		Principal:  principal,
		Namespaces: ns,
		Roots:      rs,
		Predicates: ps,
		Budget:     budget,
		ExpiresAt:  expiresAt,
	}
	if err := scope.Validate(limits.Default()); err != nil {
		return nil, err
	}
	return scope, nil
}

func dedupeStrings(values []string) []string {
	out := values[:0]
	for i, value := range values {
		if i == 0 || value != values[i-1] {
			out = append(out, value)
		}
	}
	return out
}

// Validate enforces the capability record invariants.
func (c *CapabilityScope) Validate(lim limits.Limits) error {
	if len(c.Principal) == 0 || len(c.Principal) > 256 {
		return werrors.NewNonCanonical("capability principal must be 1..=256 bytes")
	}
	if len(c.Namespaces) == 0 && len(c.Roots) == 0 {
		return werrors.NewNonCanonical(
			"capability must grant at least one namespace or root")
	}
	if err := checkCapabilityEntries(len(c.Namespaces), lim, "capability namespaces"); err != nil {
		return err
	}
	if err := checkCapabilityEntries(len(c.Roots), lim, "capability roots"); err != nil {
		return err
	}
	if err := checkCapabilityEntries(len(c.Predicates), lim, "capability predicates"); err != nil {
		return err
	}
	for _, namespace := range c.Namespaces {
		if err := validateNamespace(namespace); err != nil {
			return err
		}
	}
	if !strictlySortedStrings(c.Namespaces) {
		return werrors.NewNonCanonical(
			"capability namespaces must be sorted and unique")
	}
	if err := checkAscendingUnique(c.Roots, "capability roots"); err != nil {
		return err
	}
	if !strictlySortedStrings(c.Predicates) {
		return werrors.NewNonCanonical(
			"capability predicates must be sorted and unique")
	}
	for _, predicate := range c.Predicates {
		if err := validatePredicate(predicate); err != nil {
			return err
		}
	}
	if c.Budget != nil {
		if err := c.Budget.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// ToWCB returns the canonical WCB value.
func (c *CapabilityScope) ToWCB() wcb.Value {
	namespaces := make([]wcb.Value, 0, len(c.Namespaces))
	for _, namespace := range c.Namespaces {
		namespaces = append(namespaces, wcb.TextValue(namespace))
	}
	m := map[string]wcb.Value{
		"schema":     wcb.TextValue(CapabilitySchema),
		"principal":  wcb.TextValue(c.Principal),
		"namespaces": wcb.ArrayValue(namespaces),
		"roots":      hashesToValue(c.Roots),
	}
	if len(c.Predicates) != 0 {
		predicates := make([]wcb.Value, 0, len(c.Predicates))
		for _, predicate := range c.Predicates {
			predicates = append(predicates, wcb.TextValue(predicate))
		}
		m["predicates"] = wcb.ArrayValue(predicates)
	}
	if c.Budget != nil {
		m["budget"] = c.Budget.ToWCB()
	}
	if c.ExpiresAt != nil {
		m["expires_at"] = wcb.TimestampValue(*c.ExpiresAt)
	}
	return wcb.MapValue(m)
}

// CanonicalBytes encodes the record canonically.
func (c *CapabilityScope) CanonicalBytes(lim limits.Limits) ([]byte, error) {
	if err := c.Validate(lim); err != nil {
		return nil, err
	}
	return wcb.Encode(c.ToWCB(), lim)
}

// ID returns H("weft/capability/v1", canonical bytes).
func (c *CapabilityScope) ID() (hash.Hash32, error) {
	b, err := c.CanonicalBytes(limits.Default())
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(hash.DomainCapability, b), nil
}

// CapabilityScopeFromCanonicalBytes decodes canonical capability bytes strictly.
func CapabilityScopeFromCanonicalBytes(b []byte, lim limits.Limits) (*CapabilityScope, error) {
	value, err := wcb.Decode(b, lim)
	if err != nil {
		return nil, err
	}
	if value.Kind != wcb.Map {
		return nil, werrors.NewNonCanonical("capability record must be a WCB map")
	}
	m := value.Map
	for key := range m {
		switch key {
		case "schema", "principal", "namespaces", "roots", "predicates", "budget", "expires_at":
		default:
			return nil, werrors.NonCanonicalf("unknown capability record key `%s`", key)
		}
	}
	schema, err := requiredText(m, "schema")
	if err != nil {
		return nil, err
	}
	if schema != CapabilitySchema {
		return nil, werrors.UnknownSchemaf("unknown capability schema `%s`", schema)
	}
	principal, err := requiredText(m, "principal")
	if err != nil {
		return nil, err
	}
	namespaces, err := requiredTextArray(m, "namespaces", "capability namespaces")
	if err != nil {
		return nil, err
	}
	roots, err := requiredHashes(m, "roots", lim, "capability roots")
	if err != nil {
		return nil, err
	}
	var predicates []string
	if v, ok := m["predicates"]; ok {
		predicates, err = textArray(v, "capability predicates")
		if err != nil {
			return nil, err
		}
	}
	var budget *Budget
	if v, ok := m["budget"]; ok {
		budget, err = BudgetFromWCB(v)
		if err != nil {
			return nil, err
		}
	}
	var expiresAt *int64
	if v, ok := m["expires_at"]; ok {
		if v.Kind != wcb.Timestamp {
			return nil, werrors.NewNonCanonical(
				"capability expires_at must be a timestamp")
		}
		value := v.I
		expiresAt = &value
	}
	scope := &CapabilityScope{
		Principal:  principal,
		Namespaces: namespaces,
		Roots:      roots,
		Predicates: predicates,
		Budget:     budget,
		ExpiresAt:  expiresAt,
	}
	if err := scope.Validate(lim); err != nil {
		return nil, err
	}
	reencoded, err := wcb.Encode(scope.ToWCB(), lim)
	if err != nil {
		return nil, err
	}
	if !bytesEqual(reencoded, b) {
		return nil, werrors.NewNonCanonical("capability bytes are not in canonical form")
	}
	return scope, nil
}
