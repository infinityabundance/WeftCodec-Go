package context

import (
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// CausalClosure is the causal closure record (WEFT-CAUSAL §1).
type CausalClosure struct {
	Parents []hash.Hash32
	Closure []hash.Hash32
}

// FromParents builds a closure from explicit parents only: sorted, unique.
func FromParents(parents []hash.Hash32) (*CausalClosure, error) {
	sorted, err := sortUniqueHashes(parents, "causal parents")
	if err != nil {
		return nil, err
	}
	closure := &CausalClosure{
		Parents: sorted,
		Closure: append([]hash.Hash32(nil), sorted...),
	}
	if err := closure.Validate(limits.Default()); err != nil {
		return nil, err
	}
	return closure, nil
}

// NewCausalClosure builds and validates a closure from explicit parents and a
// full closure.
func NewCausalClosure(parents, closure []hash.Hash32) (*CausalClosure, error) {
	c := &CausalClosure{Parents: parents, Closure: closure}
	if err := c.Validate(limits.Default()); err != nil {
		return nil, err
	}
	return c, nil
}

// Validate enforces the canonical array rules and the superset rule.
func (c *CausalClosure) Validate(lim limits.Limits) error {
	if err := checkCausalEntries(len(c.Parents), lim, "causal parents"); err != nil {
		return err
	}
	if err := checkCausalEntries(len(c.Closure), lim, "causal closure"); err != nil {
		return err
	}
	if err := checkAscendingUnique(c.Parents, "causal parents"); err != nil {
		return err
	}
	if err := checkAscendingUnique(c.Closure, "causal closure"); err != nil {
		return err
	}
	for _, parent := range c.Parents {
		if !containsHash(c.Closure, parent) {
			return werrors.NonCanonicalf("causal closure is missing parent %s", parent)
		}
	}
	return nil
}

// ToWCB returns the canonical WCB value.
func (c *CausalClosure) ToWCB() wcb.Value {
	return wcb.MapValue(map[string]wcb.Value{
		"schema":  wcb.TextValue(CausalSchema),
		"parents": hashesToValue(c.Parents),
		"closure": hashesToValue(c.Closure),
	})
}

// CanonicalBytes encodes the record canonically.
func (c *CausalClosure) CanonicalBytes(lim limits.Limits) ([]byte, error) {
	if err := c.Validate(lim); err != nil {
		return nil, err
	}
	return wcb.Encode(c.ToWCB(), lim)
}

// ID returns H("weft/causal/v1", canonical bytes).
func (c *CausalClosure) ID() (hash.Hash32, error) {
	b, err := c.CanonicalBytes(limits.Default())
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(hash.DomainCausal, b), nil
}

// CausalClosureFromCanonicalBytes decodes canonical closure bytes strictly.
func CausalClosureFromCanonicalBytes(b []byte, lim limits.Limits) (*CausalClosure, error) {
	value, err := wcb.Decode(b, lim)
	if err != nil {
		return nil, err
	}
	if value.Kind != wcb.Map {
		return nil, werrors.NewNonCanonical("causal record must be a WCB map")
	}
	m := value.Map
	for key := range m {
		switch key {
		case "schema", "parents", "closure":
		default:
			return nil, werrors.NonCanonicalf("unknown causal record key `%s`", key)
		}
	}
	schema, err := requiredText(m, "schema")
	if err != nil {
		return nil, err
	}
	if schema != CausalSchema {
		return nil, werrors.UnknownSchemaf("unknown causal schema `%s`", schema)
	}
	parents, err := requiredHashes(m, "parents", lim, "causal parents")
	if err != nil {
		return nil, err
	}
	closure, err := requiredHashes(m, "closure", lim, "causal closure")
	if err != nil {
		return nil, err
	}
	record := &CausalClosure{Parents: parents, Closure: closure}
	if err := record.Validate(lim); err != nil {
		return nil, err
	}
	reencoded, err := wcb.Encode(record.ToWCB(), lim)
	if err != nil {
		return nil, err
	}
	if !bytesEqual(reencoded, b) {
		return nil, werrors.NewNonCanonical("causal bytes are not in canonical form")
	}
	return record, nil
}

// Equal reports structural equality.
func (c *CausalClosure) Equal(other *CausalClosure) bool {
	if c == nil || other == nil {
		return c == other
	}
	return equalHashes(c.Parents, other.Parents) && equalHashes(c.Closure, other.Closure)
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
