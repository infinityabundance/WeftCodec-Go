package context

import (
	"fmt"
	"strings"

	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/selector"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// ContextRoot is a Context Root record (WEFT-CONTEXT-ROOT §2).
type ContextRoot struct {
	Namespace      string
	SelectorBytes  []byte
	SelectorID     hash.Hash32
	Snapshot       hash.Hash32
	Resolver       string
	Causal         []hash.Hash32
	Ordering       string
	CapabilityHash *hash.Hash32
	Budget         *Budget
}

// NewContextRoot builds a Context Root from a parsed selector and pinned view
// parameters. Causal entries are sorted and deduplicated.
func NewContextRoot(
	sel *selector.Selector,
	snapshot hash.Hash32,
	resolver string,
	causal []hash.Hash32,
	capabilityHash *hash.Hash32,
	budget *Budget,
) (*ContextRoot, error) {
	if err := validateNamespace(sel.Namespace); err != nil {
		return nil, err
	}
	if snapshot == hash.Zero {
		return nil, werrors.NewNonCanonical("snapshot must not be all zero")
	}
	if err := validateResolver(resolver); err != nil {
		return nil, err
	}
	if capabilityHash != nil && *capabilityHash == hash.Zero {
		return nil, werrors.NewNonCanonical("capability hash must not be all zero")
	}
	if budget != nil {
		if err := budget.Validate(); err != nil {
			return nil, err
		}
	}
	sorted, err := sortUniqueHashes(causal, "causal closure")
	if err != nil {
		return nil, err
	}
	lim := limits.Default()
	selectorBytes, err := sel.CanonicalBytes(lim)
	if err != nil {
		return nil, err
	}
	selectorID := hash.Domain(hash.DomainSelector, selectorBytes)
	root := &ContextRoot{
		Namespace:      sel.Namespace,
		SelectorBytes:  append([]byte(nil), selectorBytes...),
		SelectorID:     selectorID,
		Snapshot:       snapshot,
		Resolver:       resolver,
		Causal:         sorted,
		Ordering:       OrderingSelectorThenObjectID,
		CapabilityHash: capabilityHash,
		Budget:         budget,
	}
	if err := root.Validate(lim); err != nil {
		return nil, err
	}
	return root, nil
}

// Selector decodes the embedded selector.
func (r *ContextRoot) Selector() (*selector.Selector, error) {
	return selector.FromCanonicalBytes(r.SelectorBytes, limits.Default())
}

// Validate checks every record invariant.
func (r *ContextRoot) Validate(lim limits.Limits) error {
	if err := validateNamespace(r.Namespace); err != nil {
		return err
	}
	if err := validateResolver(r.Resolver); err != nil {
		return err
	}
	if r.Snapshot == hash.Zero {
		return werrors.NewNonCanonical("snapshot must not be all zero")
	}
	if r.Ordering != OrderingSelectorThenObjectID {
		return werrors.UnsupportedVersionf(
			"unknown ordering policy `%s` (v1 supports `%s`)",
			r.Ordering, OrderingSelectorThenObjectID)
	}
	if r.CapabilityHash != nil && *r.CapabilityHash == hash.Zero {
		return werrors.NewNonCanonical("capability hash must not be all zero")
	}
	if err := checkCausalEntries(len(r.Causal), lim, "causal closure"); err != nil {
		return err
	}
	if err := checkAscendingUnique(r.Causal, "causal closure"); err != nil {
		return err
	}
	if r.Budget != nil {
		if err := r.Budget.Validate(); err != nil {
			return err
		}
	}
	sel, err := selector.FromCanonicalBytes(r.SelectorBytes, lim)
	if err != nil {
		return err
	}
	if sel.Namespace != r.Namespace {
		return werrors.NonCanonicalf(
			"selector namespace `%s` does not match context namespace `%s`",
			sel.Namespace, r.Namespace)
	}
	reencoded, err := sel.CanonicalBytes(lim)
	if err != nil {
		return err
	}
	if !bytesEqual(reencoded, r.SelectorBytes) {
		return werrors.NewNonCanonical("selector bytes are not in canonical form")
	}
	expected := hash.Domain(hash.DomainSelector, r.SelectorBytes)
	if expected != r.SelectorID {
		return werrors.NewNonCanonical(
			"selector_id does not match the embedded selector bytes")
	}
	return nil
}

// ToWCB returns the canonical WCB value.
func (r *ContextRoot) ToWCB() wcb.Value {
	m := map[string]wcb.Value{
		"schema":      wcb.TextValue(ContextRootSchema),
		"namespace":   wcb.TextValue(r.Namespace),
		"selector":    wcb.BytesValue(append([]byte(nil), r.SelectorBytes...)),
		"selector_id": wcb.BytesValue(hashBytes(r.SelectorID)),
		"snapshot":    wcb.BytesValue(hashBytes(r.Snapshot)),
		"resolver":    wcb.TextValue(r.Resolver),
		"causal":      hashesToValue(r.Causal),
		"ordering":    wcb.TextValue(r.Ordering),
	}
	if r.CapabilityHash != nil {
		m["capability_hash"] = wcb.BytesValue(hashBytes(*r.CapabilityHash))
	}
	if r.Budget != nil {
		m["budget"] = r.Budget.ToWCB()
	}
	return wcb.MapValue(m)
}

// CanonicalBytes encodes the record canonically.
func (r *ContextRoot) CanonicalBytes(lim limits.Limits) ([]byte, error) {
	if err := r.Validate(lim); err != nil {
		return nil, err
	}
	return wcb.Encode(r.ToWCB(), lim)
}

// CanonicalHex returns the canonical record as lowercase hex.
func (r *ContextRoot) CanonicalHex(lim limits.Limits) (string, error) {
	b, err := r.CanonicalBytes(lim)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", b), nil
}

// ID returns H("weft/context-root/v1", canonical bytes).
func (r *ContextRoot) ID() (hash.Hash32, error) {
	b, err := r.CanonicalBytes(limits.Default())
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(hash.DomainContextRoot, b), nil
}

// AuthoritativeURI returns weft://namespace/c/<ContextRootId>.
func (r *ContextRoot) AuthoritativeURI() (string, error) {
	id, err := r.ID()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%s%s/c/%s", URIPrefix, r.Namespace, id.Hex()), nil
}

// DescriptorURI returns the descriptive spelling for this root and selector
// source. The source must parse to exactly the record's canonical selector.
func (r *ContextRoot) DescriptorURI(selectorSource string) (string, error) {
	parsed, err := selector.ParseWith(selectorSource, r.Namespace, limits.Default())
	if err != nil {
		return "", err
	}
	canonical, err := parsed.CanonicalBytes(limits.Default())
	if err != nil {
		return "", err
	}
	if !bytesEqual(canonical, r.SelectorBytes) {
		return "", werrors.NewNonCanonical(
			"selector source does not denote the record's canonical selector")
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%s%s/c/%s@%s", URIPrefix, r.Namespace,
		PercentEncode(selectorSource), r.Snapshot.Hex())
	var params []string
	if len(r.Causal) != 0 {
		params = append(params, "closure="+hashesToHexList(r.Causal))
	}
	if r.CapabilityHash != nil {
		params = append(params, "cap="+r.CapabilityHash.Hex())
	}
	if r.Budget != nil {
		params = append(params, "budget="+r.Budget.QueryValue())
	}
	if r.Resolver != DefaultResolver {
		params = append(params, "resolver="+r.Resolver)
	}
	if len(params) != 0 {
		b.WriteByte('?')
		b.WriteString(strings.Join(params, "&"))
	}
	return b.String(), nil
}

// ContextRootFromCanonicalBytes decodes canonical record bytes strictly.
func ContextRootFromCanonicalBytes(b []byte, lim limits.Limits) (*ContextRoot, error) {
	value, err := wcb.Decode(b, lim)
	if err != nil {
		return nil, err
	}
	if value.Kind != wcb.Map {
		return nil, werrors.NewNonCanonical("context root must be a WCB map")
	}
	m := value.Map
	for key := range m {
		switch key {
		case "schema", "namespace", "selector", "selector_id", "snapshot",
			"resolver", "causal", "ordering", "capability_hash", "budget":
		default:
			return nil, werrors.NonCanonicalf("unknown context-root key `%s`", key)
		}
	}
	schema, err := requiredText(m, "schema")
	if err != nil {
		return nil, err
	}
	if schema != ContextRootSchema {
		return nil, werrors.UnknownSchemaf("unknown context-root schema `%s`", schema)
	}
	namespace, err := requiredText(m, "namespace")
	if err != nil {
		return nil, err
	}
	selectorBytes, err := requiredBytes(m, "selector")
	if err != nil {
		return nil, err
	}
	selectorID, err := requiredHash(m, "selector_id", "selector_id")
	if err != nil {
		return nil, err
	}
	snapshot, err := requiredHash(m, "snapshot", "snapshot")
	if err != nil {
		return nil, err
	}
	resolver, err := requiredText(m, "resolver")
	if err != nil {
		return nil, err
	}
	causal, err := requiredHashes(m, "causal", lim, "causal closure")
	if err != nil {
		return nil, err
	}
	ordering, err := requiredText(m, "ordering")
	if err != nil {
		return nil, err
	}
	var capabilityHash *hash.Hash32
	if v, ok := m["capability_hash"]; ok {
		h, err := hashFromValue(v, "capability_hash")
		if err != nil {
			return nil, err
		}
		capabilityHash = &h
	}
	var budget *Budget
	if v, ok := m["budget"]; ok {
		budget, err = BudgetFromWCB(v)
		if err != nil {
			return nil, err
		}
	}
	root := &ContextRoot{
		Namespace:      namespace,
		SelectorBytes:  selectorBytes,
		SelectorID:     selectorID,
		Snapshot:       snapshot,
		Resolver:       resolver,
		Causal:         causal,
		Ordering:       ordering,
		CapabilityHash: capabilityHash,
		Budget:         budget,
	}
	if err := root.Validate(lim); err != nil {
		return nil, err
	}
	reencoded, err := wcb.Encode(root.ToWCB(), lim)
	if err != nil {
		return nil, err
	}
	if !bytesEqual(reencoded, b) {
		return nil, werrors.NewNonCanonical(
			"context-root bytes are not in canonical form")
	}
	return root, nil
}

// BuildContextRoot is the convenience constructor used by the CLI: it parses
// the selector source, validates the snapshot hex, and canonicalizes causal.
func BuildContextRoot(
	selectorSource, namespace, snapshotHex, resolver string,
	causal []hash.Hash32,
	capabilityHash *hash.Hash32,
	budget *Budget,
) (*ContextRoot, error) {
	lim := limits.Default()
	sel, err := selector.ParseWith(selectorSource, namespace, lim)
	if err != nil {
		return nil, err
	}
	snapshot, err := parseLowerHex32(snapshotHex, "snapshot")
	if err != nil {
		return nil, err
	}
	return NewContextRoot(sel, snapshot, resolver, causal, capabilityHash, budget)
}
