package selector

import (
	"bytes"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

func mustParse(t *testing.T, source string) *Selector {
	t.Helper()
	s, err := Parse(source)
	if err != nil {
		t.Fatalf("%s: %v", source, err)
	}
	return s
}

func idHex(t *testing.T, s *Selector) string {
	t.Helper()
	id, err := s.SelectorID(limits.Default())
	if err != nil {
		t.Fatal(err)
	}
	return id.Hex()
}

func TestCanonicalExample(t *testing.T) {
	source := `select attrs.project == "demo" and relation(derived_from, ` +
		`obj:00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff) ` +
		`order by attrs.seq desc limit 10`
	s := mustParse(t, source)
	if s.Namespace != DefaultNamespace {
		t.Fatalf("namespace %q", s.Namespace)
	}
	if s.Limit == nil || *s.Limit != 10 {
		t.Fatal("limit")
	}
	if len(s.Order) != 1 || !s.Order[0].Desc {
		t.Fatal("order")
	}
	canonical, err := s.CanonicalBytes(limits.Default())
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := FromCanonicalBytes(canonical, limits.Default())
	if err != nil {
		t.Fatal(err)
	}
	if !decoded.Equal(s) {
		t.Fatal("AST round-trip mismatch")
	}
}

func TestEquivalence(t *testing.T) {
	a := mustParse(t, "select attrs.x == 1 order by a")
	c := mustParse(t, "select attrs.x==1 order by a asc")
	d := mustParse(t, "select\nattrs.x == 01\torder by a")
	if idHex(t, a) != idHex(t, c) || idHex(t, a) != idHex(t, d) {
		t.Fatal("whitespace, formatting, and leading-zero equivalence failed")
	}

	without := mustParse(t, "select attrs.x == 1")
	withZero := mustParse(t, "select attrs.x == 1 offset 0")
	left, _ := without.CanonicalBytes(limits.Default())
	right, _ := withZero.CanonicalBytes(limits.Default())
	if !bytes.Equal(left, right) {
		t.Fatal("offset 0 must be omitted")
	}

	upper := mustParse(t, "select relation(p, AB112233445566778899aabbccddeeff00112233445566778899aabbccddeeff)")
	lower := mustParse(t, "select relation(p, obj:ab112233445566778899aabbccddeeff00112233445566778899aabbccddeeff)")
	if idHex(t, upper) != idHex(t, lower) {
		t.Fatal("hex case and obj: prefix equivalence failed")
	}
}

func TestOperandOrderChangesIdentity(t *testing.T) {
	a := mustParse(t, "select a == 1 and b == 2")
	b := mustParse(t, "select b == 2 and a == 1")
	if idHex(t, a) == idHex(t, b) {
		t.Fatal("operand reordering must change identity")
	}
}

func TestNormalization(t *testing.T) {
	if mustParse(t, "select not not true").Predicate.Kind != ExprTrue {
		t.Fatal("double negation")
	}
	if mustParse(t, "select not false").Predicate.Kind != ExprTrue {
		t.Fatal("not false")
	}
	s := mustParse(t, "select a == 1 and (b == 2 and c == 3)")
	if s.Predicate.Kind != ExprAnd || len(s.Predicate.Items) != 3 {
		t.Fatal("and flattening")
	}
	s = mustParse(t, "select a == 1 or (b == 2 or c == 3)")
	if s.Predicate.Kind != ExprOr || len(s.Predicate.Items) != 3 {
		t.Fatal("or flattening")
	}
}

func TestGrammarForms(t *testing.T) {
	for _, source := range []string{
		"select true",
		"select false",
		"select has(attrs.x)",
		`select tag("demo")`,
		"select relation(p, 00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff)",
		`select a == "x"`,
		"select a != 1",
		"select a < -2",
		"select a <= 3 and b >= 4 or not c > 5",
		"select a == null order by a, b desc limit 5 offset 2",
	} {
		if _, err := Parse(source); err != nil {
			t.Fatalf("%s: %v", source, err)
		}
	}
}

func TestErrors(t *testing.T) {
	cases := []string{
		"select a = 1",
		`select a == "\uD800"`,
		`select a == "\q"`,
		"select a == \"raw\ncontrol\"",
		"SELECT true",
	}
	for _, source := range cases {
		_, err := Parse(source)
		if err == nil || werrors.ClassOf(err) != werrors.Parse {
			t.Fatalf("%q: got %v, want Parse", source, err)
		}
	}
	if _, err := Parse(`select a == "\uD83D\uDE00"`); err != nil {
		t.Fatalf("valid surrogate pair: %v", err)
	}
}

func TestLimits(t *testing.T) {
	lim := limits.Default()
	lim.MaxSelectorBytes = 4
	if _, err := ParseWith("select true", DefaultNamespace, lim); werrors.ClassOf(err) != werrors.LimitExceeded {
		t.Fatalf("selector bytes: %v", err)
	}
	lim = limits.Default()
	lim.MaxFieldSegments = 1
	if _, err := ParseWith("select a.b == 1", DefaultNamespace, lim); werrors.ClassOf(err) != werrors.LimitExceeded {
		t.Fatalf("field segments: %v", err)
	}
	lim = limits.Default()
	lim.MaxSelectorDepth = 1
	if _, err := ParseWith("select not not true", DefaultNamespace, lim); werrors.ClassOf(err) != werrors.LimitExceeded {
		t.Fatalf("depth: %v", err)
	}
}

func TestFromCanonicalRejects(t *testing.T) {
	lim := limits.Default()

	s := mustParse(t, "select a == 1 offset 2")
	m := s.ToWCB().Map
	m["offset"] = wcb.U64Value(0)
	bytesWithZeroOffset, err := wcb.Encode(wcb.MapValue(m), lim)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := FromCanonicalBytes(bytesWithZeroOffset, lim); werrors.ClassOf(err) != werrors.NonCanonical {
		t.Fatalf("offset zero: %v", err)
	}

	s = mustParse(t, "select true")
	m = s.ToWCB().Map
	m["schema"] = wcb.TextValue("other")
	wrongSchema, err := wcb.Encode(wcb.MapValue(m), lim)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := FromCanonicalBytes(wrongSchema, lim); werrors.ClassOf(err) != werrors.UnknownSchema {
		t.Fatalf("unknown schema: %v", err)
	}

	// Nested same-operator and/or must be flattened.
	s = mustParse(t, "select a == 1 and b == 2 and c == 3")
	m = s.ToWCB().Map
	items := s.Predicate.Items
	nested := wcb.ArrayValue([]wcb.Value{
		wcb.TextValue("and"),
		wcb.ArrayValue([]wcb.Value{
			wcb.ArrayValue([]wcb.Value{
				wcb.TextValue("and"),
				wcb.ArrayValue([]wcb.Value{exprToValue(items[0]), exprToValue(items[1])}),
			}),
			exprToValue(items[2]),
		}),
	})
	m["predicate"] = nested
	nestedBytes, err := wcb.Encode(wcb.MapValue(m), lim)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := FromCanonicalBytes(nestedBytes, lim); werrors.ClassOf(err) != werrors.NonCanonical {
		t.Fatalf("nested and: %v", err)
	}

	// not true must be normalized.
	s = mustParse(t, "select not false")
	m = s.ToWCB().Map
	m["predicate"] = wcb.ArrayValue([]wcb.Value{
		wcb.TextValue("not"),
		wcb.ArrayValue([]wcb.Value{wcb.TextValue("true")}),
	})
	notTrue, err := wcb.Encode(wcb.MapValue(m), lim)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := FromCanonicalBytes(notTrue, lim); werrors.ClassOf(err) != werrors.NonCanonical {
		t.Fatalf("not true: %v", err)
	}
}

func TestSelectorIDDomain(t *testing.T) {
	s := mustParse(t, "select true")
	canonical, err := s.CanonicalBytes(limits.Default())
	if err != nil {
		t.Fatal(err)
	}
	id, err := s.SelectorID(limits.Default())
	if err != nil {
		t.Fatal(err)
	}
	if id != hash.Domain(hash.DomainSelector, canonical) {
		t.Fatal("selector id domain")
	}
}
