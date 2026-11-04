// Package selector implements WeftQL parsing, normalization, canonical WCB
// encoding, and selector identity (WEFT-SELECTORS v1).
//
// The canonical bytes of the AST are hashed to produce SelectorId:
//
//	selector_bytes = WCB(selector map)
//	selector_id    = H("weft/selector/v1", selector_bytes)
//
// SelectorId identifies the expression, never a result set.
package selector

import (
	"strings"

	"github.com/infinityabundance/WeftCodec-Go/hash"
)

// Schema is the v1 selector schema string.
const Schema = "weft.selector.v1"

// DefaultNamespace is used by the text grammar (which has no namespace clause).
const DefaultNamespace = "default"

var reserved = map[string]bool{
	"select": true, "and": true, "or": true, "not": true,
	"true": true, "false": true, "null": true,
	"has": true, "tag": true, "relation": true,
	"order": true, "by": true, "asc": true, "desc": true,
	"limit": true, "offset": true,
}

func isKeyword(s string) bool { return reserved[s] }

// Field is a dotted attribute path (limited to 16 segments).
type Field struct {
	Segments []string
}

// String joins segments with dots.
func (f Field) String() string { return strings.Join(f.Segments, ".") }

// LiteralKind enumerates scalar literal kinds.
type LiteralKind uint8

const (
	LitNull LiteralKind = iota
	LitBool
	LitU64
	LitI64
	LitStr
)

// Literal is a selector scalar literal.
type Literal struct {
	Kind LiteralKind
	Bool bool
	U    uint64
	I    int64
	Str  string
}

func NullLiteral() Literal        { return Literal{Kind: LitNull} }
func BoolLiteral(v bool) Literal  { return Literal{Kind: LitBool, Bool: v} }
func U64Literal(v uint64) Literal { return Literal{Kind: LitU64, U: v} }
func I64Literal(v int64) Literal  { return Literal{Kind: LitI64, I: v} }
func StrLiteral(v string) Literal { return Literal{Kind: LitStr, Str: v} }

// CmpOp is a comparison operator.
type CmpOp uint8

const (
	OpEq CmpOp = iota
	OpNe
	OpLt
	OpLe
	OpGt
	OpGe
)

func (o CmpOp) discriminator() string {
	switch o {
	case OpEq:
		return "attr_eq"
	case OpNe:
		return "attr_ne"
	case OpLt:
		return "attr_lt"
	case OpLe:
		return "attr_le"
	case OpGt:
		return "attr_gt"
	default:
		return "attr_ge"
	}
}

// ExprKind enumerates predicate node kinds.
type ExprKind uint8

const (
	ExprTrue ExprKind = iota
	ExprFalse
	ExprHas
	ExprTag
	ExprRelation
	ExprCompare
	ExprAnd
	ExprOr
	ExprNot
)

// Expr is a predicate AST node. And/Or operands are in source order; order is
// semantic for evidence and is never sorted.
type Expr struct {
	Kind   ExprKind
	Field  Field
	Op     CmpOp
	Value  Literal
	Name   string // tag name or relation predicate
	Target hash.Hash32
	Items  []*Expr
	Inner  *Expr
}

// Equal reports structural equality.
func (e *Expr) Equal(other *Expr) bool {
	if e == nil || other == nil {
		return e == other
	}
	if e.Kind != other.Kind || e.Op != other.Op || e.Name != other.Name ||
		e.Target != other.Target || e.Value != other.Value ||
		e.Field.String() != other.Field.String() {
		return false
	}
	if len(e.Items) != len(other.Items) {
		return false
	}
	for i := range e.Items {
		if !e.Items[i].Equal(other.Items[i]) {
			return false
		}
	}
	return e.Inner.Equal(other.Inner)
}

// OrderItem is one ordering term. Desc false means ascending (the default).
type OrderItem struct {
	Field Field
	Desc  bool
}

// Selector is a parsed, normalized selector.
type Selector struct {
	Namespace string
	Predicate *Expr
	Order     []OrderItem
	Limit     *uint64
	Offset    *uint64
}

// Equal reports structural equality.
func (s *Selector) Equal(other *Selector) bool {
	if s == nil || other == nil {
		return s == other
	}
	if s.Namespace != other.Namespace || !s.Predicate.Equal(other.Predicate) ||
		len(s.Order) != len(other.Order) {
		return false
	}
	for i := range s.Order {
		if s.Order[i].Field.String() != other.Order[i].Field.String() || s.Order[i].Desc != other.Order[i].Desc {
			return false
		}
	}
	if (s.Limit == nil) != (other.Limit == nil) || (s.Offset == nil) != (other.Offset == nil) {
		return false
	}
	if s.Limit != nil && *s.Limit != *other.Limit {
		return false
	}
	if s.Offset != nil && *s.Offset != *other.Offset {
		return false
	}
	return true
}
