package selector

import (
	"bytes"
	"strings"

	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// ToWCB returns the canonical WCB value of the selector map.
func (s *Selector) ToWCB() wcb.Value {
	m := map[string]wcb.Value{
		"schema":    wcb.TextValue(Schema),
		"namespace": wcb.TextValue(s.Namespace),
		"predicate": exprToValue(s.Predicate),
	}
	if len(s.Order) != 0 {
		items := make([]wcb.Value, 0, len(s.Order))
		for _, item := range s.Order {
			dir := "asc"
			if item.Desc {
				dir = "desc"
			}
			items = append(items, wcb.MapValue(map[string]wcb.Value{
				"field": wcb.TextValue(item.Field.String()),
				"dir":   wcb.TextValue(dir),
			}))
		}
		m["order"] = wcb.ArrayValue(items)
	}
	if s.Limit != nil {
		m["limit"] = wcb.U64Value(*s.Limit)
	}
	if s.Offset != nil && *s.Offset != 0 {
		m["offset"] = wcb.U64Value(*s.Offset)
	}
	return wcb.MapValue(m)
}

// CanonicalBytes returns canonical WCB bytes for the selector.
func (s *Selector) CanonicalBytes(lim limits.Limits) ([]byte, error) {
	return wcb.Encode(s.ToWCB(), lim)
}

// SelectorID returns H("weft/selector/v1", canonical bytes).
func (s *Selector) SelectorID(lim limits.Limits) (hash.Hash32, error) {
	canonical, err := s.CanonicalBytes(lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(hash.DomainSelector, canonical), nil
}

// CanonicalBytes parses a selector source with default namespace and limits and
// returns its canonical bytes.
func CanonicalBytes(source string) ([]byte, error) {
	s, err := Parse(source)
	if err != nil {
		return nil, err
	}
	return s.CanonicalBytes(limits.Default())
}

// SelectorID parses a selector source with default namespace and limits and
// returns its selector id.
func SelectorID(source string) (hash.Hash32, error) {
	s, err := Parse(source)
	if err != nil {
		return hash.Hash32{}, err
	}
	return s.SelectorID(limits.Default())
}

func exprToValue(e *Expr) wcb.Value {
	switch e.Kind {
	case ExprTrue:
		return wcb.ArrayValue([]wcb.Value{wcb.TextValue("true")})
	case ExprFalse:
		return wcb.ArrayValue([]wcb.Value{wcb.TextValue("false")})
	case ExprHas:
		return wcb.ArrayValue([]wcb.Value{
			wcb.TextValue("has"), wcb.TextValue(e.Field.String()),
		})
	case ExprTag:
		return wcb.ArrayValue([]wcb.Value{
			wcb.TextValue("tag"), wcb.TextValue(e.Name),
		})
	case ExprRelation:
		return wcb.ArrayValue([]wcb.Value{
			wcb.TextValue("relation"),
			wcb.TextValue(e.Name),
			wcb.BytesValue(e.Target[:]),
		})
	case ExprCompare:
		return wcb.ArrayValue([]wcb.Value{
			wcb.TextValue(e.Op.discriminator()),
			wcb.TextValue(e.Field.String()),
			literalToValue(e.Value),
		})
	case ExprAnd, ExprOr:
		disc := "and"
		if e.Kind == ExprOr {
			disc = "or"
		}
		items := make([]wcb.Value, 0, len(e.Items))
		for _, item := range e.Items {
			items = append(items, exprToValue(item))
		}
		return wcb.ArrayValue([]wcb.Value{
			wcb.TextValue(disc), wcb.ArrayValue(items),
		})
	case ExprNot:
		return wcb.ArrayValue([]wcb.Value{
			wcb.TextValue("not"), exprToValue(e.Inner),
		})
	default:
		return wcb.NullValue()
	}
}

func literalToValue(l Literal) wcb.Value {
	switch l.Kind {
	case LitNull:
		return wcb.NullValue()
	case LitBool:
		return wcb.BoolValue(l.Bool)
	case LitU64:
		return wcb.U64Value(l.U)
	case LitI64:
		return wcb.I64Value(l.I)
	default:
		return wcb.TextValue(l.Str)
	}
}

func literalFromValue(v wcb.Value) (Literal, error) {
	switch v.Kind {
	case wcb.Null:
		return NullLiteral(), nil
	case wcb.Bool:
		return BoolLiteral(v.Bool), nil
	case wcb.U64:
		return U64Literal(v.U), nil
	case wcb.I64:
		return I64Literal(v.I), nil
	case wcb.Text:
		return StrLiteral(v.Str), nil
	default:
		return Literal{}, werrors.NewNonCanonical(
			"selector literal must be text, integer, bool, or null")
	}
}

func textOf(v wcb.Value, what string) (string, error) {
	if v.Kind != wcb.Text {
		return "", werrors.NewNonCanonical(what + " must be text")
	}
	return v.Str, nil
}

func requireArity(items []wcb.Value, expected int, what string) error {
	if len(items) != expected {
		return werrors.NonCanonicalf(
			"%s expression has %d elements, expected %d", what, len(items), expected)
	}
	return nil
}

func fieldFromCanonicalText(text string, lim limits.Limits) (Field, error) {
	segments := strings.Split(text, ".")
	for _, segment := range segments {
		if segment == "" {
			return Field{}, werrors.NewNonCanonical("empty field path segment")
		}
	}
	if uint64(len(segments)) > lim.MaxFieldSegments {
		return Field{}, werrors.NewLimitExceeded("field path has too many segments")
	}
	for _, segment := range segments {
		if !validIdent(segment) || isKeyword(segment) {
			return Field{}, werrors.NonCanonicalf(
				"invalid field path segment `%s`", segment)
		}
	}
	return Field{Segments: segments}, nil
}

// FromCanonicalBytes decodes canonical selector bytes back into an AST,
// rejecting every non-canonical form.
func FromCanonicalBytes(b []byte, lim limits.Limits) (*Selector, error) {
	value, err := wcb.Decode(b, lim)
	if err != nil {
		return nil, err
	}
	if value.Kind != wcb.Map {
		return nil, werrors.NewNonCanonical("selector value must be a WCB map")
	}
	m := value.Map
	for key := range m {
		switch key {
		case "schema", "namespace", "predicate", "order", "limit", "offset":
		default:
			return nil, werrors.NonCanonicalf("unknown selector map key `%s`", key)
		}
	}

	schemaValue, ok := m["schema"]
	if !ok {
		return nil, werrors.NewNonCanonical("selector map is missing schema")
	}
	if schemaValue.Kind != wcb.Text {
		return nil, werrors.NewNonCanonical("schema must be text")
	}
	if schemaValue.Str != Schema {
		return nil, werrors.UnknownSchemaf("unknown selector schema `%s`", schemaValue.Str)
	}

	namespaceValue, ok := m["namespace"]
	if !ok {
		return nil, werrors.NewNonCanonical("selector map is missing namespace")
	}
	if namespaceValue.Kind != wcb.Text {
		return nil, werrors.NewNonCanonical("namespace must be text")
	}
	if namespaceValue.Str == "" {
		return nil, werrors.NewNonCanonical("namespace must be non-empty")
	}

	predicateValue, ok := m["predicate"]
	if !ok {
		return nil, werrors.NewNonCanonical("selector map is missing predicate")
	}
	var operands uint64
	predicate, err := exprFromValue(predicateValue, lim, &operands, 0)
	if err != nil {
		return nil, err
	}

	var order []OrderItem
	if orderValue, ok := m["order"]; ok {
		if orderValue.Kind != wcb.Array {
			return nil, werrors.NewNonCanonical("order must be an array")
		}
		seen := make(map[string]bool)
		for _, item := range orderValue.Array {
			if item.Kind != wcb.Map {
				return nil, werrors.NewNonCanonical("order item must be a map")
			}
			for key := range item.Map {
				if key != "field" && key != "dir" {
					return nil, werrors.NewNonCanonical("unknown order item key")
				}
			}
			fieldValue, ok := item.Map["field"]
			if !ok || fieldValue.Kind != wcb.Text {
				return nil, werrors.NewNonCanonical("order field must be text")
			}
			field, err := fieldFromCanonicalText(fieldValue.Str, lim)
			if err != nil {
				return nil, err
			}
			dirValue, ok := item.Map["dir"]
			if !ok || dirValue.Kind != wcb.Text {
				return nil, werrors.NewNonCanonical(`order dir must be text "asc" or "desc"`)
			}
			var desc bool
			switch dirValue.Str {
			case "asc":
				desc = false
			case "desc":
				desc = true
			default:
				return nil, werrors.NewNonCanonical(`order dir must be text "asc" or "desc"`)
			}
			if seen[field.String()] {
				return nil, werrors.NewNonCanonical("duplicate order field")
			}
			seen[field.String()] = true
			order = append(order, OrderItem{Field: field, Desc: desc})
		}
	}

	var limitPtr *uint64
	if limitValue, ok := m["limit"]; ok {
		if limitValue.Kind != wcb.U64 {
			return nil, werrors.NewNonCanonical("limit must be u64")
		}
		v := limitValue.U
		limitPtr = &v
	}
	var offsetPtr *uint64
	if offsetValue, ok := m["offset"]; ok {
		if offsetValue.Kind != wcb.U64 {
			return nil, werrors.NewNonCanonical("offset must be u64")
		}
		if offsetValue.U == 0 {
			return nil, werrors.NewNonCanonical("offset 0 must be omitted from canonical form")
		}
		v := offsetValue.U
		offsetPtr = &v
	}

	s := &Selector{
		Namespace: namespaceValue.Str,
		Predicate: predicate,
		Order:     order,
		Limit:     limitPtr,
		Offset:    offsetPtr,
	}
	reencoded, err := s.CanonicalBytes(lim)
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(reencoded, b) {
		return nil, werrors.NewNonCanonical(
			"selector bytes are not in canonical normalized form")
	}
	return s, nil
}

func exprFromValue(value wcb.Value, lim limits.Limits, operands *uint64, depth uint64) (*Expr, error) {
	if depth > lim.MaxSelectorDepth {
		return nil, werrors.NewLimitExceeded("selector expression nesting too deep")
	}
	if *operands == ^uint64(0) {
		return nil, werrors.NewLimitExceeded("selector operand count overflow")
	}
	*operands = *operands + 1
	if *operands > lim.MaxSelectorOperands {
		return nil, werrors.LimitExceededf(
			"selector operand count %d exceeds limit %d", *operands, lim.MaxSelectorOperands)
	}
	if value.Kind != wcb.Array {
		return nil, werrors.NewNonCanonical("selector expression must be an array")
	}
	items := value.Array
	if len(items) == 0 {
		return nil, werrors.NewNonCanonical("empty selector expression")
	}
	discriminator, err := textOf(items[0], "selector discriminator")
	if err != nil {
		return nil, err
	}
	switch discriminator {
	case "true":
		if err := requireArity(items, 1, "`true`"); err != nil {
			return nil, err
		}
		return &Expr{Kind: ExprTrue}, nil
	case "false":
		if err := requireArity(items, 1, "`false`"); err != nil {
			return nil, err
		}
		return &Expr{Kind: ExprFalse}, nil
	case "has":
		if err := requireArity(items, 2, "has"); err != nil {
			return nil, err
		}
		text, err := textOf(items[1], "has field")
		if err != nil {
			return nil, err
		}
		field, err := fieldFromCanonicalText(text, lim)
		if err != nil {
			return nil, err
		}
		return &Expr{Kind: ExprHas, Field: field}, nil
	case "tag":
		if err := requireArity(items, 2, "tag"); err != nil {
			return nil, err
		}
		name, err := textOf(items[1], "tag name")
		if err != nil {
			return nil, err
		}
		return &Expr{Kind: ExprTag, Name: name}, nil
	case "relation":
		if err := requireArity(items, 3, "relation"); err != nil {
			return nil, err
		}
		predicate, err := textOf(items[1], "relation predicate")
		if err != nil {
			return nil, err
		}
		if items[2].Kind != wcb.Bytes || len(items[2].Bytes) != hash.HashLen {
			return nil, werrors.NewNonCanonical("relation target must be exactly 32 bytes")
		}
		target, err := hash.FromSlice(items[2].Bytes)
		if err != nil {
			return nil, werrors.NewNonCanonical("relation target must be a WCB bytes value")
		}
		return &Expr{Kind: ExprRelation, Name: predicate, Target: target}, nil
	case "attr_eq", "attr_ne", "attr_lt", "attr_le", "attr_gt", "attr_ge":
		if err := requireArity(items, 3, "comparison"); err != nil {
			return nil, err
		}
		var op CmpOp
		switch discriminator {
		case "attr_eq":
			op = OpEq
		case "attr_ne":
			op = OpNe
		case "attr_lt":
			op = OpLt
		case "attr_le":
			op = OpLe
		case "attr_gt":
			op = OpGt
		default:
			op = OpGe
		}
		text, err := textOf(items[1], "comparison field")
		if err != nil {
			return nil, err
		}
		field, err := fieldFromCanonicalText(text, lim)
		if err != nil {
			return nil, err
		}
		literal, err := literalFromValue(items[2])
		if err != nil {
			return nil, err
		}
		return &Expr{Kind: ExprCompare, Field: field, Op: op, Value: literal}, nil
	case "and", "or":
		if err := requireArity(items, 2, "boolean operator"); err != nil {
			return nil, err
		}
		if items[1].Kind != wcb.Array {
			return nil, werrors.NewNonCanonical("and/or operands must be a WCB array")
		}
		nested := items[1].Array
		if len(nested) < 2 {
			return nil, werrors.NewNonCanonical("and/or requires at least 2 operands")
		}
		parsed := make([]*Expr, 0, len(nested))
		kind := ExprAnd
		if discriminator == "or" {
			kind = ExprOr
		}
		for _, item := range nested {
			child, err := exprFromValue(item, lim, operands, depth+1)
			if err != nil {
				return nil, err
			}
			if child.Kind == kind {
				return nil, werrors.NewNonCanonical(
					"nested same-operator and/or must be flattened")
			}
			parsed = append(parsed, child)
		}
		return &Expr{Kind: kind, Items: parsed}, nil
	case "not":
		if err := requireArity(items, 2, "not"); err != nil {
			return nil, err
		}
		child, err := exprFromValue(items[1], lim, operands, depth+1)
		if err != nil {
			return nil, err
		}
		switch child.Kind {
		case ExprTrue, ExprFalse:
			return nil, werrors.NewNonCanonical(
				"not true/false must be normalized to false/true")
		case ExprNot:
			return nil, werrors.NewNonCanonical("double negation must be removed")
		default:
			return &Expr{Kind: ExprNot, Inner: child}, nil
		}
	default:
		return nil, werrors.NonCanonicalf(
			"unknown selector discriminator `%s`", discriminator)
	}
}
