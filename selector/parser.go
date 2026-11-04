package selector

import (
	"math"
	"strings"
	"unicode/utf8"

	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// Parse parses with the default namespace and default limits.
func Parse(source string) (*Selector, error) {
	return ParseWith(source, DefaultNamespace, limits.Default())
}

// ParseWith parses and normalizes a WeftQL selector with an explicit namespace
// and limits.
func ParseWith(source, namespace string, lim limits.Limits) (*Selector, error) {
	if uint64(len(source)) > lim.MaxSelectorBytes {
		return nil, werrors.LimitExceededf(
			"selector source length %d exceeds limit %d", len(source), lim.MaxSelectorBytes)
	}
	if !utf8.ValidString(source) {
		return nil, werrors.NewParse("selector source is not valid UTF-8")
	}
	if namespace == "" {
		return nil, werrors.NewParse("namespace must not be empty")
	}
	if uint64(len(namespace)) > lim.MaxStringBytes {
		return nil, werrors.NewLimitExceeded("namespace too long")
	}
	tokens, err := (&lexer{src: source}).tokenize()
	if err != nil {
		return nil, err
	}
	p := &parser{tokens: tokens, limits: lim}
	return p.parseQuery(namespace)
}

type parser struct {
	tokens   []token
	index    int
	limits   limits.Limits
	operands uint64
}

func (p *parser) peek() token {
	i := p.index
	if i >= len(p.tokens) {
		i = len(p.tokens) - 1
	}
	return p.tokens[i]
}

func (p *parser) peekPos() int { return p.peek().pos }

func (p *parser) bump() token {
	tok := p.peek()
	if p.index+1 < len(p.tokens) {
		p.index++
	}
	return tok
}

func (p *parser) checkDepth(depth uint32) error {
	if uint64(depth) > p.limits.MaxSelectorDepth {
		return werrors.LimitExceededf(
			"selector nesting depth %d exceeds limit %d", depth, p.limits.MaxSelectorDepth)
	}
	return nil
}

func (p *parser) bumpOperand() error {
	if p.operands == ^uint64(0) {
		return werrors.NewLimitExceeded("selector operand count overflow")
	}
	p.operands++
	if p.operands > p.limits.MaxSelectorOperands {
		return werrors.LimitExceededf(
			"selector operand count %d exceeds limit %d", p.operands, p.limits.MaxSelectorOperands)
	}
	return nil
}

func (p *parser) eatKeyword(keyword string) bool {
	tok := p.peek()
	if tok.kind == tokIdent && tok.text == keyword {
		p.bump()
		return true
	}
	return false
}

func (p *parser) expectKeyword(keyword string) (int, error) {
	pos := p.peekPos()
	tok := p.bump()
	if tok.kind == tokIdent && tok.text == keyword {
		return pos, nil
	}
	return 0, werrors.ParseAtf(pos, "expected `%s`", keyword)
}

func (p *parser) expectToken(kind tokenKind, name string) (int, error) {
	pos := p.peekPos()
	tok := p.bump()
	if tok.kind == kind {
		return pos, nil
	}
	return 0, werrors.ParseAtf(tok.pos, "expected %s", name)
}

func (p *parser) parseQuery(namespace string) (*Selector, error) {
	if _, err := p.expectKeyword("select"); err != nil {
		return nil, err
	}
	predicate, err := p.parseOr(0)
	if err != nil {
		return nil, err
	}
	predicate = normalize(predicate)

	var order []OrderItem
	if p.eatKeyword("order") {
		if _, err := p.expectKeyword("by"); err != nil {
			return nil, err
		}
		seen := make(map[string]bool)
		for {
			field, err := p.parseField()
			if err != nil {
				return nil, err
			}
			desc := false
			if p.eatKeyword("desc") {
				desc = true
			} else {
				p.eatKeyword("asc")
			}
			if seen[field.String()] {
				return nil, werrors.ParseAtf(p.peekPos(), "duplicate order field")
			}
			seen[field.String()] = true
			order = append(order, OrderItem{Field: field, Desc: desc})
			if p.peek().kind == tokComma {
				p.bump()
				continue
			}
			break
		}
	}

	var limitValue *uint64
	if p.eatKeyword("limit") {
		v, err := p.parseU64("limit")
		if err != nil {
			return nil, err
		}
		limitValue = &v
	}
	var offsetValue *uint64
	if p.eatKeyword("offset") {
		v, err := p.parseU64("offset")
		if err != nil {
			return nil, err
		}
		offsetValue = &v
	}
	if p.peek().kind != tokEOF {
		return nil, werrors.ParseAtf(p.peekPos(), "unexpected token after selector query")
	}
	if offsetValue != nil && *offsetValue == 0 {
		offsetValue = nil
	}

	return &Selector{
		Namespace: namespace,
		Predicate: predicate,
		Order:     order,
		Limit:     limitValue,
		Offset:    offsetValue,
	}, nil
}

func (p *parser) parseU64(clause string) (uint64, error) {
	tok := p.bump()
	if tok.kind != tokNumber {
		return 0, werrors.ParseAtf(tok.pos, "expected integer after `%s`", clause)
	}
	if strings.HasPrefix(tok.text, "-") {
		return 0, werrors.ParseAtf(tok.pos, "%s must not be negative", clause)
	}
	v, message := parseDigitsU64(tok.text)
	if message != "" {
		return 0, werrors.ParseAtf(tok.pos, "invalid %s: %s", clause, message)
	}
	return v, nil
}

func (p *parser) parseOr(depth uint32) (*Expr, error) {
	if err := p.checkDepth(depth); err != nil {
		return nil, err
	}
	first, err := p.parseAnd(depth)
	if err != nil {
		return nil, err
	}
	items := []*Expr{first}
	for p.eatKeyword("or") {
		item, err := p.parseAnd(depth)
		if err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	if len(items) == 1 {
		return items[0], nil
	}
	if err := p.bumpOperand(); err != nil {
		return nil, err
	}
	return &Expr{Kind: ExprOr, Items: items}, nil
}

func (p *parser) parseAnd(depth uint32) (*Expr, error) {
	if err := p.checkDepth(depth); err != nil {
		return nil, err
	}
	first, err := p.parseUnary(depth)
	if err != nil {
		return nil, err
	}
	items := []*Expr{first}
	for p.eatKeyword("and") {
		item, err := p.parseUnary(depth)
		if err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	if len(items) == 1 {
		return items[0], nil
	}
	if err := p.bumpOperand(); err != nil {
		return nil, err
	}
	return &Expr{Kind: ExprAnd, Items: items}, nil
}

func (p *parser) parseUnary(depth uint32) (*Expr, error) {
	if err := p.checkDepth(depth); err != nil {
		return nil, err
	}
	if p.eatKeyword("not") {
		inner, err := p.parseUnary(depth + 1)
		if err != nil {
			return nil, err
		}
		if err := p.bumpOperand(); err != nil {
			return nil, err
		}
		return &Expr{Kind: ExprNot, Inner: inner}, nil
	}
	return p.parsePrimary(depth)
}

func (p *parser) parsePrimary(depth uint32) (*Expr, error) {
	if err := p.checkDepth(depth); err != nil {
		return nil, err
	}
	if p.peek().kind == tokLParen {
		p.bump()
		inner, err := p.parseOr(depth + 1)
		if err != nil {
			return nil, err
		}
		if _, err := p.expectToken(tokRParen, "`)`"); err != nil {
			return nil, err
		}
		return inner, nil
	}
	if p.eatKeyword("true") {
		if err := p.bumpOperand(); err != nil {
			return nil, err
		}
		return &Expr{Kind: ExprTrue}, nil
	}
	if p.eatKeyword("false") {
		if err := p.bumpOperand(); err != nil {
			return nil, err
		}
		return &Expr{Kind: ExprFalse}, nil
	}
	if p.eatKeyword("has") {
		if _, err := p.expectToken(tokLParen, "`(`"); err != nil {
			return nil, err
		}
		field, err := p.parseField()
		if err != nil {
			return nil, err
		}
		if _, err := p.expectToken(tokRParen, "`)`"); err != nil {
			return nil, err
		}
		if err := p.bumpOperand(); err != nil {
			return nil, err
		}
		return &Expr{Kind: ExprHas, Field: field}, nil
	}
	if p.eatKeyword("tag") {
		if _, err := p.expectToken(tokLParen, "`(`"); err != nil {
			return nil, err
		}
		name, err := p.parseNameArgument("tag argument")
		if err != nil {
			return nil, err
		}
		if _, err := p.expectToken(tokRParen, "`)`"); err != nil {
			return nil, err
		}
		if err := p.bumpOperand(); err != nil {
			return nil, err
		}
		return &Expr{Kind: ExprTag, Name: name}, nil
	}
	if p.eatKeyword("relation") {
		if _, err := p.expectToken(tokLParen, "`(`"); err != nil {
			return nil, err
		}
		predicate, err := p.parseNameArgument("relation predicate")
		if err != nil {
			return nil, err
		}
		if _, err := p.expectToken(tokComma, "`,`"); err != nil {
			return nil, err
		}
		target, err := p.parseObjectRef()
		if err != nil {
			return nil, err
		}
		if _, err := p.expectToken(tokRParen, "`)`"); err != nil {
			return nil, err
		}
		if err := p.bumpOperand(); err != nil {
			return nil, err
		}
		return &Expr{Kind: ExprRelation, Name: predicate, Target: target}, nil
	}

	field, err := p.parseField()
	if err != nil {
		return nil, err
	}
	op, err := p.parseCmp()
	if err != nil {
		return nil, err
	}
	value, err := p.parseLiteral()
	if err != nil {
		return nil, err
	}
	if err := p.bumpOperand(); err != nil {
		return nil, err
	}
	return &Expr{Kind: ExprCompare, Field: field, Op: op, Value: value}, nil
}

func (p *parser) parseField() (Field, error) {
	tok := p.bump()
	if tok.kind != tokIdent {
		return Field{}, werrors.ParseAtf(tok.pos, "expected field name")
	}
	if isKeyword(tok.text) {
		return Field{}, werrors.ParseAtf(tok.pos, "reserved keyword `%s` cannot be a field", tok.text)
	}
	segments := []string{tok.text}
	for p.peek().kind == tokDot {
		p.bump()
		seg := p.bump()
		if seg.kind != tokIdent {
			return Field{}, werrors.ParseAtf(seg.pos, "expected field path segment")
		}
		if isKeyword(seg.text) {
			return Field{}, werrors.ParseAtf(seg.pos, "reserved keyword `%s` cannot be a field", seg.text)
		}
		segments = append(segments, seg.text)
	}
	if uint64(len(segments)) > p.limits.MaxFieldSegments {
		return Field{}, werrors.LimitExceededf(
			"field path has %d segments, limit %d", len(segments), p.limits.MaxFieldSegments)
	}
	return Field{Segments: segments}, nil
}

// parseNameArgument accepts a string literal or a non-keyword identifier. The
// grammar spells these arguments as strings, but the worked example in
// WEFT-SELECTORS §7 writes `relation(derived_from, ...)` bare; both forms have
// the same canonical encoding.
func (p *parser) parseNameArgument(what string) (string, error) {
	tok := p.bump()
	if tok.kind == tokStr {
		return tok.text, nil
	}
	if tok.kind == tokIdent && !isKeyword(tok.text) {
		return tok.text, nil
	}
	return "", werrors.ParseAtf(tok.pos, "expected string or identifier for %s", what)
}

func (p *parser) parseCmp() (CmpOp, error) {
	tok := p.bump()
	switch tok.kind {
	case tokEqEq:
		return OpEq, nil
	case tokNe:
		return OpNe, nil
	case tokLt:
		return OpLt, nil
	case tokLe:
		return OpLe, nil
	case tokGt:
		return OpGt, nil
	case tokGe:
		return OpGe, nil
	default:
		return 0, werrors.ParseAtf(tok.pos, "expected comparison operator")
	}
}

func (p *parser) parseLiteral() (Literal, error) {
	tok := p.bump()
	switch tok.kind {
	case tokStr:
		return StrLiteral(tok.text), nil
	case tokNumber:
		v, message := parseIntegerLiteral(tok.text)
		if message != "" {
			return Literal{}, werrors.ParseAtf(tok.pos, "invalid integer literal: %s", message)
		}
		return v, nil
	case tokIdent:
		switch tok.text {
		case "true":
			return BoolLiteral(true), nil
		case "false":
			return BoolLiteral(false), nil
		case "null":
			return NullLiteral(), nil
		}
	}
	return Literal{}, werrors.ParseAtf(
		tok.pos, "expected literal (string, int, bool, null)")
}

func (p *parser) parseObjectRef() (hash.Hash32, error) {
	tok := p.peek()
	if tok.kind == tokIdent && tok.text == "obj" &&
		p.index+1 < len(p.tokens) && p.tokens[p.index+1].kind == tokColon {
		p.bump()
		p.bump()
		return p.parseHexToken(p.bump())
	}
	return p.parseHexToken(p.bump())
}

func (p *parser) parseHexToken(tok token) (hash.Hash32, error) {
	if tok.kind != tokIdent && tok.kind != tokNumber {
		return hash.Hash32{}, werrors.ParseAtf(tok.pos, "expected 64-digit object reference")
	}
	if len(tok.text) != 64 || !allHex(tok.text) {
		return hash.Hash32{}, werrors.ParseAtf(
			tok.pos, "object reference must be exactly 64 hex digits")
	}
	h, err := hash.FromHex(tok.text)
	if err != nil {
		return hash.Hash32{}, werrors.ParseAtf(tok.pos, "invalid object reference")
	}
	return h, nil
}

func allHex(s string) bool {
	for i := 0; i < len(s); i++ {
		b := s[i]
		if !((b >= '0' && b <= '9') || (b >= 'a' && b <= 'f') || (b >= 'A' && b <= 'F')) {
			return false
		}
	}
	return true
}

func validIdent(s string) bool {
	if s == "" {
		return false
	}
	if !isAlpha(s[0]) && s[0] != '_' {
		return false
	}
	for i := 1; i < len(s); i++ {
		if !isAlphaNum(s[i]) && s[i] != '_' {
			return false
		}
	}
	return true
}

func parseDigitsU64(text string) (uint64, string) {
	if text == "" {
		return 0, "expected decimal digits"
	}
	var value uint64
	for i := 0; i < len(text); i++ {
		b := text[i]
		if b < '0' || b > '9' {
			return 0, "expected decimal digits"
		}
		d := uint64(b - '0')
		if value > (math.MaxUint64-d)/10 {
			return 0, "integer exceeds u64"
		}
		value = value*10 + d
	}
	return value, ""
}

func parseIntegerLiteral(text string) (Literal, string) {
	negative := false
	digits := text
	if strings.HasPrefix(text, "-") {
		negative = true
		digits = text[1:]
	}
	if digits == "" {
		return Literal{}, "expected decimal digits"
	}
	var magnitude uint64
	for i := 0; i < len(digits); i++ {
		b := digits[i]
		if b < '0' || b > '9' {
			return Literal{}, "expected decimal digits"
		}
		d := uint64(b - '0')
		if magnitude > (math.MaxUint64-d)/10 {
			return Literal{}, "integer is too large"
		}
		magnitude = magnitude*10 + d
	}
	if negative {
		if magnitude == 0 {
			return U64Literal(0), ""
		}
		if magnitude > 1<<63 {
			return Literal{}, "negative integer is below -2^63"
		}
		if magnitude == 1<<63 {
			return I64Literal(math.MinInt64), ""
		}
		return I64Literal(-int64(magnitude)), ""
	}
	return U64Literal(magnitude), ""
}

// normalize applies the only normalizations allowed by WEFT-SELECTORS §2:
// not true/false, double negation, and flattening nested same-operator
// and/or nodes. Operand order is preserved.
func normalize(e *Expr) *Expr {
	switch e.Kind {
	case ExprNot:
		inner := normalize(e.Inner)
		switch inner.Kind {
		case ExprTrue:
			return &Expr{Kind: ExprFalse}
		case ExprFalse:
			return &Expr{Kind: ExprTrue}
		case ExprNot:
			return inner.Inner
		default:
			return &Expr{Kind: ExprNot, Inner: inner}
		}
	case ExprAnd, ExprOr:
		flattened := make([]*Expr, 0, len(e.Items))
		for _, item := range e.Items {
			n := normalize(item)
			if n.Kind == e.Kind {
				flattened = append(flattened, n.Items...)
			} else {
				flattened = append(flattened, n)
			}
		}
		return &Expr{Kind: e.Kind, Items: flattened}
	default:
		return e
	}
}
