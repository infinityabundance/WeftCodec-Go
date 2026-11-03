package selector

import (
	"strings"
	"unicode/utf8"

	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

type tokenKind uint8

const (
	tokIdent tokenKind = iota
	tokNumber
	tokStr
	tokLParen
	tokRParen
	tokComma
	tokDot
	tokColon
	tokEqEq
	tokNe
	tokLe
	tokGe
	tokLt
	tokGt
	tokEOF
)

type token struct {
	kind tokenKind
	text string
	pos  int
}

type lexer struct {
	src string
	pos int
}

func (l *lexer) peek() (byte, bool) {
	if l.pos < len(l.src) {
		return l.src[l.pos], true
	}
	return 0, false
}

func (l *lexer) bump() (byte, bool) {
	b, ok := l.peek()
	if ok {
		l.pos++
	}
	return b, ok
}

func (l *lexer) takeWhile(pred func(byte) bool) string {
	start := l.pos
	for {
		b, ok := l.peek()
		if !ok || !pred(b) {
			break
		}
		l.pos++
	}
	return l.src[start:l.pos]
}

func (l *lexer) tokenize() ([]token, error) {
	tokens := make([]token, 0, 16)
	for {
		tok, err := l.next()
		if err != nil {
			return nil, err
		}
		tokens = append(tokens, tok)
		if tok.kind == tokEOF {
			return tokens, nil
		}
	}
}

func (l *lexer) next() (token, error) {
	for {
		b, ok := l.peek()
		if !ok {
			return token{kind: tokEOF, pos: l.pos}, nil
		}
		if b == ' ' || b == '\t' || b == '\n' || b == '\r' {
			l.pos++
			continue
		}
		break
	}
	start := l.pos
	b, _ := l.peek()
	switch {
	case b == '(':
		l.pos++
		return token{kind: tokLParen, pos: start}, nil
	case b == ')':
		l.pos++
		return token{kind: tokRParen, pos: start}, nil
	case b == ',':
		l.pos++
		return token{kind: tokComma, pos: start}, nil
	case b == '.':
		l.pos++
		return token{kind: tokDot, pos: start}, nil
	case b == ':':
		l.pos++
		return token{kind: tokColon, pos: start}, nil
	case b == '=':
		l.pos++
		if next, ok := l.bump(); !ok || next != '=' {
			return token{}, werrors.ParseAtf(start, "expected `==`")
		}
		return token{kind: tokEqEq, pos: start}, nil
	case b == '!':
		l.pos++
		if next, ok := l.bump(); !ok || next != '=' {
			return token{}, werrors.ParseAtf(start, "expected `!=`")
		}
		return token{kind: tokNe, pos: start}, nil
	case b == '<':
		l.pos++
		if next, ok := l.peek(); ok && next == '=' {
			l.pos++
			return token{kind: tokLe, pos: start}, nil
		}
		return token{kind: tokLt, pos: start}, nil
	case b == '>':
		l.pos++
		if next, ok := l.peek(); ok && next == '=' {
			l.pos++
			return token{kind: tokGe, pos: start}, nil
		}
		return token{kind: tokGt, pos: start}, nil
	case b == '"':
		s, err := l.lexString()
		if err != nil {
			return token{}, err
		}
		return token{kind: tokStr, text: s, pos: start}, nil
	case b == '-':
		l.pos++
		if next, ok := l.peek(); !ok || next < '0' || next > '9' {
			return token{}, werrors.ParseAtf(start, "expected digits after `-`")
		}
		digits := l.takeWhile(isDigit)
		return token{kind: tokNumber, text: "-" + digits, pos: start}, nil
	case b >= '0' && b <= '9':
		text := l.takeWhile(func(c byte) bool { return isAlphaNum(c) || c == '_' })
		return token{kind: tokNumber, text: text, pos: start}, nil
	case isAlpha(b) || b == '_':
		text := l.takeWhile(func(c byte) bool { return isAlphaNum(c) || c == '_' })
		return token{kind: tokIdent, text: text, pos: start}, nil
	default:
		return token{}, werrors.ParseAtf(start, "unexpected character 0x%02x", b)
	}
}

func (l *lexer) lexString() (string, error) {
	start := l.pos
	l.pos++ // opening quote
	var out strings.Builder
	for {
		b, ok := l.peek()
		if !ok {
			return "", werrors.ParseAtf(start, "unterminated string")
		}
		switch {
		case b == '"':
			l.pos++
			return out.String(), nil
		case b == '\\':
			l.pos++
			if err := l.lexEscape(&out); err != nil {
				return "", err
			}
		case b < 0x20:
			return "", werrors.ParseAtf(l.pos, "raw control character in string")
		default:
			r, size := utf8.DecodeRuneInString(l.src[l.pos:])
			if r == utf8.RuneError && size == 1 {
				return "", werrors.ParseAtf(l.pos, "invalid UTF-8 in string")
			}
			out.WriteRune(r)
			l.pos += size
		}
	}
}

func (l *lexer) lexEscape(out *strings.Builder) error {
	pos := l.pos
	b, ok := l.bump()
	if !ok {
		return werrors.ParseAtf(pos, "unterminated escape")
	}
	switch b {
	case '"':
		out.WriteByte('"')
	case '\\':
		out.WriteByte('\\')
	case '/':
		out.WriteByte('/')
	case 'b':
		out.WriteByte('\u0008')
	case 'f':
		out.WriteByte('\u000c')
	case 'n':
		out.WriteByte('\n')
	case 'r':
		out.WriteByte('\r')
	case 't':
		out.WriteByte('\t')
	case 'u':
		high, err := l.readHex4()
		if err != nil {
			return err
		}
		var code uint32
		if high >= 0xD800 && high <= 0xDBFF {
			if next, ok := l.peek(); !ok || next != '\\' {
				return werrors.ParseAtf(pos, "high surrogate without low surrogate")
			}
			l.pos++
			if next, ok := l.bump(); !ok || next != 'u' {
				return werrors.ParseAtf(pos, "high surrogate without low surrogate")
			}
			low, err := l.readHex4()
			if err != nil {
				return err
			}
			if low < 0xDC00 || low > 0xDFFF {
				return werrors.ParseAtf(pos, "invalid low surrogate")
			}
			code = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
		} else if high >= 0xDC00 && high <= 0xDFFF {
			return werrors.ParseAtf(pos, "lone low surrogate")
		} else {
			code = high
		}
		r := rune(code)
		if !utf8.ValidRune(r) {
			return werrors.ParseAtf(pos, "invalid unicode scalar value")
		}
		out.WriteRune(r)
	default:
		return werrors.ParseAtf(pos, "invalid escape `\\%c`", b)
	}
	return nil
}

func (l *lexer) readHex4() (uint32, error) {
	start := l.pos
	var value uint32
	for i := 0; i < 4; i++ {
		b, ok := l.bump()
		if !ok {
			return 0, werrors.ParseAtf(start, "truncated \\u escape")
		}
		var digit uint32
		switch {
		case b >= '0' && b <= '9':
			digit = uint32(b - '0')
		case b >= 'a' && b <= 'f':
			digit = uint32(b-'a') + 10
		case b >= 'A' && b <= 'F':
			digit = uint32(b-'A') + 10
		default:
			return 0, werrors.ParseAtf(start, "invalid hex digit in \\u escape")
		}
		value = (value << 4) | digit
	}
	return value, nil
}

func isDigit(b byte) bool    { return b >= '0' && b <= '9' }
func isAlpha(b byte) bool    { return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') }
func isAlphaNum(b byte) bool { return isAlpha(b) || isDigit(b) }
