// Package wcb implements Weft Canonical Bytes (WEFTCODEC-FORMAT §8).
//
// WCB is a strict deterministic subset used for structured frame bodies and
// selector canonicalization. Tags are fixed one-byte values, integers are
// minimal varints, map keys are text and are ordered by their full encoded key
// bytes (tag + length + bytes), compared as unsigned strings.
//
// Decoders reject non-canonical input (non-minimal varints, unsorted or
// duplicate keys, non-text keys, trailing bytes, unknown tags, invalid UTF-8)
// with NonCanonical; truncated input is Malformed.
package wcb

import (
	"bytes"
	"sort"
	"unicode/utf8"

	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/varint"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// Tags.
const (
	TagNull      byte = 0x00
	TagFalse     byte = 0x01
	TagTrue      byte = 0x02
	TagU64       byte = 0x03
	TagI64       byte = 0x04
	TagText      byte = 0x05
	TagBytes     byte = 0x06
	TagArray     byte = 0x07
	TagMap       byte = 0x08
	TagTimestamp byte = 0x09
)

// Kind enumerates WCB value kinds.
type Kind uint8

const (
	Null Kind = iota
	Bool
	U64
	I64
	Timestamp
	Text
	Bytes
	Array
	Map
)

// Value is a WCB value. Only the field(s) matching Kind are meaningful.
type Value struct {
	Kind  Kind
	Bool  bool
	U     uint64
	I     int64
	Str   string
	Bytes []byte
	Array []Value
	Map   map[string]Value
}

// Constructors.

func NullValue() Value             { return Value{Kind: Null} }
func BoolValue(v bool) Value       { return Value{Kind: Bool, Bool: v} }
func U64Value(v uint64) Value      { return Value{Kind: U64, U: v} }
func I64Value(v int64) Value       { return Value{Kind: I64, I: v} }
func TimestampValue(v int64) Value { return Value{Kind: Timestamp, I: v} }
func TextValue(v string) Value     { return Value{Kind: Text, Str: v} }
func BytesValue(v []byte) Value    { return Value{Kind: Bytes, Bytes: v} }
func ArrayValue(v []Value) Value   { return Value{Kind: Array, Array: v} }
func MapValue(v map[string]Value) Value {
	if v == nil {
		v = map[string]Value{}
	}
	return Value{Kind: Map, Map: v}
}

// Encode encodes v with default limits.
func EncodeDefault(v Value) ([]byte, error) { return Encode(v, limits.Default()) }

// Encode encodes v to canonical WCB bytes.
func Encode(v Value, lim limits.Limits) ([]byte, error) {
	e := encoder{limits: lim}
	if err := e.value(v, 0); err != nil {
		return nil, err
	}
	return e.out, nil
}

// DecodeDefault decodes canonical WCB bytes with default limits.
func DecodeDefault(b []byte) (Value, error) { return Decode(b, limits.Default()) }

// Decode decodes canonical WCB bytes, rejecting every non-canonical form.
func Decode(b []byte, lim limits.Limits) (Value, error) {
	if len(b) == 0 {
		return Value{}, werrors.NewMalformed("empty WCB value")
	}
	d := decoder{input: b, limits: lim}
	v, err := d.value(0)
	if err != nil {
		return Value{}, err
	}
	if d.pos != len(b) {
		return Value{}, werrors.NonCanonicalf(
			"trailing bytes after WCB value: %d of %d unconsumed", len(b)-d.pos, len(b))
	}
	return v, nil
}

type encoder struct {
	out    []byte
	limits limits.Limits
	items  uint64
}

func (e *encoder) bumpItem() error {
	if e.items == ^uint64(0) {
		return werrors.NewLimitExceeded("WCB item count overflow")
	}
	e.items++
	if e.items > e.limits.MaxStructuredItems {
		return werrors.LimitExceededf(
			"WCB item count %d exceeds limit %d", e.items, e.limits.MaxStructuredItems)
	}
	return nil
}

func (e *encoder) checkText(n int, what string) error {
	if uint64(n) > e.limits.MaxStringBytes {
		return werrors.LimitExceededf(
			"%s length %d exceeds limit %d", what, n, e.limits.MaxStringBytes)
	}
	return nil
}

func (e *encoder) enter(depth uint32) (uint32, error) {
	if depth == ^uint32(0) {
		return 0, werrors.NewLimitExceeded("WCB depth overflow")
	}
	next := depth + 1
	if uint64(next) > e.limits.MaxStructuredDepth {
		return 0, werrors.LimitExceededf(
			"WCB nesting depth %d exceeds limit %d", next, e.limits.MaxStructuredDepth)
	}
	return next, nil
}

func (e *encoder) value(v Value, depth uint32) error {
	if err := e.bumpItem(); err != nil {
		return err
	}
	switch v.Kind {
	case Null:
		e.out = append(e.out, TagNull)
	case Bool:
		if v.Bool {
			e.out = append(e.out, TagTrue)
		} else {
			e.out = append(e.out, TagFalse)
		}
	case U64:
		e.out = append(e.out, TagU64)
		e.out = varint.Append(e.out, v.U)
	case I64:
		e.out = append(e.out, TagI64)
		e.out = varint.Append(e.out, varint.ZigzagEncode(v.I))
	case Timestamp:
		e.out = append(e.out, TagTimestamp)
		e.out = varint.Append(e.out, varint.ZigzagEncode(v.I))
	case Text:
		if err := e.checkText(len(v.Str), "text"); err != nil {
			return err
		}
		e.out = append(e.out, TagText)
		e.out = varint.Append(e.out, uint64(len(v.Str)))
		e.out = append(e.out, v.Str...)
	case Bytes:
		if err := e.checkText(len(v.Bytes), "bytes"); err != nil {
			return err
		}
		e.out = append(e.out, TagBytes)
		e.out = varint.Append(e.out, uint64(len(v.Bytes)))
		e.out = append(e.out, v.Bytes...)
	case Array:
		child, err := e.enter(depth)
		if err != nil {
			return err
		}
		if uint64(len(v.Array)) > e.limits.MaxStructuredItems {
			return werrors.NewLimitExceeded("WCB array too large")
		}
		e.out = append(e.out, TagArray)
		e.out = varint.Append(e.out, uint64(len(v.Array)))
		for _, item := range v.Array {
			if err := e.value(item, child); err != nil {
				return err
			}
		}
	case Map:
		child, err := e.enter(depth)
		if err != nil {
			return err
		}
		if uint64(len(v.Map)) > e.limits.MaxStructuredItems {
			return werrors.NewLimitExceeded("WCB map too large")
		}
		type entry struct {
			key     string
			encoded []byte
		}
		entries := make([]entry, 0, len(v.Map))
		for key := range v.Map {
			if err := e.checkText(len(key), "map key"); err != nil {
				return err
			}
			encoded := make([]byte, 0, 1+varint.EncodedLen(uint64(len(key)))+len(key))
			encoded = append(encoded, TagText)
			encoded = varint.Append(encoded, uint64(len(key)))
			encoded = append(encoded, key...)
			entries = append(entries, entry{key: key, encoded: encoded})
		}
		sort.Slice(entries, func(i, j int) bool {
			return bytes.Compare(entries[i].encoded, entries[j].encoded) < 0
		})
		e.out = append(e.out, TagMap)
		e.out = varint.Append(e.out, uint64(len(entries)))
		for _, ent := range entries {
			e.out = append(e.out, ent.encoded...)
			val, ok := v.Map[ent.key]
			if !ok {
				return werrors.NewMalformed("map key vanished during encoding")
			}
			if err := e.value(val, child); err != nil {
				return err
			}
		}
	default:
		return werrors.Malformedf("unknown WCB value kind %d", v.Kind)
	}
	return nil
}

type decoder struct {
	input  []byte
	pos    int
	limits limits.Limits
	items  uint64
}

func (d *decoder) bumpItem() error {
	if d.items == ^uint64(0) {
		return werrors.NewLimitExceeded("WCB item count overflow")
	}
	d.items++
	if d.items > d.limits.MaxStructuredItems {
		return werrors.LimitExceededf(
			"WCB item count %d exceeds limit %d", d.items, d.limits.MaxStructuredItems)
	}
	return nil
}

func (d *decoder) readByte() (byte, error) {
	if d.pos < 0 || d.pos >= len(d.input) {
		return 0, werrors.NewMalformed("truncated WCB value")
	}
	b := d.input[d.pos]
	d.pos++
	return b, nil
}

func (d *decoder) readVarint() (uint64, error) {
	v, next, err := varint.DecodeAt(d.input, d.pos)
	if err != nil {
		return 0, err
	}
	d.pos = next
	return v, nil
}

func (d *decoder) remaining() int { return len(d.input) - d.pos }

func (d *decoder) take(n uint64, what string) ([]byte, error) {
	if n > d.limits.MaxStringBytes {
		return nil, werrors.LimitExceededf(
			"%s length %d exceeds limit %d", what, n, d.limits.MaxStringBytes)
	}
	ni, err := limits.ToInt(n, what)
	if err != nil {
		return nil, err
	}
	if ni > d.remaining() {
		return nil, werrors.NewMalformed("truncated WCB " + what)
	}
	start := d.pos
	d.pos += ni
	return d.input[start:d.pos], nil
}

func (d *decoder) enter(depth uint32) (uint32, error) {
	if depth == ^uint32(0) {
		return 0, werrors.NewLimitExceeded("WCB depth overflow")
	}
	next := depth + 1
	if uint64(next) > d.limits.MaxStructuredDepth {
		return 0, werrors.LimitExceededf(
			"WCB nesting depth %d exceeds limit %d", next, d.limits.MaxStructuredDepth)
	}
	return next, nil
}

func (d *decoder) checkCount(count, minEach uint64, what string) (int, error) {
	if count > d.limits.MaxStructuredItems {
		return 0, werrors.LimitExceededf(
			"WCB %s count %d exceeds limit %d", what, count, d.limits.MaxStructuredItems)
	}
	if count != 0 && minEach > ^uint64(0)/count {
		return 0, werrors.NewLimitExceeded("WCB count overflow")
	}
	minBytes := count * minEach
	if minBytes > uint64(d.remaining()) {
		return 0, werrors.Malformedf(
			"truncated WCB %s: count %d cannot fit in %d remaining bytes",
			what, count, d.remaining())
	}
	ni, err := limits.ToInt(count, what)
	if err != nil {
		return 0, err
	}
	return ni, nil
}

func (d *decoder) value(depth uint32) (Value, error) {
	if err := d.bumpItem(); err != nil {
		return Value{}, err
	}
	tag, err := d.readByte()
	if err != nil {
		return Value{}, err
	}
	switch tag {
	case TagNull:
		return NullValue(), nil
	case TagFalse:
		return BoolValue(false), nil
	case TagTrue:
		return BoolValue(true), nil
	case TagU64:
		u, err := d.readVarint()
		if err != nil {
			return Value{}, err
		}
		return U64Value(u), nil
	case TagI64:
		u, err := d.readVarint()
		if err != nil {
			return Value{}, err
		}
		return I64Value(varint.ZigzagDecode(u)), nil
	case TagTimestamp:
		u, err := d.readVarint()
		if err != nil {
			return Value{}, err
		}
		return TimestampValue(varint.ZigzagDecode(u)), nil
	case TagText:
		n, err := d.readVarint()
		if err != nil {
			return Value{}, err
		}
		raw, err := d.take(n, "text")
		if err != nil {
			return Value{}, err
		}
		if !utf8.Valid(raw) {
			return Value{}, werrors.NewNonCanonical("WCB text is not valid UTF-8")
		}
		return TextValue(string(raw)), nil
	case TagBytes:
		n, err := d.readVarint()
		if err != nil {
			return Value{}, err
		}
		raw, err := d.take(n, "bytes")
		if err != nil {
			return Value{}, err
		}
		cp := make([]byte, len(raw))
		copy(cp, raw)
		return BytesValue(cp), nil
	case TagArray:
		child, err := d.enter(depth)
		if err != nil {
			return Value{}, err
		}
		count, err := d.readVarint()
		if err != nil {
			return Value{}, err
		}
		n, err := d.checkCount(count, 1, "array")
		if err != nil {
			return Value{}, err
		}
		items := make([]Value, 0, n)
		for i := 0; i < n; i++ {
			item, err := d.value(child)
			if err != nil {
				return Value{}, err
			}
			items = append(items, item)
		}
		return ArrayValue(items), nil
	case TagMap:
		child, err := d.enter(depth)
		if err != nil {
			return Value{}, err
		}
		count, err := d.readVarint()
		if err != nil {
			return Value{}, err
		}
		n, err := d.checkCount(count, 2, "map")
		if err != nil {
			return Value{}, err
		}
		result := make(map[string]Value, n)
		var prev []byte
		for i := 0; i < n; i++ {
			keyStart := d.pos
			keyTag, err := d.readByte()
			if err != nil {
				return Value{}, err
			}
			if keyTag != TagText {
				return Value{}, werrors.NewNonCanonical("WCB map key is not a text value (tag 0x05)")
			}
			keyLen, err := d.readVarint()
			if err != nil {
				return Value{}, err
			}
			keyBytes, err := d.take(keyLen, "map key")
			if err != nil {
				return Value{}, err
			}
			keyEnd := d.pos
			current := d.input[keyStart:keyEnd]
			if prev != nil && bytes.Compare(current, prev) <= 0 {
				return Value{}, werrors.NewNonCanonical(
					"WCB map keys are duplicated or not sorted by encoded key bytes")
			}
			prev = current
			if !utf8.Valid(keyBytes) {
				return Value{}, werrors.NewNonCanonical("WCB map key is not valid UTF-8")
			}
			key := string(keyBytes)
			if _, exists := result[key]; exists {
				return Value{}, werrors.NewNonCanonical("duplicate WCB map key")
			}
			val, err := d.value(child)
			if err != nil {
				return Value{}, err
			}
			result[key] = val
		}
		return MapValue(result), nil
	default:
		return Value{}, werrors.NonCanonicalf("unknown WCB tag 0x%02x", tag)
	}
}
