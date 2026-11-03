package modes

import (
	"encoding/binary"
	"math"

	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/varint"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

const (
	minMatch            = 8
	maxChain            = 4
	maxIndexedPositions = 1_000_000
	deltaOpCopy         = 0x00
	deltaOpInsert       = 0x01
)

type matchChain struct {
	length    uint8
	positions [maxChain]uint32
}

// EncodeDeltaOps builds a COPY/INSERT stream against base.
//
// The matcher indexes sampled 4-byte sequences of the base in a bounded hash
// table (at most one million indexed positions, four candidates per key) and
// greedily emits the longest match of at least 8 bytes. Each COPY reads the
// original base, never the output. The algorithm is a direct port of the Rust
// encoder, so byte-identical inputs produce byte-identical op streams.
func EncodeDeltaOps(target, base []byte, lim limits.Limits) ([]byte, error) {
	if err := limits.CheckLen(uint64(len(target)), lim.MaxDecodedBytes, "decoded data"); err != nil {
		return nil, err
	}
	out := make([]byte, 0)
	var ops uint64

	indexable := uint64(len(base)) <= math.MaxUint32 && len(target) >= 4
	stride := 1
	if len(base) > maxIndexedPositions {
		stride = len(base) / maxIndexedPositions
	}
	if stride < 1 {
		stride = 1
	}

	table := make(map[uint32]matchChain)
	if indexable && len(base) >= 4 {
		for pos := 0; pos+4 <= len(base); pos++ {
			if pos%stride != 0 {
				continue
			}
			key := binary.LittleEndian.Uint32(base[pos : pos+4])
			chain := table[key]
			if int(chain.length) < maxChain {
				chain.positions[chain.length] = uint32(pos)
				chain.length++
				table[key] = chain
			}
		}
	}

	literalStart := 0
	i := 0
	for i < len(target) {
		bestLen := 0
		bestPos := 0
		if indexable && len(target)-i >= 4 {
			key := binary.LittleEndian.Uint32(target[i : i+4])
			if chain, ok := table[key]; ok {
				for idx := 0; idx < int(chain.length); idx++ {
					basePos := int(chain.positions[idx])
					maxLen := min(len(base)-basePos, len(target)-i)
					l := 0
					for l < maxLen && base[basePos+l] == target[i+l] {
						l++
					}
					if l > bestLen {
						bestLen = l
						bestPos = basePos
					}
				}
			}
		}
		if bestLen >= minMatch {
			if literalStart < i {
				if err := emitInsert(target[literalStart:i], &out, &ops, lim); err != nil {
					return nil, err
				}
			}
			out = append(out, deltaOpCopy)
			out = varint.Append(out, uint64(bestPos))
			out = varint.Append(out, uint64(bestLen))
			if err := bumpDeltaOp(&ops, lim); err != nil {
				return nil, err
			}
			i += bestLen
			literalStart = i
		} else {
			i++
		}
	}
	if literalStart < len(target) {
		if err := emitInsert(target[literalStart:], &out, &ops, lim); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func bumpDeltaOp(ops *uint64, lim limits.Limits) error {
	if *ops == ^uint64(0) {
		return werrors.NewLimitExceeded("delta op count overflow")
	}
	*ops = *ops + 1
	if *ops > lim.MaxDeltaOps {
		return werrors.LimitExceededf(
			"delta op count %d exceeds limit %d", *ops, lim.MaxDeltaOps)
	}
	return nil
}

func emitInsert(b []byte, out *[]byte, ops *uint64, lim limits.Limits) error {
	if len(b) == 0 {
		return nil
	}
	*out = append(*out, deltaOpInsert)
	*out = varint.Append(*out, uint64(len(b)))
	*out = append(*out, b...)
	return bumpDeltaOp(ops, lim)
}

// DecodeDeltaOps decodes a COPY/INSERT stream against base.
func DecodeDeltaOps(body, base []byte, expectedLen uint64, lim limits.Limits) ([]byte, error) {
	if err := limits.CheckLen(expectedLen, lim.MaxDecodedBytes, "decoded data"); err != nil {
		return nil, err
	}
	expected, err := limits.ToInt(expectedLen, "decoded length")
	if err != nil {
		return nil, err
	}
	out := make([]byte, 0, min(expected, 1<<20))
	pos := 0
	var ops uint64

	for pos < len(body) {
		if err := bumpDeltaOp(&ops, lim); err != nil {
			return nil, err
		}
		opcode := body[pos]
		pos++
		switch opcode {
		case deltaOpCopy:
			offset, err := varint.Decode(body, &pos)
			if err != nil {
				return nil, err
			}
			length, err := varint.Decode(body, &pos)
			if err != nil {
				return nil, err
			}
			if length == 0 {
				return nil, werrors.NewMalformed("delta COPY has zero length")
			}
			end, err := limits.AddChecked(offset, length, "delta COPY")
			if err != nil {
				return nil, werrors.NewMalformed("delta COPY offset+length overflow")
			}
			if end > uint64(len(base)) {
				return nil, werrors.Malformedf(
					"delta COPY range %d..%d exceeds base length %d", offset, end, len(base))
			}
			newLen, err := limits.AddChecked(uint64(len(out)), length, "delta output")
			if err != nil {
				return nil, werrors.NewCorrupt("delta output length overflow")
			}
			if newLen > expectedLen {
				return nil, werrors.NewCorrupt("delta emits more bytes than declared decoded_len")
			}
			start, err := limits.ToInt(offset, "delta COPY offset")
			if err != nil {
				return nil, werrors.NewMalformed(err.Error())
			}
			n, err := limits.ToInt(length, "delta COPY length")
			if err != nil {
				return nil, werrors.NewMalformed(err.Error())
			}
			if start > len(base) || n > len(base)-start {
				return nil, werrors.NewMalformed("delta COPY range exceeds base")
			}
			out = append(out, base[start:start+n]...)
		case deltaOpInsert:
			length, err := varint.Decode(body, &pos)
			if err != nil {
				return nil, err
			}
			if length == 0 {
				return nil, werrors.NewMalformed("delta INSERT has zero length")
			}
			n, err := limits.ToInt(length, "delta INSERT length")
			if err != nil {
				return nil, werrors.NewMalformed(err.Error())
			}
			if n > len(body)-pos {
				return nil, werrors.NewMalformed("delta INSERT extends past body")
			}
			end := pos + n
			newLen, err := limits.AddChecked(uint64(len(out)), length, "delta output")
			if err != nil {
				return nil, werrors.NewCorrupt("delta output length overflow")
			}
			if newLen > expectedLen {
				return nil, werrors.NewCorrupt("delta emits more bytes than declared decoded_len")
			}
			out = append(out, body[pos:end]...)
			pos = end
		default:
			return nil, werrors.Malformedf("unknown delta opcode 0x%02x", opcode)
		}
	}
	if uint64(len(out)) != expectedLen {
		return nil, werrors.Corruptf(
			"delta produced %d bytes but decoded_len is %d", len(out), expectedLen)
	}
	return out, nil
}
