package frame

import "hash/crc32"

var castagnoli = crc32.MakeTable(crc32.Castagnoli)

// crc32c computes Castagnoli CRC32 over the concatenation of parts. Go's
// Castagnoli table matches the v1 init/final-XOR convention.
func crc32c(parts ...[]byte) uint32 {
	h := crc32.New(castagnoli)
	for _, p := range parts {
		_, _ = h.Write(p)
	}
	return h.Sum32()
}
