// Package transport implements the `weft.transport.v1` transport-aligned
// container (spec/WEFT-TRANSPORT.md).
//
// The container is 4096-byte header || 56-byte section entries || zero padding
// || complete unmodified WFT1 frames. Each section starts at a multiple of its
// declared alignment (512, 4096, or 65536) so a consumer can fetch exactly one
// section with one aligned range read.
//
// This is a layout, not a transport. No DMA, GPU-memory, RDMA, kernel-bypass,
// deployed-3FS, throughput, or latency claim follows from this package.
package transport

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

const (
	// Magic is the container magic.
	Magic = "WFTT"
	// FormatMajor and FormatMinor are the v1 versions.
	FormatMajor uint16 = 1
	FormatMinor uint16 = 0
	// HeaderLen is the fixed transport header length.
	HeaderLen = 4096
	// SectionEntryLen is one section-table entry length.
	SectionEntryLen = 56
	// MaxSections is the v1 section bound.
	MaxSections uint32 = 4096
	// DomainTransport is the container-level payload digest domain.
	DomainTransport = "weft/transport/v1"
)

// Alignments are the allowed section alignments.
var Alignments = [3]uint32{512, 4096, 65536}

var castagnoli = crc32.MakeTable(crc32.Castagnoli)

// Role is a coarse hint for why a section exists.
type Role uint8

const (
	RolePayload Role = iota
	RoleDependency
	RoleMetadata
	RoleProvenance
)

// Name returns the CLI spelling.
func (r Role) Name() string {
	switch r {
	case RolePayload:
		return "payload"
	case RoleDependency:
		return "dependency"
	case RoleMetadata:
		return "metadata"
	case RoleProvenance:
		return "provenance"
	default:
		return fmt.Sprintf("role(%d)", uint8(r))
	}
}

// ParseRole parses the CLI spelling.
func ParseRole(text string) (Role, error) {
	switch text {
	case "payload":
		return RolePayload, nil
	case "dependency":
		return RoleDependency, nil
	case "metadata":
		return RoleMetadata, nil
	case "provenance":
		return RoleProvenance, nil
	default:
		return 0, werrors.Malformedf("unknown transport role %q", text)
	}
}

// Item is one frame to place in a container.
type Item struct {
	Bytes []byte
	Role  Role
}

// Section is a validated section-table entry.
type Section struct {
	Offset    uint64
	Length    uint64
	Alignment uint32
	Mode      uint8
	Role      Role
	Root      hash.Hash32
}

// End is the end of the enclosed frame bytes.
func (s Section) End() uint64 { return s.Offset + s.Length }

// Header is the parsed fixed header.
type Header struct {
	Flags              uint32
	SectionCount       uint32
	SectionTableOffset uint64
	TotalLen           uint64
	PayloadHash        hash.Hash32
}

// Transport is a structurally validated container.
type Transport struct {
	Header   Header
	Sections []Section
	Bytes    []byte
}

// RangeReader is the byte-range source used for a single-section fetch. Wire it
// to the backend's BlobPlane.GetRange (or read_into for a shared region).
type RangeReader interface {
	ReadRange(offset, length uint64) ([]byte, error)
}

// AllowedAlignment reports whether alignment is a v1 alignment.
func AllowedAlignment(alignment uint32) bool {
	for _, allowed := range Alignments {
		if alignment == allowed {
			return true
		}
	}
	return false
}

func alignUp(value uint64, alignment uint32) (uint64, error) {
	if alignment == 0 || alignment&(alignment-1) != 0 {
		return 0, werrors.Malformedf("transport alignment %d is not a power of two", alignment)
	}
	a := uint64(alignment)
	remainder := value % a
	if remainder == 0 {
		return value, nil
	}
	sum := value + (a - remainder)
	if sum < value {
		return 0, werrors.NewLimitExceeded("transport alignment overflows u64")
	}
	return sum, nil
}

func tableEnd(count uint32) (uint64, error) {
	entries, err := limits.MulChecked(uint64(count), SectionEntryLen, "transport table")
	if err != nil {
		return 0, err
	}
	return limits.AddChecked(HeaderLen, entries, "transport table end")
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

// Create packs frames into a container at alignment. Section order is the
// supplied order; the output is deterministic.
func Create(frames []Item, alignment uint32, lim limits.Limits) ([]byte, error) {
	if len(frames) == 0 {
		return nil, werrors.NewMalformed("a transport container needs at least one section")
	}
	if uint32(len(frames)) > MaxSections {
		return nil, werrors.LimitExceededf(
			"transport section count %d exceeds the %d bound", len(frames), MaxSections)
	}
	if !AllowedAlignment(alignment) {
		return nil, werrors.Malformedf(
			"transport alignment %d is not one of 512, 4096, 65536", alignment)
	}
	end, err := tableEnd(uint32(len(frames)))
	if err != nil {
		return nil, err
	}
	cursor := end
	sections := make([]Section, 0, len(frames))
	lengths := make([]uint64, 0, len(frames))
	hasher := hash.NewDomainHasher(DomainTransport)
	for _, item := range frames {
		parsed, err := frame.Parse(item.Bytes, lim)
		if err != nil {
			return nil, err
		}
		cursor, err = alignUp(cursor, alignment)
		if err != nil {
			return nil, err
		}
		length := uint64(len(item.Bytes))
		if err := limits.CheckLen(length, lim.MaxFrameBytes, "transport section"); err != nil {
			return nil, err
		}
		sections = append(sections, Section{
			Offset:    cursor,
			Length:    length,
			Alignment: alignment,
			Mode:      uint8(parsed.Header.Mode),
			Role:      item.Role,
			Root:      parsed.Header.ContentHash,
		})
		lengths = append(lengths, length)
		hasher.Update(item.Bytes)
		cursor += length
		if cursor < length {
			return nil, werrors.NewLimitExceeded("transport section end overflows u64")
		}
	}
	total, err := alignUp(cursor, alignment)
	if err != nil {
		return nil, err
	}
	if err := limits.CheckLen(total, lim.MaxTotalFetchedBytes, "transport container"); err != nil {
		return nil, err
	}
	totalInt, err := limits.ToInt(total, "transport container")
	if err != nil {
		return nil, err
	}
	out := make([]byte, totalInt)
	copy(out[0:4], Magic)
	binary.LittleEndian.PutUint16(out[4:6], FormatMajor)
	binary.LittleEndian.PutUint16(out[6:8], FormatMinor)
	binary.LittleEndian.PutUint32(out[8:12], 0)
	binary.LittleEndian.PutUint32(out[12:16], uint32(len(frames)))
	binary.LittleEndian.PutUint64(out[16:24], HeaderLen)
	binary.LittleEndian.PutUint64(out[24:32], total)
	digest := hasher.Finalize()
	copy(out[32:64], digest.Bytes())
	for index, section := range sections {
		at := HeaderLen + index*SectionEntryLen
		binary.LittleEndian.PutUint64(out[at:at+8], section.Offset)
		binary.LittleEndian.PutUint64(out[at+8:at+16], section.Length)
		binary.LittleEndian.PutUint32(out[at+16:at+20], section.Alignment)
		out[at+20] = section.Mode
		out[at+21] = byte(section.Role)
		copy(out[at+24:at+56], section.Root.Bytes())
		start, err := limits.ToInt(section.Offset, "section offset")
		if err != nil {
			return nil, err
		}
		copy(out[start:start+int(lengths[index])], frames[index].Bytes)
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

// ReadHeader parses and validates the fixed header.
func ReadHeader(b []byte) (Header, error) {
	if len(b) < HeaderLen {
		return Header{}, werrors.Malformedf(
			"transport header is %d bytes, shorter than the %d-byte minimum", len(b), HeaderLen)
	}
	if string(b[0:4]) != Magic {
		return Header{}, werrors.NewMalformed("bad transport magic (expected WFTT)")
	}
	major := binary.LittleEndian.Uint16(b[4:6])
	minor := binary.LittleEndian.Uint16(b[6:8])
	if major != FormatMajor || minor != FormatMinor {
		return Header{}, werrors.UnsupportedVersionf(
			"unsupported transport version %d.%d (this reader implements 1.0)", major, minor)
	}
	flags := binary.LittleEndian.Uint32(b[8:12])
	if flags != 0 {
		return Header{}, werrors.UnsupportedFeaturef(
			"transport header flags 0x%08x are reserved in v1", flags)
	}
	count := binary.LittleEndian.Uint32(b[12:16])
	if count == 0 {
		return Header{}, werrors.NewMalformed("a transport container needs at least one section")
	}
	if err := limits.CheckLen(uint64(count), uint64(MaxSections), "transport section count"); err != nil {
		return Header{}, err
	}
	tableOffset := binary.LittleEndian.Uint64(b[16:24])
	if tableOffset != HeaderLen {
		return Header{}, werrors.Malformedf(
			"transport section_table_offset %d must be %d in v1", tableOffset, HeaderLen)
	}
	total := binary.LittleEndian.Uint64(b[24:32])
	end, err := tableEnd(count)
	if err != nil {
		return Header{}, err
	}
	if total < end {
		return Header{}, werrors.Malformedf(
			"transport total_len %d is below the table end %d", total, end)
	}
	payloadHash, err := hash.FromSlice(b[32:64])
	if err != nil {
		return Header{}, err
	}
	for _, value := range b[64:HeaderLen] {
		if value != 0 {
			return Header{}, werrors.NewMalformed("transport header reserved bytes must be zero")
		}
	}
	return Header{
		Flags:              flags,
		SectionCount:       count,
		SectionTableOffset: tableOffset,
		TotalLen:           total,
		PayloadHash:        payloadHash,
	}, nil
}

// ReadSectionTable parses and structurally validates the section table.
func ReadSectionTable(header Header, table []byte) ([]Section, error) {
	expected, err := limits.MulChecked(uint64(header.SectionCount), SectionEntryLen, "transport table")
	if err != nil {
		return nil, err
	}
	expectedInt, err := limits.ToInt(expected, "transport table")
	if err != nil {
		return nil, err
	}
	if len(table) != expectedInt {
		return nil, werrors.Malformedf(
			"transport table is %d bytes but %d sections require %d",
			len(table), header.SectionCount, expectedInt)
	}
	tableEndValue, err := tableEnd(header.SectionCount)
	if err != nil {
		return nil, err
	}
	sections := make([]Section, 0, header.SectionCount)
	previousEnd := tableEndValue
	for index := 0; index < int(header.SectionCount); index++ {
		at := index * SectionEntryLen
		offset := binary.LittleEndian.Uint64(table[at : at+8])
		length := binary.LittleEndian.Uint64(table[at+8 : at+16])
		alignment := binary.LittleEndian.Uint32(table[at+16 : at+20])
		mode := table[at+20]
		role := Role(table[at+21])
		if role > RoleProvenance {
			return nil, werrors.Malformedf(
				"transport section %d role %d is not defined in v1", index, role)
		}
		reserved := binary.LittleEndian.Uint16(table[at+22 : at+24])
		root, err := hash.FromSlice(table[at+24 : at+56])
		if err != nil {
			return nil, err
		}
		if reserved != 0 {
			return nil, werrors.Malformedf(
				"transport section %d reserved field must be zero", index)
		}
		if !AllowedAlignment(alignment) {
			return nil, werrors.Malformedf(
				"transport section %d alignment %d is not one of 512, 4096, 65536", index, alignment)
		}
		if offset%uint64(alignment) != 0 {
			return nil, werrors.Malformedf(
				"transport section %d offset %d is not a multiple of %d", index, offset, alignment)
		}
		if length == 0 {
			return nil, werrors.Malformedf("transport section %d length is zero", index)
		}
		end := offset + length
		if end < offset {
			return nil, werrors.Malformedf("transport section %d end overflows", index)
		}
		if offset < tableEndValue {
			return nil, werrors.Malformedf(
				"transport section %d offset %d lies inside the header or table", index, offset)
		}
		if end > header.TotalLen {
			return nil, werrors.Malformedf(
				"transport section %d ends at %d, past total_len %d", index, end, header.TotalLen)
		}
		if offset < previousEnd {
			last := sections[len(sections)-1]
			if offset == last.Offset {
				return nil, werrors.Malformedf(
					"transport section %d duplicates offset %d", index, offset)
			}
			return nil, werrors.Malformedf(
				"transport section %d overlaps the previous section", index)
		}
		previousEnd = end
		sections = append(sections, Section{
			Offset:    offset,
			Length:    length,
			Alignment: alignment,
			Mode:      mode,
			Role:      role,
			Root:      root,
		})
	}
	return sections, nil
}

// ParseMetadata validates structure, per-section frame prefixes (mode and
// content root), zero padding, and the final aligned length. It does not
// compute the container-level payload hash.
func ParseMetadata(b []byte, lim limits.Limits) (*Transport, error) {
	header, err := ReadHeader(b)
	if err != nil {
		return nil, err
	}
	if header.TotalLen != uint64(len(b)) {
		return nil, werrors.Malformedf(
			"transport total_len %d does not match the %d bytes supplied",
			header.TotalLen, len(b))
	}
	tableEndValue, err := tableEnd(header.SectionCount)
	if err != nil {
		return nil, err
	}
	tableEndInt, err := limits.ToInt(tableEndValue, "transport table end")
	if err != nil {
		return nil, err
	}
	sections, err := ReadSectionTable(header, b[HeaderLen:tableEndInt])
	if err != nil {
		return nil, err
	}
	for index, section := range sections {
		if err := limits.CheckLen(section.Length, lim.MaxFrameBytes, "transport section"); err != nil {
			return nil, err
		}
		start, err := limits.ToInt(section.Offset, "section offset")
		if err != nil {
			return nil, err
		}
		end, err := limits.ToInt(section.End(), "section end")
		if err != nil {
			return nil, err
		}
		frameHeader, err := prefixHeader(b[start:end], lim)
		if err != nil {
			return nil, err
		}
		if frameHeader.ContentHash != section.Root {
			return nil, werrors.Corruptf(
				"transport section %d root %s does not match the enclosed frame header %s",
				index, section.Root.Hex(), frameHeader.ContentHash.Hex())
		}
		if uint8(frameHeader.Mode) != section.Mode {
			return nil, werrors.Corruptf(
				"transport section %d mode %d does not match the enclosed frame mode %d",
				index, section.Mode, uint8(frameHeader.Mode))
		}
	}
	cursor := tableEndValue
	for index, section := range sections {
		if section.Offset < cursor {
			return nil, werrors.Malformedf(
				"transport section %d starts before the previous section ends", index)
		}
		start, err := limits.ToInt(cursor, "padding offset")
		if err != nil {
			return nil, err
		}
		end, err := limits.ToInt(section.Offset, "padding end")
		if err != nil {
			return nil, err
		}
		for _, value := range b[start:end] {
			if value != 0 {
				return nil, werrors.Malformedf(
					"transport padding before section %d is not zero", index)
			}
		}
		cursor = section.End()
	}
	tailStart, err := limits.ToInt(cursor, "padding offset")
	if err != nil {
		return nil, err
	}
	for _, value := range b[tailStart:] {
		if value != 0 {
			return nil, werrors.NewMalformed("transport trailing padding is not zero")
		}
	}
	last := sections[len(sections)-1]
	aligned, err := alignUp(last.End(), last.Alignment)
	if err != nil {
		return nil, err
	}
	if aligned != header.TotalLen {
		return nil, werrors.Malformedf(
			"transport total_len %d is not the aligned end %d of the last section",
			header.TotalLen, aligned)
	}
	return &Transport{Header: header, Sections: sections, Bytes: b}, nil
}

// VerifyPayload recomputes the container-level payload digest.
func VerifyPayload(t *Transport) error {
	hasher := hash.NewDomainHasher(DomainTransport)
	for index, section := range t.Sections {
		start, err := limits.ToInt(section.Offset, "section offset")
		if err != nil {
			return err
		}
		end, err := limits.ToInt(section.End(), "section end")
		if err != nil {
			return err
		}
		if end > len(t.Bytes) {
			return werrors.Corruptf("transport section %d is outside the container", index)
		}
		hasher.Update(t.Bytes[start:end])
	}
	computed := hasher.Finalize()
	if computed != t.Header.PayloadHash {
		return werrors.Corruptf(
			"transport payload_hash %s does not match the header's %s",
			computed.Hex(), t.Header.PayloadHash.Hex())
	}
	return nil
}

// Verify performs full verification: structure, every section's complete
// frame, and the container payload digest.
func Verify(b []byte, lim limits.Limits) (*Transport, error) {
	t, err := ParseMetadata(b, lim)
	if err != nil {
		return nil, err
	}
	for index, section := range t.Sections {
		start, err := limits.ToInt(section.Offset, "section offset")
		if err != nil {
			return nil, err
		}
		end, err := limits.ToInt(section.End(), "section end")
		if err != nil {
			return nil, err
		}
		parsed, err := frame.Parse(b[start:end], lim)
		if err != nil {
			return nil, err
		}
		if parsed.Header.ContentHash != section.Root {
			return nil, werrors.Corruptf(
				"transport section %d root does not match the decoded frame", index)
		}
		if uint8(parsed.Header.Mode) != section.Mode {
			return nil, werrors.Corruptf(
				"transport section %d mode does not match the decoded frame", index)
		}
	}
	if err := VerifyPayload(t); err != nil {
		return nil, err
	}
	return t, nil
}

// VerifySectionBytes verifies one fetched section. It cannot verify the
// container-level payload hash.
func VerifySectionBytes(section Section, b []byte, lim limits.Limits) (*frame.Frame, error) {
	if uint64(len(b)) != section.Length {
		return nil, werrors.Corruptf(
			"transport section is %d bytes but the table records %d", len(b), section.Length)
	}
	parsed, err := frame.Parse(b, lim)
	if err != nil {
		return nil, err
	}
	if parsed.Header.ContentHash != section.Root {
		return nil, werrors.Corruptf(
			"transport section root %s does not match the fetched frame %s",
			section.Root.Hex(), parsed.Header.ContentHash.Hex())
	}
	if uint8(parsed.Header.Mode) != section.Mode {
		return nil, werrors.Corruptf(
			"transport section mode %d does not match the fetched frame %d",
			section.Mode, uint8(parsed.Header.Mode))
	}
	return parsed, nil
}

// FetchSection issues one range read for exactly one section.
func FetchSection(reader RangeReader, section Section) ([]byte, error) {
	b, err := reader.ReadRange(section.Offset, section.Length)
	if err != nil {
		return nil, err
	}
	if uint64(len(b)) != section.Length {
		return nil, werrors.Corruptf(
			"transport section at %d returned %d bytes but the table records %d",
			section.Offset, len(b), section.Length)
	}
	return b, nil
}

// prefixHeader validates only the 96-byte frame header and its CRC, mirroring
// the Rust `frame::parse_prefix` use in ParseMetadata.
func prefixHeader(b []byte, lim limits.Limits) (frame.Header, error) {
	if len(b) < frame.HeaderLen {
		return frame.Header{}, werrors.Malformedf(
			"frame prefix is %d bytes, shorter than the %d-byte header", len(b), frame.HeaderLen)
	}
	if string(b[0:4]) != frame.Magic {
		return frame.Header{}, werrors.NewMalformed("bad frame magic (expected WFT1)")
	}
	if b[4] != frame.FormatMajor {
		return frame.Header{}, werrors.UnsupportedVersionf(
			"unsupported format major %d", b[4])
	}
	if b[5] != frame.FormatMinor {
		return frame.Header{}, werrors.UnsupportedVersionf(
			"unsupported format minor %d", b[5])
	}
	stored := binary.LittleEndian.Uint32(b[72:76])
	if crc32c(b[0:72], b[76:96]) != stored {
		return frame.Header{}, werrors.NewCorrupt("header CRC32C mismatch")
	}
	mode, err := frame.ModeFromU8(b[7])
	if err != nil {
		return frame.Header{}, err
	}
	contentHash, err := hash.FromSlice(b[40:72])
	if err != nil {
		return frame.Header{}, err
	}
	reserved0 := binary.LittleEndian.Uint16(b[38:40])
	reserved1 := binary.LittleEndian.Uint32(b[84:88])
	reserved2 := binary.LittleEndian.Uint64(b[88:96])
	if reserved0 != 0 || reserved1 != 0 || reserved2 != 0 {
		return frame.Header{}, werrors.NewMalformed("reserved header fields must be zero")
	}
	domain := b[6]
	if domain > frame.DomainStructured {
		return frame.Header{}, werrors.Malformedf("invalid domain value %d", domain)
	}
	flags := binary.LittleEndian.Uint16(b[8:10])
	const knownFlags = frame.FlagChunked | frame.FlagSelfContained | frame.FlagDictRaw |
		frame.FlagOriginalHash
	if flags&^uint16(knownFlags) != 0 {
		return frame.Header{}, werrors.UnsupportedFeaturef(
			"unknown critical flag bits 0x%04x", flags&^uint16(0x000f))
	}
	if flags&frame.FlagOriginalHash != 0 {
		return frame.Header{}, werrors.NewUnsupportedFeature("ORIGINAL_HASH flag is reserved in v1")
	}
	if flags&frame.FlagSelfContained != 0 {
		return frame.Header{}, werrors.NewUnsupportedFeature(
			"SELF_CONTAINED is reserved and not produced in v1")
	}
	features := binary.LittleEndian.Uint16(b[36:38])
	const knownFeatures = frame.FeatureDependencies | frame.FeatureChunkHashes
	if features&^uint16(knownFeatures) != 0 {
		return frame.Header{}, werrors.UnsupportedFeaturef(
			"unknown critical feature bits 0x%04x", features&^uint16(0x0003))
	}
	decodedLen := binary.LittleEndian.Uint64(b[12:20])
	depCount := binary.LittleEndian.Uint32(b[28:32])
	chunkCount := binary.LittleEndian.Uint32(b[32:36])
	if err := limits.CheckLen(decodedLen, lim.MaxDecodedBytes, "decoded data"); err != nil {
		return frame.Header{}, err
	}
	if err := limits.CheckLen(
		uint64(depCount), lim.MaxDependencyCount, "dependency count"); err != nil {
		return frame.Header{}, err
	}
	if err := limits.CheckLen(uint64(chunkCount), lim.MaxChunks, "chunk count"); err != nil {
		return frame.Header{}, err
	}
	depTable, err := limits.MulChecked(uint64(depCount), frame.DepEntryLen, "dependency table")
	if err != nil {
		return frame.Header{}, werrors.NewMalformed(err.Error())
	}
	chunkTable, err := limits.MulChecked(uint64(chunkCount), frame.ChunkEntryLen, "chunk table")
	if err != nil {
		return frame.Header{}, werrors.NewMalformed(err.Error())
	}
	expectedTable, err := limits.AddChecked(depTable, chunkTable, "frame table")
	if err != nil {
		return frame.Header{}, werrors.NewMalformed(err.Error())
	}
	tableLen := binary.LittleEndian.Uint64(b[76:84])
	if tableLen != expectedTable {
		return frame.Header{}, werrors.Malformedf(
			"table_len %d does not match %d dependencies and %d chunks",
			tableLen, depCount, chunkCount)
	}
	total, err := limits.AddChecked(frame.HeaderLen, tableLen, "frame prefix")
	if err != nil {
		return frame.Header{}, werrors.NewMalformed(err.Error())
	}
	if total > uint64(len(b)) {
		return frame.Header{}, werrors.Malformedf(
			"frame prefix is %d bytes, shorter than its %d-byte header and table", len(b), total)
	}
	chunked := flags&frame.FlagChunked != 0
	if chunked {
		if chunkCount == 0 {
			return frame.Header{}, werrors.NewMalformed("CHUNKED set with an empty chunk table")
		}
		if features&frame.FeatureChunkHashes == 0 {
			return frame.Header{}, werrors.NewMalformed(
				"CHUNKED set without the CHUNK_HASHES feature")
		}
	} else {
		if chunkCount != 0 {
			return frame.Header{}, werrors.NewMalformed(
				"chunk table present without the CHUNKED flag")
		}
		if features&frame.FeatureChunkHashes != 0 {
			return frame.Header{}, werrors.NewMalformed(
				"CHUNK_HASHES set without the CHUNKED flag")
		}
	}
	if depCount > 0 && features&frame.FeatureDependencies == 0 {
		return frame.Header{}, werrors.NewMalformed(
			"dependency table present without the DEPENDENCIES feature")
	}
	return frame.Header{
		FormatMajor: b[4],
		FormatMinor: b[5],
		Domain:      b[6],
		Mode:        mode,
		Flags:       binary.LittleEndian.Uint16(b[8:10]),
		Level:       binary.LittleEndian.Uint16(b[10:12]),
		DecodedLen:  binary.LittleEndian.Uint64(b[12:20]),
		BodyLen:     binary.LittleEndian.Uint64(b[20:28]),
		DepCount:    binary.LittleEndian.Uint32(b[28:32]),
		ChunkCount:  binary.LittleEndian.Uint32(b[32:36]),
		Features:    binary.LittleEndian.Uint16(b[36:38]),
		ContentHash: contentHash,
		TableLen:    binary.LittleEndian.Uint64(b[76:84]),
	}, nil
}

func crc32c(parts ...[]byte) uint32 {
	var running uint32
	for _, part := range parts {
		running = crc32.Update(running, castagnoli, part)
	}
	return running
}
