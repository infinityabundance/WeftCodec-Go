package conformance_test

import (
	"bytes"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/transport"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

type transportVector struct {
	Name         string   `json:"name"`
	Schema       string   `json:"schema"`
	Alignment    uint32   `json:"alignment"`
	SectionCount uint32   `json:"section_count"`
	TotalLen     uint64   `json:"total_len"`
	PayloadHash  string   `json:"payload_hash"`
	SectionRoles []string `json:"section_roles"`
	ContainerHex string   `json:"container_hex"`
	ExpectError  string   `json:"expect_error"`
	Reject       string   `json:"reject"`
}

// countReader serves a container and counts range reads, modelling the
// single-section path.
type countReader struct {
	container []byte
	reads     int
	bytes     uint64
}

func (r *countReader) ReadRange(offset, length uint64) ([]byte, error) {
	r.reads++
	r.bytes += length
	end := offset + length
	if end < offset || end > uint64(len(r.container)) {
		return nil, fmt.Errorf("range out of bounds")
	}
	out := make([]byte, length)
	copy(out, r.container[offset:end])
	return out, nil
}

// TestTransportVectors replays conformance/v1/transport. For valid vectors it
// verifies the container, proves the Go encoder reproduces the exact bytes from
// the enclosed frames, and fetches one section through a range reader. For
// rejects it checks the normative class.
func TestTransportVectors(t *testing.T) {
	files := jsonFiles(t, filepath.Join("v1", "transport"))
	if len(files) < 20 {
		t.Fatalf("expected at least 20 transport vectors, found %d", len(files))
	}
	lim := limits.Default()
	valid, rejects := 0, 0
	for _, path := range files {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			var vector transportVector
			loadJSON(t, path, &vector)
			container := mustHex(t, "container", vector.ContainerHex)

			if vector.ExpectError != "" {
				_, err := transport.Verify(container, lim)
				if err == nil {
					t.Fatalf("expected %s, verify succeeded", vector.ExpectError)
				}
				if got := string(werrors.ClassOf(err)); got != vector.ExpectError {
					t.Fatalf("expected class %s, got %s (%v)", vector.ExpectError, got, err)
				}
				rejects++
				return
			}

			parsed, err := transport.Verify(container, lim)
			if err != nil {
				t.Fatalf("verify: %v", err)
			}
			if parsed.Header.PayloadHash.Hex() != vector.PayloadHash {
				t.Fatalf("payload hash %s, want %s", parsed.Header.PayloadHash, vector.PayloadHash)
			}
			if parsed.Header.SectionCount != vector.SectionCount {
				t.Fatalf("section count %d, want %d", parsed.Header.SectionCount, vector.SectionCount)
			}
			if parsed.Header.TotalLen != vector.TotalLen || uint64(len(container)) != vector.TotalLen {
				t.Fatalf("total_len %d, want %d (bytes %d)",
					parsed.Header.TotalLen, vector.TotalLen, len(container))
			}

			// Re-encode the enclosed frames in table order at the same
			// alignment. The result must be byte-identical.
			items := make([]transport.Item, 0, len(parsed.Sections))
			for _, section := range parsed.Sections {
				start, _ := limits.ToInt(section.Offset, "offset")
				end, _ := limits.ToInt(section.End(), "end")
				items = append(items, transport.Item{
					Bytes: append([]byte(nil), container[start:end]...),
					Role:  section.Role,
				})
			}
			rebuilt, err := transport.Create(items, vector.Alignment, lim)
			if err != nil {
				t.Fatalf("re-encode: %v", err)
			}
			if !bytes.Equal(rebuilt, container) {
				t.Fatalf("re-encoded container differs (%d vs %d bytes)", len(rebuilt), len(container))
			}

			// Single-section path: header + table + exactly one section.
			reader := &countReader{container: container}
			headerBytes, err := reader.ReadRange(0, transport.HeaderLen)
			if err != nil {
				t.Fatalf("header read: %v", err)
			}
			header, err := transport.ReadHeader(headerBytes)
			if err != nil {
				t.Fatalf("header: %v", err)
			}
			tableBytes, err := reader.ReadRange(
				transport.HeaderLen, uint64(header.SectionCount)*transport.SectionEntryLen)
			if err != nil {
				t.Fatalf("table read: %v", err)
			}
			sections, err := transport.ReadSectionTable(header, tableBytes)
			if err != nil {
				t.Fatalf("table: %v", err)
			}
			sectionBytes, err := transport.FetchSection(reader, sections[0])
			if err != nil {
				t.Fatalf("fetch: %v", err)
			}
			if _, err := transport.VerifySectionBytes(sections[0], sectionBytes, lim); err != nil {
				t.Fatalf("section verify: %v", err)
			}
			if reader.reads != 3 {
				t.Fatalf("single-section path issued %d range reads, want 3", reader.reads)
			}
			if reader.bytes != transport.HeaderLen+uint64(header.SectionCount)*transport.SectionEntryLen+sections[0].Length {
				t.Fatalf("single-section path read %d bytes", reader.bytes)
			}
			if sections[0].Offset%uint64(sections[0].Alignment) != 0 {
				t.Fatalf("section offset %d is not aligned to %d", sections[0].Offset, sections[0].Alignment)
			}
			valid++
		})
	}
	if valid < 5 {
		t.Fatalf("expected at least 5 valid transport vectors, found %d", valid)
	}
	if rejects < 10 {
		t.Fatalf("expected at least 10 transport rejects, found %d", rejects)
	}
}
