package context_test

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"testing"

	weft "github.com/infinityabundance/WeftCodec-Go/context"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/selector"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

type budgetJSON struct {
	MaxResults *uint64 `json:"max_results"`
	MaxBytes   *uint64 `json:"max_bytes"`
	MaxDepth   *uint64 `json:"max_depth"`
	MaxWork    *uint64 `json:"max_work"`
}

func (b *budgetJSON) toBudget() *weft.Budget {
	if b == nil {
		return nil
	}
	return &weft.Budget{
		MaxResults: b.MaxResults,
		MaxBytes:   b.MaxBytes,
		MaxDepth:   b.MaxDepth,
		MaxWork:    b.MaxWork,
	}
}

type contextVector struct {
	Name                 string      `json:"name"`
	Namespace            string      `json:"namespace"`
	SelectorSource       string      `json:"selector_source"`
	SelectorCanonicalHex string      `json:"selector_canonical_hex"`
	SelectorID           string      `json:"selector_id"`
	Snapshot             string      `json:"snapshot"`
	Resolver             string      `json:"resolver"`
	Causal               []string    `json:"causal"`
	CapabilityHash       *string     `json:"capability_hash"`
	Budget               *budgetJSON `json:"budget"`
	Ordering             string      `json:"ordering"`
	CanonicalHex         string      `json:"canonical_hex"`
	ContextRootID        string      `json:"context_root_id"`
	URI                  string      `json:"uri"`
	Descriptor           string      `json:"descriptor"`
	RecordHex            string      `json:"record_hex"`
	ExpectError          string      `json:"expect_error"`
}

type causalVector struct {
	Name         string   `json:"name"`
	Parents      []string `json:"parents"`
	Closure      []string `json:"closure"`
	CanonicalHex string   `json:"canonical_hex"`
	ClosureHash  string   `json:"closure_hash"`
	RecordHex    string   `json:"record_hex"`
	ExpectError  string   `json:"expect_error"`
}

func vectorFiles(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("cannot read %s: %v", dir, err)
	}
	var files []string
	for _, entry := range entries {
		if !entry.IsDir() && filepath.Ext(entry.Name()) == ".json" {
			files = append(files, filepath.Join(dir, entry.Name()))
		}
	}
	sort.Strings(files)
	return files
}

func load[T any](t *testing.T, path string) T {
	t.Helper()
	var out T
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("cannot read %s: %v", path, err)
	}
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	return out
}

func mustHex(t *testing.T, text string) []byte {
	t.Helper()
	b, err := hex.DecodeString(text)
	if err != nil {
		t.Fatalf("bad hex: %v", err)
	}
	return b
}

func mustHash(t *testing.T, text string) hash.Hash32 {
	t.Helper()
	h, err := hash.FromHex(text)
	if err != nil {
		t.Fatalf("bad hash %q: %v", text, err)
	}
	return h
}

func TestContextVectors(t *testing.T) {
	files := vectorFiles(t, filepath.Join("..", "conformance", "v1", "context"))
	if len(files) < 15 {
		t.Fatalf("expected at least 15 context vectors, found %d", len(files))
	}
	lim := limits.Default()
	for _, path := range files {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			vector := load[contextVector](t, path)
			if vector.ExpectError != "" {
				_, err := weft.ContextRootFromCanonicalBytes(mustHex(t, vector.RecordHex), lim)
				if err == nil {
					t.Fatalf("%s: expected %s, decode succeeded", vector.Name, vector.ExpectError)
				}
				if got := string(werrors.ClassOf(err)); got != vector.ExpectError {
					t.Fatalf("%s: expected %s, got %s (%v)",
						vector.Name, vector.ExpectError, got, err)
				}
				return
			}

			root, err := weft.ContextRootFromCanonicalBytes(mustHex(t, vector.CanonicalHex), lim)
			if err != nil {
				t.Fatalf("%s: %v", vector.Name, err)
			}
			id, err := root.ID()
			if err != nil {
				t.Fatalf("%s: %v", vector.Name, err)
			}
			if id.Hex() != vector.ContextRootID {
				t.Fatalf("%s: id %s, want %s", vector.Name, id, vector.ContextRootID)
			}
			uri, err := root.AuthoritativeURI()
			if err != nil {
				t.Fatalf("%s: %v", vector.Name, err)
			}
			if uri != vector.URI {
				t.Fatalf("%s: uri %s, want %s", vector.Name, uri, vector.URI)
			}
			if root.Namespace != vector.Namespace {
				t.Fatalf("%s: namespace %s, want %s", vector.Name, root.Namespace, vector.Namespace)
			}
			if root.SelectorID.Hex() != vector.SelectorID {
				t.Fatalf("%s: selector id %s, want %s", vector.Name, root.SelectorID, vector.SelectorID)
			}
			if hex.EncodeToString(root.SelectorBytes) != vector.SelectorCanonicalHex {
				t.Fatalf("%s: selector bytes differ", vector.Name)
			}
			if root.Snapshot.Hex() != vector.Snapshot {
				t.Fatalf("%s: snapshot %s, want %s", vector.Name, root.Snapshot, vector.Snapshot)
			}
			if root.Resolver != vector.Resolver {
				t.Fatalf("%s: resolver %s, want %s", vector.Name, root.Resolver, vector.Resolver)
			}
			if root.Ordering != vector.Ordering {
				t.Fatalf("%s: ordering %s, want %s", vector.Name, root.Ordering, vector.Ordering)
			}
			if len(root.Causal) != len(vector.Causal) {
				t.Fatalf("%s: %d causal entries, want %d", vector.Name, len(root.Causal), len(vector.Causal))
			}
			for i, expected := range vector.Causal {
				if root.Causal[i].Hex() != expected {
					t.Fatalf("%s: causal[%d] %s, want %s", vector.Name, i, root.Causal[i], expected)
				}
			}
			if vector.CapabilityHash == nil {
				if root.CapabilityHash != nil {
					t.Fatalf("%s: unexpected capability hash", vector.Name)
				}
			} else if root.CapabilityHash == nil || root.CapabilityHash.Hex() != *vector.CapabilityHash {
				t.Fatalf("%s: capability hash mismatch", vector.Name)
			}
			if !root.Budget.Equal(vector.Budget.toBudget()) {
				t.Fatalf("%s: budget mismatch", vector.Name)
			}

			// Source reproduces the canonical selector.
			parsed, err := selector.ParseWith(vector.SelectorSource, root.Namespace, lim)
			if err != nil {
				t.Fatalf("%s: source: %v", vector.Name, err)
			}
			canonical, err := parsed.CanonicalBytes(lim)
			if err != nil {
				t.Fatalf("%s: %v", vector.Name, err)
			}
			if !bytesEqual(canonical, root.SelectorBytes) {
				t.Fatalf("%s: source does not reproduce selector bytes", vector.Name)
			}

			// Descriptor round-trip.
			parsedURI, err := weft.ParseContextURI(vector.Descriptor, lim)
			if err != nil {
				t.Fatalf("%s: descriptor: %v", vector.Name, err)
			}
			canonicalURI, err := parsedURI.CanonicalURI(lim)
			if err != nil {
				t.Fatalf("%s: canonicalize: %v", vector.Name, err)
			}
			if canonicalURI != vector.URI {
				t.Fatalf("%s: descriptor -> %s, want %s", vector.Name, canonicalURI, vector.URI)
			}
			rebuilt, err := parsedURI.ToContextRoot(lim)
			if err != nil {
				t.Fatalf("%s: rebuild: %v", vector.Name, err)
			}
			if rebuilt == nil {
				t.Fatalf("%s: descriptor produced no root", vector.Name)
			}
			rebuiltBytes, err := rebuilt.CanonicalBytes(lim)
			if err != nil {
				t.Fatalf("%s: %v", vector.Name, err)
			}
			origBytes := mustHex(t, vector.CanonicalHex)
			if !bytesEqual(rebuiltBytes, origBytes) {
				t.Fatalf("%s: descriptor record differs", vector.Name)
			}
		})
	}
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestCausalVectors(t *testing.T) {
	files := vectorFiles(t, filepath.Join("..", "conformance", "v1", "causal"))
	if len(files) < 10 {
		t.Fatalf("expected at least 10 causal vectors, found %d", len(files))
	}
	lim := limits.Default()
	for _, path := range files {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			vector := load[causalVector](t, path)
			if vector.ExpectError != "" {
				_, err := weft.CausalClosureFromCanonicalBytes(mustHex(t, vector.RecordHex), lim)
				if err == nil {
					t.Fatalf("%s: expected %s, decode succeeded", vector.Name, vector.ExpectError)
				}
				if got := string(werrors.ClassOf(err)); got != vector.ExpectError {
					t.Fatalf("%s: expected %s, got %s (%v)",
						vector.Name, vector.ExpectError, got, err)
				}
				return
			}
			closure, err := weft.CausalClosureFromCanonicalBytes(mustHex(t, vector.CanonicalHex), lim)
			if err != nil {
				t.Fatalf("%s: %v", vector.Name, err)
			}
			id, err := closure.ID()
			if err != nil {
				t.Fatalf("%s: %v", vector.Name, err)
			}
			if id.Hex() != vector.ClosureHash {
				t.Fatalf("%s: id %s, want %s", vector.Name, id, vector.ClosureHash)
			}
			if len(closure.Parents) != len(vector.Parents) {
				t.Fatalf("%s: parents length", vector.Name)
			}
			for i, expected := range vector.Parents {
				if closure.Parents[i].Hex() != expected {
					t.Fatalf("%s: parent[%d]", vector.Name, i)
				}
			}
			if len(closure.Closure) != len(vector.Closure) {
				t.Fatalf("%s: closure length", vector.Name)
			}
			for i, expected := range vector.Closure {
				if closure.Closure[i].Hex() != expected {
					t.Fatalf("%s: closure[%d]", vector.Name, i)
				}
			}
		})
	}
}

func TestIdentityMode(t *testing.T) {
	mode, err := weft.ParseIdentityMode("bytes_only")
	if err != nil || mode != weft.BytesOnly {
		t.Fatalf("bytes_only: %v %v", mode, err)
	}
	mode, err = weft.ParseIdentityMode("causal_exact")
	if err != nil || mode != weft.CausalExact {
		t.Fatalf("causal_exact: %v %v", mode, err)
	}
	if weft.CausalExact.String() != "causal_exact" {
		t.Fatalf("string form: %s", weft.CausalExact)
	}
	if _, err := weft.ParseIdentityMode("causalish"); err == nil {
		t.Fatalf("accepted unknown identity mode")
	} else if got := string(werrors.ClassOf(err)); got != "UnsupportedVersion" {
		t.Fatalf("class %s, want UnsupportedVersion", got)
	}
}

func TestBudgetAndCapability(t *testing.T) {
	budget, err := weft.ParseBudget("max_results=10,max_bytes=1024,max_depth=3")
	if err != nil {
		t.Fatalf("budget: %v", err)
	}
	if *budget.MaxResults != 10 || *budget.MaxDepth != 3 {
		t.Fatalf("budget fields wrong")
	}
	for _, bad := range []string{"", "max_results=0", "max_results=01", "x=1",
		"max_results=1,max_results=2", "max_results="} {
		if _, err := weft.ParseBudget(bad); err == nil {
			t.Fatalf("accepted bad budget %q", bad)
		}
	}

	expires := int64(123)
	scope, err := weft.NewCapabilityScope(
		"agent-1",
		[]string{"default"},
		[]hash.Hash32{mustHash(t, repeat("33", 32))},
		[]string{"derived_from"},
		budget,
		&expires,
	)
	if err != nil {
		t.Fatalf("capability: %v", err)
	}
	b, err := scope.CanonicalBytes(limits.Default())
	if err != nil {
		t.Fatalf("capability bytes: %v", err)
	}
	decoded, err := weft.CapabilityScopeFromCanonicalBytes(b, limits.Default())
	if err != nil {
		t.Fatalf("capability decode: %v", err)
	}
	id, err := scope.ID()
	if err != nil {
		t.Fatalf("capability id: %v", err)
	}
	decodedID, err := decoded.ID()
	if err != nil {
		t.Fatalf("capability id: %v", err)
	}
	if id != decodedID {
		t.Fatalf("capability id mismatch")
	}
	if _, err := weft.NewCapabilityScope("p", nil, nil, nil, nil, nil); err == nil {
		t.Fatalf("empty capability accepted")
	}
}

func TestRejectsBadURIs(t *testing.T) {
	lim := limits.Default()
	snapshot := repeat("22", 32)
	id := repeat("5a", 32)
	cases := []string{
		"http://default/c/" + id,
		"weft://default/x/" + id,
		"weft://default/c/" + upper(id),
		"weft://default/c/" + repeat("00", 32),
		"weft://default/c/" + id + "?x=1",
		"weft://bad ns/c/" + id,
		"weft://default/c/select%20true@" + repeat("00", 32),
		"weft://default/c/select%20true@" + snapshot + "?closure=bb",
		"weft://default/c/%41@" + snapshot,
		"weft://default/c/select%20true@" + snapshot + "?budget=max_results=0",
		"weft://default/c/select%20true@" + snapshot + "?budget=max_results=2,max_results=3",
		"weft://default/c/select%20true@" + snapshot + "#frag",
	}
	for _, candidate := range cases {
		if _, err := weft.ParseContextURI(candidate, lim); err == nil {
			t.Fatalf("accepted bad URI %q", candidate)
		}
	}
}

func repeat(s string, n int) string {
	out := ""
	for i := 0; i < n; i++ {
		out += s
	}
	return out
}

func upper(s string) string {
	b := []byte(s)
	for i := range b {
		if b[i] >= 'a' && b[i] <= 'f' {
			b[i] -= 'a' - 'A'
		}
	}
	return string(b)
}
