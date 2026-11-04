package generated_test

import (
	"testing"

	"github.com/infinityabundance/WeftCodec-Go/context"
	"github.com/infinityabundance/WeftCodec-Go/generated"
	"github.com/infinityabundance/WeftCodec-Go/hash"
)

// TestGeneratedConstantsMatchRuntime fails if the hand-written runtime layer
// drifts from the generated schema constants. The byte-level drift court is
// scripts/gen-types.sh --check.
func TestGeneratedConstantsMatchRuntime(t *testing.T) {
	cases := []struct {
		name string
		got  string
		want string
	}{
		{"context-root schema", generated.ContextRootSchema, context.ContextRootSchema},
		{"causal schema", generated.CausalSchema, context.CausalSchema},
		{"capability schema", generated.CapabilitySchema, context.CapabilitySchema},
		{"ordering", generated.OrderingSelectorThenObjectID, context.OrderingSelectorThenObjectID},
		{"resolver", generated.DefaultResolver, context.DefaultResolver},
		{"uri prefix", generated.URIPrefix, context.URIPrefix},
		{"context-root domain", generated.DomainContextRoot, hash.DomainContextRoot},
		{"causal domain", generated.DomainCausal, hash.DomainCausal},
		{"capability domain", generated.DomainCapability, hash.DomainCapability},
		{"selector domain", generated.DomainSelector, hash.DomainSelector},
	}
	for _, tc := range cases {
		if tc.got != tc.want {
			t.Errorf("%s: generated %q != runtime %q", tc.name, tc.got, tc.want)
		}
	}
	if generated.SchemaSourceSHA256 == "" {
		t.Fatal("generated schema source hash is empty")
	}
	if got := len(generated.ContextRootRecordFields); got != 9 {
		t.Fatalf("context-root field count %d, want 9", got)
	}
	if got := len(generated.BudgetRecordFields); got != 4 {
		t.Fatalf("budget field count %d, want 4", got)
	}
}
