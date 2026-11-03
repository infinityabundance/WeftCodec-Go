package context

import (
	"fmt"
	"strings"

	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/selector"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// URI forms.
const (
	FormAuthoritative = "authoritative"
	FormDescriptor    = "descriptor"
)

// ParsedURI is a parsed weft:// URI. Authoritative URIs carry only the address;
// descriptors carry the parameters needed to rebuild the record.
type ParsedURI struct {
	Form           string
	Namespace      string
	ID             hash.Hash32
	SelectorSource string
	Snapshot       hash.Hash32
	Closure        []hash.Hash32
	CapabilityHash *hash.Hash32
	Budget         *Budget
	Resolver       string
}

// ParseContextURI parses either the authoritative or descriptive form.
func ParseContextURI(input string, lim limits.Limits) (*ParsedURI, error) {
	if uint64(len(input)) > lim.MaxURIBytes {
		return nil, werrors.LimitExceededf(
			"URI length %d exceeds limit %d", len(input), lim.MaxURIBytes)
	}
	if !strings.HasPrefix(input, URIPrefix) {
		return nil, werrors.NewParse("URI must start with `weft://`")
	}
	rest := input[len(URIPrefix):]
	if strings.ContainsRune(rest, '#') {
		return nil, werrors.NewParse("URI fragments are not permitted")
	}
	authority, pathAndQuery, ok := strings.Cut(rest, "/")
	if !ok {
		return nil, werrors.NewParse("URI is missing the `/c/` path")
	}
	if err := validateNamespace(authority); err != nil {
		return nil, asParse(err, "namespace")
	}
	if !strings.HasPrefix(pathAndQuery, "c/") {
		return nil, werrors.NewParse("URI path must start with `/c/`")
	}
	tail := pathAndQuery[2:]
	if tail == "" {
		return nil, werrors.NewParse("URI is missing the address after `/c/`")
	}
	path := tail
	query := ""
	hasQuery := false
	if index := strings.IndexByte(tail, '?'); index >= 0 {
		path = tail[:index]
		query = tail[index+1:]
		hasQuery = true
	}
	if path == "" {
		return nil, werrors.NewParse("URI is missing the address after `/c/`")
	}

	if at := strings.IndexByte(path, '@'); at >= 0 {
		encodedSelector := path[:at]
		snapshotHex := path[at+1:]
		if strings.ContainsRune(snapshotHex, '@') {
			return nil, werrors.NewParse("descriptor contains more than one raw `@`")
		}
		if encodedSelector == "" {
			return nil, werrors.NewParse("descriptor selector text is empty")
		}
		selectorSource, err := PercentDecode(encodedSelector)
		if err != nil {
			return nil, err
		}
		snapshot, err := parseLowerHex32(snapshotHex, "snapshot")
		if err != nil {
			return nil, err
		}
		var closure []hash.Hash32
		var capabilityHash *hash.Hash32
		var budget *Budget
		resolver := DefaultResolver
		if hasQuery {
			if query == "" {
				return nil, werrors.NewParse("empty URI query")
			}
			seen := map[string]bool{}
			for _, param := range strings.Split(query, "&") {
				if param == "" {
					return nil, werrors.NewParse("empty URI query parameter")
				}
				key, value, ok := strings.Cut(param, "=")
				if !ok {
					return nil, werrors.NewParse(
						fmt.Sprintf("query parameter `%s` has no `=`", param))
				}
				if seen[key] {
					return nil, werrors.NewParse(
						fmt.Sprintf("duplicate query parameter `%s`", key))
				}
				seen[key] = true
				switch key {
				case "closure":
					if value == "" {
						return nil, werrors.NewParse("closure parameter must not be empty")
					}
					for _, entry := range strings.Split(value, ",") {
						h, err := parseLowerHex32(entry, "closure entry")
						if err != nil {
							return nil, err
						}
						closure = append(closure, h)
					}
				case "cap":
					h, err := parseLowerHex32(value, "cap")
					if err != nil {
						return nil, err
					}
					capabilityHash = &h
				case "budget":
					budget, err = ParseBudget(value)
					if err != nil {
						return nil, err
					}
				case "resolver":
					if err := validateResolver(value); err != nil {
						return nil, asParse(err, "resolver")
					}
					resolver = value
				default:
					return nil, werrors.NewParse(
						fmt.Sprintf("unknown URI query parameter `%s`", key))
				}
			}
		}
		return &ParsedURI{
			Form:           FormDescriptor,
			Namespace:      authority,
			SelectorSource: selectorSource,
			Snapshot:       snapshot,
			Closure:        closure,
			CapabilityHash: capabilityHash,
			Budget:         budget,
			Resolver:       resolver,
		}, nil
	}

	if hasQuery {
		return nil, werrors.NewParse("the authoritative form does not accept a query")
	}
	id, err := parseLowerHex32(path, "ContextRootId")
	if err != nil {
		return nil, err
	}
	return &ParsedURI{
		Form:      FormAuthoritative,
		Namespace: authority,
		ID:        id,
	}, nil
}

func asParse(err error, what string) error {
	if typed, ok := err.(*werrors.Error); ok {
		return werrors.NewParse(what + ": " + typed.Msg)
	}
	return werrors.NewParse(what + ": " + err.Error())
}

// ToContextRoot builds the Context Root for a descriptor; it returns nil for an
// authoritative URI (an address alone does not reveal its record).
func (p *ParsedURI) ToContextRoot(lim limits.Limits) (*ContextRoot, error) {
	if p.Form == FormAuthoritative {
		return nil, nil
	}
	sel, err := selector.ParseWith(p.SelectorSource, p.Namespace, lim)
	if err != nil {
		return nil, err
	}
	causal, err := sortUniqueHashes(p.Closure, "causal closure")
	if err != nil {
		return nil, err
	}
	return NewContextRoot(sel, p.Snapshot, p.Resolver, causal, p.CapabilityHash, p.Budget)
}

// CanonicalURI returns the authoritative URI, canonicalizing descriptors.
func (p *ParsedURI) CanonicalURI(lim limits.Limits) (string, error) {
	if p.Form == FormAuthoritative {
		return fmt.Sprintf("%s%s/c/%s", URIPrefix, p.Namespace, p.ID.Hex()), nil
	}
	root, err := p.ToContextRoot(lim)
	if err != nil {
		return "", err
	}
	return root.AuthoritativeURI()
}
