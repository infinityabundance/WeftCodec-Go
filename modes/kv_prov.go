// Mode 19 KV_CACHE_PROV (WEFT-AI-CODEC §20, WEFT-KV-PROVENANCE).
//
// Mode 12 KV_CACHE freezes a KV page's bytes and geometry but not the model
// weights, adapters, tokenizer, attention configuration, positional treatment,
// cache layout, or sharing domain that produced it. A consumer that resolves a
// page by content hash alone can therefore reuse an incompatible prefix
// silently.
//
// Mode 19 binds the page to those dimensions. The payload is the mode-12
// payload verbatim; the only wire difference is the descriptor envelope. The
// page identity is
//
//	descriptor_root = H("weft/kv/prov/v1", WCB(descriptor))
//
// so two pages with identical token bytes but different provenance necessarily
// have different roots.
package modes

import (
	"bytes"
	"math"
	"sort"

	"github.com/infinityabundance/WeftCodec-Go/frame"
	"github.com/infinityabundance/WeftCodec-Go/hash"
	"github.com/infinityabundance/WeftCodec-Go/limits"
	"github.com/infinityabundance/WeftCodec-Go/wcb"
	"github.com/infinityabundance/WeftCodec-Go/werrors"
)

// KvProvSchema is the canonical descriptor schema.
const KvProvSchema = "weft.ai.kv_prov.v1"

// KV provenance identity domains.
const (
	DomainKvProv       = "weft/kv/prov/v1"
	DomainKvAdapterSet = "weft/kv/adapters/v1"
	DomainKvTokenizer  = "weft/kv/tokenizer/v1"
)

const kvRootLen = 32

// KvPositional is a positional-encoding treatment. v1 defines three names.
type KvPositional string

const (
	KvPositionalRope  KvPositional = "rope"
	KvPositionalAlibi KvPositional = "alibi"
	KvPositionalNone  KvPositional = "none"
)

// ParseKvPositional parses a canonical positional treatment.
func ParseKvPositional(name string) (KvPositional, error) {
	switch name {
	case string(KvPositionalRope):
		return KvPositionalRope, nil
	case string(KvPositionalAlibi):
		return KvPositionalAlibi, nil
	case string(KvPositionalNone):
		return KvPositionalNone, nil
	default:
		return "", werrors.Malformedf(
			"unknown KV provenance positional treatment %q (v1 allows rope, alibi, none)", name)
	}
}

// KvAttention is the attention configuration dimension. RopeThetaBits and
// RopeScaleBits are IEEE-754 binary64 bit patterns and must be finite.
type KvAttention struct {
	HeadCount     uint32
	HeadDim       uint32
	SlidingWindow uint64
	RopeThetaBits uint64
	RopeScaleBits uint64
}

func (a KvAttention) validate() error {
	if a.HeadCount == 0 {
		return werrors.NewMalformed(
			"KV provenance attention head_count must be greater than zero")
	}
	if a.HeadDim == 0 {
		return werrors.NewMalformed(
			"KV provenance attention head_dim must be greater than zero")
	}
	if math.IsInf(math.Float64frombits(a.RopeThetaBits), 0) ||
		math.IsNaN(math.Float64frombits(a.RopeThetaBits)) {
		return werrors.NewMalformed(
			"KV provenance attention rope_theta_bits must be a finite binary64")
	}
	if math.IsInf(math.Float64frombits(a.RopeScaleBits), 0) ||
		math.IsNaN(math.Float64frombits(a.RopeScaleBits)) {
		return werrors.NewMalformed(
			"KV provenance attention rope_scale_bits must be a finite binary64")
	}
	return nil
}

func (a KvAttention) value() wcb.Value {
	return wcb.MapValue(map[string]wcb.Value{
		"head_count":      wcb.U64Value(uint64(a.HeadCount)),
		"head_dim":        wcb.U64Value(uint64(a.HeadDim)),
		"sliding_window":  wcb.U64Value(a.SlidingWindow),
		"rope_theta_bits": wcb.U64Value(a.RopeThetaBits),
		"rope_scale_bits": wcb.U64Value(a.RopeScaleBits),
	})
}

// KvLayout is the cache-layout dimension.
type KvLayout struct {
	Dtype      KvDtype
	PageRows   uint64
	LayerCount uint64
}

func (l KvLayout) validate() error {
	if l.PageRows == 0 {
		return werrors.NewMalformed("KV provenance layout page_rows must be at least 1")
	}
	if l.LayerCount == 0 {
		return werrors.NewMalformed("KV provenance layout layer_count must be at least 1")
	}
	return nil
}

func (l KvLayout) value() wcb.Value {
	return wcb.MapValue(map[string]wcb.Value{
		"dtype":       wcb.TextValue(l.Dtype.Name),
		"page_rows":   wcb.U64Value(l.PageRows),
		"layer_count": wcb.U64Value(l.LayerCount),
	})
}

// KvExtension is one explicit extension dimension.
type KvExtension struct {
	Name  string
	Value wcb.Value
}

func (e KvExtension) value() wcb.Value {
	return wcb.MapValue(map[string]wcb.Value{
		"name":  wcb.TextValue(e.Name),
		"value": e.Value,
	})
}

// KvProvenanceInput is caller-supplied provenance. Adapters and extensions may
// be in any order; Canonicalize sorts them and derives the set/tokenizer roots.
type KvProvenanceInput struct {
	ModelRoot          hash.Hash32
	AdapterRoots       []hash.Hash32
	TokenizerScheme    string
	TokenizerVocabSize uint64
	Attention          KvAttention
	Positional         KvPositional
	Layout             KvLayout
	Salt               hash.Hash32
	Extensions         []KvExtension
}

// WithAdapter appends one adapter root.
func (p KvProvenanceInput) WithAdapter(root hash.Hash32) KvProvenanceInput {
	p.AdapterRoots = append(append([]hash.Hash32(nil), p.AdapterRoots...), root)
	return p
}

// WithExtension appends one extension dimension.
func (p KvProvenanceInput) WithExtension(name string, value wcb.Value) KvProvenanceInput {
	p.Extensions = append(append([]KvExtension(nil), p.Extensions...), KvExtension{Name: name, Value: value})
	return p
}

// KvProvenance holds canonical (sorted, derived) provenance dimensions.
type KvProvenance struct {
	ModelRoot          hash.Hash32
	AdapterRoots       []hash.Hash32
	AdapterSetRoot     hash.Hash32
	TokenizerScheme    string
	TokenizerVocabSize uint64
	TokenizerRoot      hash.Hash32
	Attention          KvAttention
	Positional         KvPositional
	Layout             KvLayout
	Salt               hash.Hash32
	Extensions         []KvExtension
}

func isZeroHash(h hash.Hash32) bool {
	var zero hash.Hash32
	return h == zero
}

// AdapterSetRoot is H("weft/kv/adapters/v1", WCB(sorted adapter roots)).
func AdapterSetRoot(roots []hash.Hash32) (hash.Hash32, error) {
	items := make([]wcb.Value, 0, len(roots))
	for _, root := range roots {
		items = append(items, wcb.BytesValue(root.Bytes()))
	}
	encoded, err := wcb.EncodeDefault(wcb.ArrayValue(items))
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(DomainKvAdapterSet, encoded), nil
}

// TokenizerRoot is H("weft/kv/tokenizer/v1", WCB({scheme, vocab_size})).
func TokenizerRoot(scheme string, vocabSize uint64) (hash.Hash32, error) {
	value := wcb.MapValue(map[string]wcb.Value{
		"scheme":     wcb.TextValue(scheme),
		"vocab_size": wcb.U64Value(vocabSize),
	})
	encoded, err := wcb.EncodeDefault(value)
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(DomainKvTokenizer, encoded), nil
}

// Canonicalize sorts and derives the canonical provenance dimensions.
func (p KvProvenanceInput) Canonicalize() (KvProvenance, error) {
	if isZeroHash(p.ModelRoot) {
		return KvProvenance{}, werrors.NewMalformed(
			"KV provenance model_root must not be all zero")
	}
	adapters := append([]hash.Hash32(nil), p.AdapterRoots...)
	sort.Slice(adapters, func(i, j int) bool {
		return bytes.Compare(adapters[i].Bytes(), adapters[j].Bytes()) < 0
	})
	for _, root := range adapters {
		if isZeroHash(root) {
			return KvProvenance{}, werrors.NewMalformed(
				"KV provenance adapter roots must not be all zero")
		}
	}
	for index := 1; index < len(adapters); index++ {
		if adapters[index-1] == adapters[index] {
			return KvProvenance{}, werrors.NewMalformed(
				"KV provenance adapter roots contain a duplicate")
		}
	}
	if p.TokenizerScheme == "" {
		return KvProvenance{}, werrors.NewMalformed(
			"KV provenance tokenizer scheme must not be empty")
	}
	if p.TokenizerVocabSize == 0 {
		return KvProvenance{}, werrors.NewMalformed(
			"KV provenance tokenizer vocab_size must be at least 1")
	}
	if err := p.Attention.validate(); err != nil {
		return KvProvenance{}, err
	}
	if err := p.Layout.validate(); err != nil {
		return KvProvenance{}, err
	}
	extensions := append([]KvExtension(nil), p.Extensions...)
	sort.Slice(extensions, func(i, j int) bool {
		return extensions[i].Name < extensions[j].Name
	})
	for _, extension := range extensions {
		if extension.Name == "" {
			return KvProvenance{}, werrors.NewMalformed(
				"KV provenance extension name must not be empty")
		}
	}
	for index := 1; index < len(extensions); index++ {
		if extensions[index-1].Name == extensions[index].Name {
			return KvProvenance{}, werrors.Malformedf(
				"KV provenance extension %q is duplicated", extensions[index].Name)
		}
	}
	adapterSetRoot, err := AdapterSetRoot(adapters)
	if err != nil {
		return KvProvenance{}, err
	}
	tokenizerRoot, err := TokenizerRoot(p.TokenizerScheme, p.TokenizerVocabSize)
	if err != nil {
		return KvProvenance{}, err
	}
	return KvProvenance{
		ModelRoot:          p.ModelRoot,
		AdapterRoots:       adapters,
		AdapterSetRoot:     adapterSetRoot,
		TokenizerScheme:    p.TokenizerScheme,
		TokenizerVocabSize: p.TokenizerVocabSize,
		TokenizerRoot:      tokenizerRoot,
		Attention:          p.Attention,
		Positional:         p.Positional,
		Layout:             p.Layout,
		Salt:               p.Salt,
		Extensions:         extensions,
	}, nil
}

// KvProvDescriptor is the complete mode-19 descriptor.
type KvProvDescriptor struct {
	Page        KvCacheMeta
	Provenance  KvProvenance
	PayloadHash hash.Hash32
}

func adapterRootValues(roots []hash.Hash32) wcb.Value {
	items := make([]wcb.Value, 0, len(roots))
	for _, root := range roots {
		items = append(items, wcb.BytesValue(root.Bytes()))
	}
	return wcb.ArrayValue(items)
}

func extensionValues(extensions []KvExtension) wcb.Value {
	items := make([]wcb.Value, 0, len(extensions))
	for _, extension := range extensions {
		items = append(items, extension.value())
	}
	return wcb.ArrayValue(items)
}

func (d KvProvDescriptor) value() wcb.Value {
	p := d.Provenance
	return wcb.MapValue(map[string]wcb.Value{
		"schema":           wcb.TextValue(KvProvSchema),
		"page":             kvCacheDescriptorValue(d.Page),
		"model_root":       wcb.BytesValue(p.ModelRoot.Bytes()),
		"adapter_roots":    adapterRootValues(p.AdapterRoots),
		"adapter_set_root": wcb.BytesValue(p.AdapterSetRoot.Bytes()),
		"tokenizer": wcb.MapValue(map[string]wcb.Value{
			"scheme":     wcb.TextValue(p.TokenizerScheme),
			"vocab_size": wcb.U64Value(p.TokenizerVocabSize),
		}),
		"tokenizer_root": wcb.BytesValue(p.TokenizerRoot.Bytes()),
		"attention":      p.Attention.value(),
		"positional":     wcb.TextValue(string(p.Positional)),
		"layout":         p.Layout.value(),
		"salt":           wcb.BytesValue(p.Salt.Bytes()),
		"extensions":     extensionValues(p.Extensions),
		"payload_hash":   wcb.BytesValue(d.PayloadHash.Bytes()),
	})
}

// DescriptorValue returns the canonical descriptor value. Exposed for
// conformance tooling; CanonicalBytes is what goes on the wire.
func (d KvProvDescriptor) DescriptorValue() wcb.Value { return d.value() }

// CanonicalBytes returns canonical WCB descriptor bytes.
func (d KvProvDescriptor) CanonicalBytes(lim limits.Limits) ([]byte, error) {
	return wcb.Encode(d.value(), lim)
}

// Root is H("weft/kv/prov/v1", canonical bytes).
func (d KvProvDescriptor) Root(lim limits.Limits) (hash.Hash32, error) {
	encoded, err := d.CanonicalBytes(lim)
	if err != nil {
		return hash.Hash32{}, err
	}
	return hash.Domain(DomainKvProv, encoded), nil
}

// Validate checks every dimension and cross-checks it against the frozen page.
func (d KvProvDescriptor) Validate() error {
	p := d.Provenance
	if err := p.Attention.validate(); err != nil {
		return err
	}
	if err := p.Layout.validate(); err != nil {
		return err
	}
	if isZeroHash(p.ModelRoot) {
		return werrors.NewMalformed("KV provenance model_root must not be all zero")
	}
	for _, root := range p.AdapterRoots {
		if isZeroHash(root) {
			return werrors.NewMalformed(
				"KV provenance adapter roots must not be all zero")
		}
	}
	for index := 1; index < len(p.AdapterRoots); index++ {
		if bytes.Compare(p.AdapterRoots[index-1].Bytes(), p.AdapterRoots[index].Bytes()) >= 0 {
			return werrors.NewMalformed(
				"KV provenance adapter roots are not strictly sorted and unique")
		}
	}
	adapterSet, err := AdapterSetRoot(p.AdapterRoots)
	if err != nil {
		return err
	}
	if adapterSet != p.AdapterSetRoot {
		return werrors.NewMalformed(
			"KV provenance adapter_set_root does not match the adapter roots")
	}
	if p.TokenizerScheme == "" || p.TokenizerVocabSize == 0 {
		return werrors.NewMalformed(
			"KV provenance tokenizer descriptor must have a scheme and vocab_size >= 1")
	}
	tokenizer, err := TokenizerRoot(p.TokenizerScheme, p.TokenizerVocabSize)
	if err != nil {
		return err
	}
	if tokenizer != p.TokenizerRoot {
		return werrors.NewMalformed(
			"KV provenance tokenizer_root does not match the tokenizer descriptor")
	}
	for index := 1; index < len(p.Extensions); index++ {
		if p.Extensions[index-1].Name >= p.Extensions[index].Name {
			return werrors.NewMalformed(
				"KV provenance extensions are not strictly sorted with unique names")
		}
	}
	for _, extension := range p.Extensions {
		if extension.Name == "" {
			return werrors.NewMalformed(
				"KV provenance extension name must not be empty")
		}
	}
	if p.Layout.Dtype != d.Page.Dtype {
		return werrors.NewMalformed(
			"KV provenance layout dtype disagrees with the page dtype")
	}
	if d.Page.SeqStart >= d.Page.SeqEnd {
		return werrors.NewMalformed("KV_CACHE_PROV page sequence range underflows")
	}
	rows := d.Page.SeqEnd - d.Page.SeqStart
	if p.Layout.PageRows != rows {
		return werrors.NewMalformed(
			"KV provenance layout page_rows disagrees with the page token range")
	}
	if p.Attention.HeadCount != d.Page.HeadCount {
		return werrors.NewMalformed(
			"KV provenance attention head_count disagrees with the page")
	}
	if p.Attention.HeadDim != d.Page.HeadDim {
		return werrors.NewMalformed(
			"KV provenance attention head_dim disagrees with the page")
	}
	if uint64(d.Page.Layer) >= p.Layout.LayerCount {
		return werrors.NewMalformed(
			"KV_CACHE_PROV page layer is not below the layout layer_count")
	}
	if isZeroHash(d.PayloadHash) {
		return werrors.NewMalformed("KV provenance payload_hash must not be all zero")
	}
	return nil
}

// NewKvProvDescriptor builds and validates a descriptor for an encoded page.
func NewKvProvDescriptor(page KvCacheMeta, provenance KvProvenance, elements []byte) (KvProvDescriptor, error) {
	descriptor := KvProvDescriptor{
		Page:        page,
		Provenance:  provenance,
		PayloadHash: hash.Domain(hash.DomainChunk, elements),
	}
	if err := descriptor.Validate(); err != nil {
		return KvProvDescriptor{}, err
	}
	return descriptor, nil
}

func kvProvExactKeys(m map[string]wcb.Value, expected []string) error {
	for key := range m {
		found := false
		for _, want := range expected {
			if key == want {
				found = true
				break
			}
		}
		if !found {
			return werrors.Malformedf("KV_CACHE_PROV has unknown key %q", key)
		}
	}
	for _, want := range expected {
		if _, ok := m[want]; !ok {
			return werrors.Malformedf("KV_CACHE_PROV is missing key %q", want)
		}
	}
	return nil
}

func kvProvMapValue(m map[string]wcb.Value, key, what string) (map[string]wcb.Value, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.Map {
		return nil, werrors.Malformedf("%s %q must be a WCB map", what, key)
	}
	return v.Map, nil
}

func kvProvText(m map[string]wcb.Value, key, what string) (string, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.Text {
		return "", werrors.Malformedf("%s %q must be text", what, key)
	}
	return v.Str, nil
}

func kvProvU64(m map[string]wcb.Value, key, what string) (uint64, error) {
	v, ok := m[key]
	if !ok || v.Kind != wcb.U64 {
		return 0, werrors.Malformedf("%s %q must be u64", what, key)
	}
	return v.U, nil
}

func kvProvRoot(v wcb.Value, what string) (hash.Hash32, error) {
	if v.Kind != wcb.Bytes {
		return hash.Hash32{}, werrors.Malformedf("%s must be bytes", what)
	}
	if len(v.Bytes) != kvRootLen {
		return hash.Hash32{}, werrors.Malformedf(
			"%s must be exactly %d bytes, got %d", what, kvRootLen, len(v.Bytes))
	}
	var out [kvRootLen]byte
	copy(out[:], v.Bytes)
	return hash.FromBytes(out), nil
}

func kvProvRootField(m map[string]wcb.Value, key, what string) (hash.Hash32, error) {
	v, ok := m[key]
	if !ok {
		return hash.Hash32{}, werrors.Malformedf("%s %q is missing", what, key)
	}
	return kvProvRoot(v, what+" "+key)
}

// ParseKvProvDescriptor decodes and validates canonical descriptor bytes.
func ParseKvProvDescriptor(encoded []byte, lim limits.Limits) (KvProvDescriptor, error) {
	value, err := wcb.Decode(encoded, lim)
	if err != nil {
		return KvProvDescriptor{}, err
	}
	if value.Kind != wcb.Map {
		return KvProvDescriptor{}, werrors.NewMalformed(
			"KV_CACHE_PROV descriptor must be a WCB map")
	}
	m := value.Map
	if err := kvProvExactKeys(m, []string{
		"schema", "page", "model_root", "adapter_roots", "adapter_set_root",
		"tokenizer", "tokenizer_root", "attention", "positional", "layout",
		"salt", "extensions", "payload_hash",
	}); err != nil {
		return KvProvDescriptor{}, err
	}
	schema, err := kvProvText(m, "schema", "KV_CACHE_PROV")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	if schema != KvProvSchema {
		return KvProvDescriptor{}, werrors.Malformedf(
			"unsupported KV_CACHE_PROV schema %q", schema)
	}
	pageValue, ok := m["page"]
	if !ok {
		return KvProvDescriptor{}, werrors.NewMalformed(
			"KV_CACHE_PROV descriptor is missing page")
	}
	page, err := ParseKvCacheDescriptorValue(pageValue, lim)
	if err != nil {
		return KvProvDescriptor{}, err
	}
	modelRoot, err := kvProvRootField(m, "model_root", "KV_CACHE_PROV")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	if isZeroHash(modelRoot) {
		return KvProvDescriptor{}, werrors.NewMalformed(
			"KV provenance model_root must not be all zero")
	}
	adapterValue, ok := m["adapter_roots"]
	if !ok || adapterValue.Kind != wcb.Array {
		return KvProvDescriptor{}, werrors.NewMalformed(
			"KV_CACHE_PROV adapter_roots must be an array")
	}
	adapterRoots := make([]hash.Hash32, 0, len(adapterValue.Array))
	for _, item := range adapterValue.Array {
		root, err := kvProvRoot(item, "KV provenance adapter root")
		if err != nil {
			return KvProvDescriptor{}, err
		}
		adapterRoots = append(adapterRoots, root)
	}
	adapterSet, err := kvProvRootField(m, "adapter_set_root", "KV_CACHE_PROV")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	tokenizer, err := kvProvMapValue(m, "tokenizer", "KV_CACHE_PROV")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	if err := kvProvExactKeys(tokenizer, []string{"scheme", "vocab_size"}); err != nil {
		return KvProvDescriptor{}, err
	}
	scheme, err := kvProvText(tokenizer, "scheme", "KV_CACHE_PROV tokenizer")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	vocabSize, err := kvProvU64(tokenizer, "vocab_size", "KV_CACHE_PROV tokenizer")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	tokenizerRootRecorded, err := kvProvRootField(m, "tokenizer_root", "KV_CACHE_PROV")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	attentionMap, err := kvProvMapValue(m, "attention", "KV_CACHE_PROV")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	if err := kvProvExactKeys(attentionMap, []string{
		"head_count", "head_dim", "sliding_window", "rope_theta_bits", "rope_scale_bits",
	}); err != nil {
		return KvProvDescriptor{}, err
	}
	headCount, err := kvProvU64(attentionMap, "head_count", "KV_CACHE_PROV attention")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	if headCount > math.MaxUint32 {
		return KvProvDescriptor{}, werrors.NewMalformed(
			"KV provenance attention head_count exceeds u32")
	}
	headDim, err := kvProvU64(attentionMap, "head_dim", "KV_CACHE_PROV attention")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	if headDim > math.MaxUint32 {
		return KvProvDescriptor{}, werrors.NewMalformed(
			"KV provenance attention head_dim exceeds u32")
	}
	slidingWindow, err := kvProvU64(attentionMap, "sliding_window", "KV_CACHE_PROV attention")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	thetaBits, err := kvProvU64(attentionMap, "rope_theta_bits", "KV_CACHE_PROV attention")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	scaleBits, err := kvProvU64(attentionMap, "rope_scale_bits", "KV_CACHE_PROV attention")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	positionalName, err := kvProvText(m, "positional", "KV_CACHE_PROV")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	positional, err := ParseKvPositional(positionalName)
	if err != nil {
		return KvProvDescriptor{}, err
	}
	layoutMap, err := kvProvMapValue(m, "layout", "KV_CACHE_PROV")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	if err := kvProvExactKeys(layoutMap, []string{"dtype", "page_rows", "layer_count"}); err != nil {
		return KvProvDescriptor{}, err
	}
	dtypeName, err := kvProvText(layoutMap, "dtype", "KV_CACHE_PROV layout")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	dtype, err := ParseKvDtype(dtypeName)
	if err != nil {
		return KvProvDescriptor{}, err
	}
	pageRows, err := kvProvU64(layoutMap, "page_rows", "KV_CACHE_PROV layout")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	layerCount, err := kvProvU64(layoutMap, "layer_count", "KV_CACHE_PROV layout")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	salt, err := kvProvRootField(m, "salt", "KV_CACHE_PROV")
	if err != nil {
		return KvProvDescriptor{}, err
	}
	extensionValue, ok := m["extensions"]
	if !ok || extensionValue.Kind != wcb.Array {
		return KvProvDescriptor{}, werrors.NewMalformed(
			"KV_CACHE_PROV extensions must be an array")
	}
	extensions := make([]KvExtension, 0, len(extensionValue.Array))
	for _, item := range extensionValue.Array {
		if item.Kind != wcb.Map {
			return KvProvDescriptor{}, werrors.NewMalformed(
				"KV_CACHE_PROV extension must be a WCB map")
		}
		if err := kvProvExactKeys(item.Map, []string{"name", "value"}); err != nil {
			return KvProvDescriptor{}, err
		}
		name, err := kvProvText(item.Map, "name", "KV_CACHE_PROV extension")
		if err != nil {
			return KvProvDescriptor{}, err
		}
		value, ok := item.Map["value"]
		if !ok {
			return KvProvDescriptor{}, werrors.NewMalformed(
				"KV_CACHE_PROV extension value is missing")
		}
		extensions = append(extensions, KvExtension{Name: name, Value: value})
	}
	payloadHash, err := kvProvRootField(m, "payload_hash", "KV_CACHE_PROV")
	if err != nil {
		return KvProvDescriptor{}, err
	}

	derivedAdapterSet, err := AdapterSetRoot(adapterRoots)
	if err != nil {
		return KvProvDescriptor{}, err
	}
	if derivedAdapterSet != adapterSet {
		return KvProvDescriptor{}, werrors.NewMalformed(
			"KV provenance adapter_set_root does not match the adapter roots")
	}
	derivedTokenizer, err := TokenizerRoot(scheme, vocabSize)
	if err != nil {
		return KvProvDescriptor{}, err
	}
	if derivedTokenizer != tokenizerRootRecorded {
		return KvProvDescriptor{}, werrors.NewMalformed(
			"KV provenance tokenizer_root does not match the tokenizer descriptor")
	}

	descriptor := KvProvDescriptor{
		Page: page,
		Provenance: KvProvenance{
			ModelRoot:          modelRoot,
			AdapterRoots:       adapterRoots,
			AdapterSetRoot:     adapterSet,
			TokenizerScheme:    scheme,
			TokenizerVocabSize: vocabSize,
			TokenizerRoot:      tokenizerRootRecorded,
			Attention: KvAttention{
				HeadCount:     uint32(headCount),
				HeadDim:       uint32(headDim),
				SlidingWindow: slidingWindow,
				RopeThetaBits: thetaBits,
				RopeScaleBits: scaleBits,
			},
			Positional: positional,
			Layout: KvLayout{
				Dtype:      dtype,
				PageRows:   pageRows,
				LayerCount: layerCount,
			},
			Salt:       salt,
			Extensions: extensions,
		},
		PayloadHash: payloadHash,
	}
	if err := descriptor.Validate(); err != nil {
		return KvProvDescriptor{}, err
	}
	reencoded, err := descriptor.CanonicalBytes(lim)
	if err != nil {
		return KvProvDescriptor{}, err
	}
	if !bytes.Equal(reencoded, encoded) {
		return KvProvDescriptor{}, werrors.NewMalformed(
			"KV_CACHE_PROV descriptor is not canonical")
	}
	return descriptor, nil
}

// DecodedKvProvPage is one decoded mode-19 page.
type DecodedKvProvPage struct {
	Descriptor KvProvDescriptor
	Root       hash.Hash32
	Bytes      []byte
}

// EncodeKvProvPage encodes one provenance-bound KV page. The payload is built
// by the frozen mode-12 encoder and carried verbatim.
func EncodeKvProvPage(
	input KvCacheInput,
	elements []byte,
	provenance KvProvenanceInput,
	lim limits.Limits,
) ([]byte, error) {
	pageFrame, err := EncodeKvPage(input, elements, lim)
	if err != nil {
		return nil, err
	}
	parsed, err := frame.Parse(pageFrame, lim)
	if err != nil {
		return nil, err
	}
	page, err := KvCacheMetadata(parsed, lim)
	if err != nil {
		return nil, err
	}
	_, payload, err := splitAIBodyPrefix(parsed.Body)
	if err != nil {
		return nil, err
	}
	canonical, err := provenance.Canonicalize()
	if err != nil {
		return nil, err
	}
	descriptor, err := NewKvProvDescriptor(page, canonical, elements)
	if err != nil {
		return nil, err
	}
	body, err := buildAIBody(descriptor.value(), payload, lim)
	if err != nil {
		return nil, err
	}
	return frame.Encode(frame.Parts{
		FormatMajor:  frame.FormatMajor,
		FormatMinor:  frame.FormatMinor,
		Domain:       frame.DomainPayload,
		Mode:         frame.ModeKvCacheProv,
		Flags:        0,
		Level:        0,
		DecodedLen:   descriptor.Page.PageBytes,
		ContentHash:  descriptor.PayloadHash,
		Dependencies: nil,
		Chunks:       nil,
		Body:         body,
	}, lim)
}

// DecodeKvProvPage decodes and fully verifies one provenance-bound KV page.
func DecodeKvProvPage(f *frame.Frame, lim limits.Limits) (DecodedKvProvPage, error) {
	if f.Header.Mode != frame.ModeKvCacheProv {
		return DecodedKvProvPage{}, werrors.Malformedf(
			"expected a KV_CACHE_PROV frame, got mode %s", f.Header.Mode)
	}
	descriptor, payload, err := splitAIBodyPrefix(f.Body)
	if err != nil {
		return DecodedKvProvPage{}, err
	}
	meta, err := ParseKvProvDescriptor(descriptor, lim)
	if err != nil {
		return DecodedKvProvPage{}, err
	}
	pageBytes := meta.Page.PageBytes
	if meta.Page.PayloadLen != uint64(len(payload)) {
		return DecodedKvProvPage{}, werrors.Malformedf(
			"KV_CACHE_PROV payload is %d bytes but page.payload_len declares %d",
			len(payload), meta.Page.PayloadLen)
	}
	if f.Header.DecodedLen != pageBytes {
		return DecodedKvProvPage{}, werrors.Corruptf(
			"KV_CACHE_PROV decoded_len is %d but the page descriptor requires %d",
			f.Header.DecodedLen, pageBytes)
	}
	if f.Header.ContentHash != meta.PayloadHash {
		return DecodedKvProvPage{}, werrors.NewCorrupt(
			"KV_CACHE_PROV payload_hash does not match the frame content_hash")
	}
	bytes, err := DecodeKvPlanes(meta.Page.Dtype, payload, pageBytes)
	if err != nil {
		return DecodedKvProvPage{}, err
	}
	if uint64(len(bytes)) != pageBytes {
		return DecodedKvProvPage{}, werrors.NewCorrupt(
			"KV_CACHE_PROV reconstructed page length is inconsistent")
	}
	if hash.Domain(hash.DomainChunk, bytes) != meta.PayloadHash {
		return DecodedKvProvPage{}, werrors.NewCorrupt(
			"KV_CACHE_PROV payload hash mismatch against the reconstructed page")
	}
	root, err := meta.Root(lim)
	if err != nil {
		return DecodedKvProvPage{}, err
	}
	return DecodedKvProvPage{Descriptor: meta, Root: root, Bytes: bytes}, nil
}

// KvProvMetadata parses and validates a mode-19 descriptor without decoding
// the payload.
func KvProvMetadata(f *frame.Frame, lim limits.Limits) (KvProvDescriptor, error) {
	if f.Header.Mode != frame.ModeKvCacheProv {
		return KvProvDescriptor{}, werrors.Malformedf(
			"expected a KV_CACHE_PROV frame, got mode %s", f.Header.Mode)
	}
	descriptor, _, err := splitAIBodyPrefix(f.Body)
	if err != nil {
		return KvProvDescriptor{}, err
	}
	return ParseKvProvDescriptor(descriptor, lim)
}
