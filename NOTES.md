# WeftCodec-Go — implementation notes and verification record

Revision: `staging`, 2026-11-03.
Normative sources: `spec/WEFTCODEC-FORMAT.md` (v1.0.0) and
`spec/WEFT-SELECTORS.md` (v1.0.0). The copies in `spec/` match the program
home; the program home wins if they ever differ.

This document separates **implemented**, **verified**, and **not done**. No
claim below is made unless the command and observed output are recorded here.

---

## 1. Implemented features

### 1.1 Frame layer (`frame`)

- `WFT1` magic, 96-byte header, 48-byte dependency and chunk entries, 4-byte
  trailer.
- Header CRC32C over bytes `[0,72)` and `[76,96)` with `[72,76)` treated as
  zero; verified before any length field is trusted.
- Trailer CRC32C over every byte before it.
- Explicit `table_len`, checked arithmetic for every length, no allocation
  based on an unchecked untrusted length.
- Flag/feature bit validation: `CHUNKED`, `SELF_CONTAINED` (rejected),
  `DICT_RAW`, `ORIGINAL_HASH` (rejected), `DEPENDENCIES`, `CHUNK_HASHES`.
- Reserved fields must be zero.
- Chunk extents must be contiguous, start at 0, end exactly at `body_len`, and
  decoded lengths must sum to `decoded_len`.
- Mode rules for all six modes, matching the Rust parser's ordering of checks.
- `frame.ReadFrame(io.Reader, Limits)` and `frame.WriteFrame(io.Writer, []byte)`
  for stream-friendly use, with a pooled 64 KiB read buffer.

### 1.2 WCB (`wcb`)

- Tags `0x00..0x09`; unknown tags are `NonCanonical`.
- Minimal varints; signed and timestamp zigzag.
- UTF-8 validation for text and map keys.
- Map keys must be text and are ordered by **full encoded key bytes**
  (tag + varint length + bytes), not by Go string order.
- Duplicate and unsorted keys rejected; trailing bytes rejected; depth, item,
  and string limits enforced.
- One-byte-for-one-byte re-encoding is asserted against all frozen canonical
  vectors.

### 1.3 Modes (`modes`)

- RAW: unchunked and chunked literals; `body_len == decoded_len`.
- ZSTD: one frame or one frame per chunk; window log capped at 27; output read
  through a `declared_len + 1` limiter and rejected on any length mismatch.
- REFERENCE: empty body, exactly one dependency, length binding enforced.
- DELTA: `COPY(offset,length)` / `INSERT(length,bytes)`; zero-length ops,
  out-of-range copies, unknown opcodes, and output length mismatches rejected.
  The encoder is a direct port of the Rust greedy matcher (bounded hash table,
  four candidates per key, minimum match 8, sampled positions above one million
  base bytes), so DELTA frames are byte-identical to Rust.
- DICTIONARY: raw-content zstd dictionary (`DICT_RAW`, `dep_domain = 2`).
- STRUCTURED: canonical WCB body validated before materialization.
- Chunked decode verifies each chunk's length and hash independently.
- Dependency resolution verifies length and chunk-domain hash at the
  materialization boundary.
- Whole-frame length and content-hash verification after materialization.
- Recursive DAG decode with cycle detection, depth limit, total fetched-byte
  limit, and cumulative decoded-byte limit.
- `context.Context` is checked before resolution and before DAG traversal;
  cancellation is reported as class `Io` (the v1 taxonomy has no cancellation
  class).

### 1.4 Dependencies (`dependency`)

- `Resolver` interface with context.
- `MapResolver` (verifies on resolve) and `DirectoryResolver` (files named by
  lowercase hex digest).
- `FrameSource` and `MapFrameSource` for DAG reconstruction, with
  content-hash/bytes conflict detection.

### 1.5 Archive (`archive`)

- `WFA1` version 1 container with `u32le` entry count, `u64le` frame lengths,
  and a trailing archive CRC32C.
- Creation: parse/validate frames, reject conflicting duplicate hashes, require
  every dependency to be present, deterministic topological order (Kahn's
  algorithm, smallest digest first, matching Rust's `BTreeSet` order).
- Parse: CRC, entry-count bounds, frame-length bounds, per-entry frame parse,
  trailing-byte detection, duplicate-hash byte identity, topological-order
  enforcement.
- Decode: in-archive resolution only, cumulative decoded-byte limit, duplicate
  byte-identical entries decoded once.

### 1.6 Selectors (`selector`)

- WeftQL lexer: identifiers, numbers, punctuation, comparison operators,
  double-quoted strings with `\" \\ \/ \b \f \n \r \t \uXXXX`, surrogate-pair
  handling, raw-control rejection, unterminated-string rejection.
- Parser: `select`, boolean precedence, `not`, parentheses, `has`, `tag`,
  `relation`, comparisons, `order by`, `limit`, `offset`.
- Normalizations: `not true`/`not false`, double negation, flattening nested
  same-operator `and`/`or`. Operand order is preserved.
- Canonical WCB map (`schema`, `namespace`, `predicate`, optional `order`,
  `limit`, `offset` with `0` omitted).
- `SelectorId = H("weft/selector/v1", canonical_bytes)`.
- `FromCanonicalBytes` rejects non-canonical AST bytes, including offset 0,
  nested same-operator nodes, unnormalized `not`, unknown keys, and wrong
  schema (`UnknownSchema`), then verifies exact re-encoding.

### 1.7 CLI (`cmd/weftcodec`)

Verbs: `encode` (including `auto`), `decode`, `inspect --json`, `verify`,
`archive create`, `archive decode`, `selector id`, `selector canonicalize`.
Errors print `Class: message` and exit 1. File input/output supports `-` for
stdin/stdout.

---

## 2. Limits

Defaults are exactly spec §10. All are enforced before allocation that depends
on them; callers may lower any bound.

| Limit | Default | Enforced in |
|---|---:|---|
| `MaxFrameBytes` | 1 GiB | `frame.Parse`, `ReadFrame` |
| `MaxDecodedBytes` | 4 GiB | frame header, encoders, delta decoder, archive cumulative |
| `MaxDependencyDepth` | 8 | DAG resolver |
| `MaxDependencyCount` | 64 | `frame.Parse` |
| `MaxTotalFetchedBytes` | 8 GiB | DAG resolver, archive size |
| `MaxChunks` | 1,048,576 | `frame.Parse`, chunked encoders |
| `MaxDictionaryBytes` | 64 MiB | dictionary encode/parse |
| `MaxStructuredDepth` | 64 | WCB encode/decode |
| `MaxStructuredItems` | 1,000,000 | WCB encode/decode |
| `MaxStringBytes` | 256 MiB | WCB text/bytes/keys |
| `MaxDeltaOps` | 16,777,216 | delta encode/decode |
| `MaxSelectorBytes` | 64 KiB | selector parse/CLI |
| `MaxSelectorDepth` | 64 | selector parse/canonical decode |
| `MaxFieldSegments` | 16 | selector field paths |
| `MaxSelectorOperands` | 4,096 | selector parse/canonical decode |
| `MaxArchiveEntries` | 1,000,000 | archive create/parse |

Additional safety measures:

- zstd decoders always use `WithDecoderMaxWindow(1<<27)` and single-threaded
  decoding; output is read through `io.LimitReader(declared_len + 1)` so a
  mismatched or hostile frame cannot force output beyond the declared length
  (plus one byte to detect the violation).
- Directory and map resolvers verify length and hash before returning bytes;
  the decoder verifies again at the boundary.
- Checked add/multiply helpers return `LimitExceeded` instead of wrapping.
- No `panic` paths on untrusted input were found by the malformed-input suite.

---

## 3. Interpretations and deviations

These are deliberate and do not alter the wire bytes for conforming input.

1. **DAG decode lives in `modes`, not `dependency`.** Go forbids the
   `modes → dependency → modes` import cycle that Rust's single crate permits.
   `dependency` owns the resolver/frame-source contracts; `modes.DecodeDAG`
   owns recursive decode. Behavior is unchanged.
2. **`context.Context` is part of decode/resolve APIs.** The task requires
   cancellation. The normative taxonomy has no cancellation class, so a
   canceled context is reported as `Io`.
3. **Chunk size `0` means "unchunked" in the Go API.** Rust models this as
   `Option<usize>`; the Go API uses `int` where `0` means no chunking and
   negative values are rejected. The CLI default is 0 (unchunked). This does
   not change any encoded frame.
4. **RAW chunking.** Spec §3's `chunk_count == 0` line is read as applying to
   the unchunked form; §6 explicitly defines chunked RAW. This matches the Rust
   implementation and the frozen chunked RAW vectors.
5. **Bare selector name arguments.** The grammar spells `tag`/`relation`
   arguments as strings, but the worked example in WEFT-SELECTORS §7 uses a bare
   identifier (`relation(derived_from, ...)`). The parser accepts a non-keyword
   identifier as equivalent text; canonical bytes are identical. This matches
   Rust.
6. **Selector source that is not UTF-8** is a `Parse` error. The Rust library
   takes `&str`; its CLI converts bytes first and reports `Parse`.
7. **Object-reference helper errors.** Rust's `Hash32::from_hex` is reached
   only after a 64-hex-digit check, so its `Malformed` branch is unreachable.
   Go returns `Parse` for that unreachable path; no conforming input observes a
   difference.
8. **Error message wording differs; classes do not.** Tests compare
   `Class()`, never text, as the spec requires.
9. **ZSTD/DICTIONARY frame bytes differ from Rust.** Go uses
   `github.com/klauspost/compress/zstd` v1.17.11; Rust uses `zstd` 0.13.
   Spec §12 requires decode equivalence, not byte equality. Both directions
   pass.
10. **`stream` package.** Range reads are a separate package; the required
    package list did not name it, but the deliverable requires range reads.
11. **Hash domain length prefix saturates at `u32::MAX`.** This matches the
    Rust `hash_domain` behavior for the (unreachable for real domains) case of
    an over-long domain.
12. **`SELF_CONTAINED` appended-byte layout** is rejected, as v1 requires. The
    WFA1 archive is the supported transport container.
13. **Selector evaluation is out of scope.** This repository implements
    parsing, canonicalization, and identity only (the codec responsibility).
    It does not resolve selectors against a filesystem snapshot.

---

## 4. Verification record

Environment: `weft/go-court:1` (`golang:1.23-bookworm`, Go 1.23.12) and
`weft/rust-court:1`, both run through `scripts/docker-court.sh` with the
operator's hard limits (3 GiB memory == swap, 1024 pids, 4 CPUs, 1 GiB `/tmp`,
`no-new-privileges`, all capabilities dropped).

### 4.1 Dependency pinning

`go.mod` requires:

```
github.com/klauspost/compress v1.17.11
lukechampine.com/blake3 v1.4.1
```

`go.sum` is committed. `go mod verify` printed `all modules verified`.
Versions are pinned to the newest releases compatible with Go 1.23
(`GOTOOLCHAIN=local`); later `klauspost/compress` releases require Go 1.24/1.25.

### 4.2 The exact requested test command, and the environment issue

Command attempted (no extra environment):

```sh
scripts/docker-court.sh go "cd WeftCodec-Go && go test ./..."
```

Observed:

```
fork/exec /tmp/go-build3791173322/b104/archive.test: permission denied
fork/exec /tmp/go-build3791173322/b129/conformance.test: permission denied
... (same for every test binary)
```

Diagnosis: this host's Docker `/tmp` tmpfs refuses execution (`/bin/echo`
copied to `/tmp` also gives `Permission denied`). This is an infrastructure
property of the court, not a code failure. The tests were therefore run with
`GOTMPDIR=/work/.docker-cache/go-tmp` (mode 0777). `GOFLAGS=-buildvcs=false` is
also set because the court's remapped root cannot read the nested Git
repository's ownership metadata; VCS stamping has no effect on codec behavior.

### 4.3 Unit tests, conformance replay, race, vet, cgo-free build

Command:

```sh
scripts/docker-court.sh go "cd WeftCodec-Go && gofmt -w . && \
  GOTMPDIR=/work/.docker-cache/go-tmp GOFLAGS=-buildvcs=false go test -race ./... && \
  GOFLAGS=-buildvcs=false go vet ./... && \
  CGO_ENABLED=0 GOFLAGS=-buildvcs=false go build ./... && echo ALL_GATES_OK"
```

Observed (final run):

```
ok   github.com/infinityabundance/WeftCodec-Go/archive       1.004s
ok   github.com/infinityabundance/WeftCodec-Go/conformance   1.181s
ok   github.com/infinityabundance/WeftCodec-Go/modes         1.042s
ok   github.com/infinityabundance/WeftCodec-Go/selector      (cached)
ok   github.com/infinityabundance/WeftCodec-Go/stream        1.004s
ok   github.com/infinityabundance/WeftCodec-Go/varint        (cached)
ok   github.com/infinityabundance/WeftCodec-Go/wcb           (cached)
?    github.com/infinityabundance/WeftCodec-Go/cmd/weftcodec  [no test files]
?    ... dependency, frame, hash, limits, werrors            [no test files]
ALL_GATES_OK
```

`go vet ./...` produced no output. The `CGO_ENABLED=0` build produced no
output and no cgo symbols are used.

Verbose count (`go test -count=1 -v ./...`, awk over `--- PASS`/`--- FAIL`):

```
PASS=44 FAIL=
```

That is 44 passing top-level tests (FAIL absent means zero). Subtest counts are
included in the conformance replay:

- 29 codec vectors: success vectors decode to exact plaintext and invalid
  vectors return the expected class.
- 20 structured vectors: canonical decode/re-encode is byte-identical; rejects
  return `NonCanonical` (and truncated input `Malformed`, depth
  `LimitExceeded`).
- 22 selector vectors: identical canonical hex, identical selector id, and
  AST round-trip.
- 8 metadata re-encodings: RAW empty/one-byte/4K/64K-repetitive/chunked,
  REFERENCE, DELTA, STRUCTURED all reproduce the Rust `frame_hex` exactly.

### 4.4 Cross-language gates

Rust CLI used: `/work/.docker-cache/target-rust/release/weftcodec` (already
built in the rust court). Commands:

```sh
scripts/docker-court.sh rust "bash WeftCodec-Go/hack/cross.sh rust-gen"
scripts/docker-court.sh go "cd WeftCodec-Go && bash hack/cross.sh go-build && \
  bash hack/cross.sh go-gen && bash hack/cross.sh go-decode"
scripts/docker-court.sh rust "bash WeftCodec-Go/hack/cross.sh rust-decode"
scripts/docker-court.sh go "cd WeftCodec-Go && bash hack/cross.sh compare"
```

Observed:

```
rust-gen: wrote 8 frames to /work/.docker-cache/cross/rust
go-build: wrote /work/.docker-cache/weftcodec-go
go-gen: wrote 8 frames to /work/.docker-cache/cross/go
go-decode: decoded 8 Rust frames
rust-decode: decoded 8 Go frames

== decode equivalence ==
PASS rust->go  raw
PASS go->rust  raw
PASS rust->go  raw_empty
PASS go->rust  raw_empty
PASS rust->go  raw_chunked
PASS go->rust  raw_chunked
PASS rust->go  zstd
PASS go->rust  zstd
PASS rust->go  reference
PASS go->rust  reference
PASS rust->go  delta
PASS go->rust  delta
PASS rust->go  dictionary
PASS go->rust  dictionary
PASS rust->go  structured
PASS go->rust  structured
== deterministic frame bytes (Go == Rust) ==
PASS identical raw
PASS identical raw_empty
PASS identical raw_chunked
PASS identical reference
PASS identical delta
PASS identical structured
== zstd/dictionary are decode-equivalence modes ==
NOTE differing bytes zstd (allowed by spec §12)
NOTE differing bytes dictionary (allowed by spec §12)
cross-language gate: all checks passed
```

Matrix summary:

| Mode | Rust → Go decode | Go → Rust decode | Frame bytes |
|---|---|---|---|
| RAW (unchunked) | pass | pass | identical |
| RAW (empty) | pass | pass | identical |
| RAW (chunked, 1024) | pass | pass | identical |
| ZSTD | pass | pass | differ (allowed) |
| REFERENCE | pass | pass | identical |
| DELTA | pass | pass | identical |
| DICTIONARY | pass | pass | differ (allowed) |
| STRUCTURED | pass | pass | identical |

Note: the first cross-gate run compared Rust frames against fixtures that had
been regenerated by the Go phase and therefore failed on random payloads. The
driver was corrected so source fixtures are created once and reused; the final
run above is the authoritative one.

### 4.5 Frozen corpus integrity

`conformance/PROVENANCE.md` records the Rust source commit
`272054febe89592d77ec3f80015db17b96a6565e`, the copy date, the 71-file count,
and an aggregate SHA-256 over the sorted per-file checksums. The vectors were
not edited. The conformance tests only read them.

---

## 5. Not done / remaining gaps

- **No fuzzing.** Recommended next step: `FuzzFrameParse`, `FuzzWCBDecode`,
  `FuzzSelectorParse`, and `FuzzDeltaDecode` with the court's memory limits.
  The malformed-input unit suite is broad but hand-written.
- **No crafted zstd window-log violation test.** The decoder sets
  `WithDecoderMaxWindow(1<<27)` and rejects larger windows in the library path,
  but a malicious frame with window log 28 was not constructed to assert the
  class end-to-end.
- **No benchmarks or performance claims.** None are made.
- **No selector evaluation or cursor encoding.** These are filesystem/runtime
  responsibilities; this repository covers parsing, canonicalization, and
  identity.
- **CLI has no Go unit tests.** It is exercised end-to-end by the cross-language
  gates (encode, decode, inspect, decode with `--deps`) and manually by the
  conformance replay through library calls. Error-exit behavior was inspected
  during development but is not covered by an automated subprocess test.
- **Archive external resolution is intentionally unsupported** for
  `SELF_CONTAINED` frames, per spec §7.
- **Delta compression is the frozen greedy matcher**, not an optimal or format
  planner. It is deterministic and byte-compatible with Rust.
- **No crash/durability court.** The codec is a pure byte transform; durability
  belongs to WeftFS.

---

## 6. F1 — Context Root, causal closure, and `weft://` (implemented)

Status: **implemented and tested** for the data layer. Resolution, capability
enforcement, and AI codec modes are not implemented (see §6.6).

### 6.1 What is implemented

- **`context` package**: `Budget`, `IdentityMode` semantics, `CausalClosure`
  (`weft.causal.v1`), `CapabilityScope` (`weft.capability.v1`), `ContextRoot`
  (`weft.context-root.v1`), authoritative and descriptor `weft://` parsing and
  formatting.
- Strict WCB: unknown keys, wrong types, unsorted/duplicate/zero closure
  entries, missing parents, zero snapshot/capability hashes, empty/zero budgets,
  unknown ordering policies, and selector-binding mismatches are rejected with
  the classes in the spec.
- Domain strings `weft/context-root/v1`, `weft/causal/v1`, `weft/capability/v1`
  added to `hash`.
- Limits `MaxCausalEntries` (4,096), `MaxCapabilityEntries` (4,096),
  `MaxURIBytes` (64 KiB).
- CLI verbs `context id`, `context uri [--descriptor]`, `context parse`,
  `causal id`.
- **Vectors**: a byte-for-byte copy of the Rust-produced
  `conformance/v1/context/` (16 files) and `conformance/v1/causal/` (10 files),
  replayed by `context/context_test.go`.

### 6.2 Cross-language gate

The exchanged vector is a descriptor URI (selector text + snapshot + closure +
capability + budget). Each implementation must reproduce the other's
`ContextRootId` and authoritative URI. No source is shared.

```sh
scripts/docker-court.sh rust "cd WeftCodec && cargo build --release --quiet && cd /work/WeftCodec-Go && bash hack/context-cross.sh rust-gen"
# rust-gen: wrote 4 context cases and one causal case to
#   /work/.docker-cache/context-cross/rust
scripts/docker-court.sh go "cd WeftCodec-Go && bash hack/context-cross.sh go-build && bash hack/context-cross.sh go-gen && bash hack/context-cross.sh go-parse-rust"
# go-build: wrote /work/.docker-cache/weftcodec-go
# go-gen: wrote 4 context cases and one causal case
# ... parsed 4 rust descriptors -> go-parsed
scripts/docker-court.sh rust "cd /work/WeftCodec-Go && bash hack/context-cross.sh rust-parse-go"
# ... parsed 4 go descriptors -> rust-parsed
scripts/docker-court.sh go "cd WeftCodec-Go && bash hack/context-cross.sh compare"
```

Observed result (exact `compare` output):

```
== ContextRootId equality (id JSON) ==
PASS id true 53f70ff51a49496af01dc87e7002fa7ef7fe1ab5c192bdb326d0868b1f01a41a
PASS id filter 04f53b8eb98d9c3603d60f61075ee7f7cdafc08baef95f5ae9d58ad8bf4a5f66
PASS id escaping b04f7965b473527ee5276eb53f96cf432f38018ecc5d984fb207776050d10f46
PASS id resolver c76f19307e87ab4c7097b1019f27a698223b20be1d6cacd3fb26607d247a411e
== authoritative URI equality ==
PASS uri true|filter|escaping|resolver
== descriptor equality ==
PASS descriptor true|filter|escaping|resolver
== cross parse: Go canonicalizes Rust descriptors ==
PASS go-parse-rust true|filter|escaping|resolver
== cross parse: Rust canonicalizes Go descriptors ==
PASS rust-parse-go true|filter|escaping|resolver
== causal closure equality ==
PASS causal 58f09b15991e2de83a8f689327a928d31e3df8c80cf44562293377316b6c92ae
PASS causal canonical bytes
context cross-language gate: all checks passed
```

Note the `filter` case ID above (`04f5...`) differs from the committed
conformance vector `002_root_filter_budget` because the gate case also carries a
closure and capability hash. Both are correct for their inputs.

### 6.3 Unit and replay tests

```sh
scripts/docker-court.sh go "cd WeftCodec-Go && go test ./..."
# Observed: all packages ok, including
#   ok github.com/infinityabundance/WeftCodec-Go/context
# The context package has 5 top-level test functions; TestContextVectors runs
# 16 vector subtests and TestCausalVectors runs 10 vector subtests.

scripts/docker-court.sh go "cd WeftCodec-Go && gofmt -l context cmd/weftcodec limits hash && go vet ./..."
# Observed: no output (gofmt clean, vet clean).
```

Existing codec cross-language gate re-run: all modes decode in both directions,
all deterministic frame bytes identical, ZSTD/DICTIONARY differing bytes allowed
by §12. No regression.

`bash scripts/check-spec-drift.sh` (run in the rust court):
`spec drift: OK (24 file comparisons across 3 repos)`.

### 6.4 Deliberate Go-side decisions

These match the Rust implementation; the spec is authoritative.

1. `context uri --descriptor` filters `--descriptor` before stdlib `flag`
   parsing so it can appear anywhere in the argument list.
2. `context parse` accepts `--json` before or after the URI (stdlib `flag`
   stops at the first positional, so this is handled explicitly).
3. Descriptor parsing sorts/deduplicates `closure`; canonical record decoding
   remains strict. This mirrors `WEFT-CONTEXT-ROOT.md` §5.4.
4. `context id` uses pretty JSON with the same field names as Rust; the
   cross-language gate compares semantic fields and canonical URIs, not raw
   JSON formatting.

### 6.5 Vector provenance

`conformance/PROVENANCE.md` records the producing Rust commit and the aggregate
SHA-256 for the new files. The files are append-only; the Go tests only read
them.

### 6.6 Not implemented (labelled precisely)

- Context resolution, snapshot existence/integrity checks, authorization
  filtering, budget enforcement, evidence filtering, materialization, FUSE
  context mounts, and revocation: **specified, not implemented** (F2/F3).
- `IdentityMode` is represented as text semantics and documented in
  `spec/WEFT-CAUSAL.md`; the version-record extension is not implemented in
  either codec (no store writes in this repository).
- AI codec modes: **specified, not implemented**; no mode values allocated.
- No cryptographic capability claim. A `CapabilityHash` is an identifier.
- No performance or model-checking claim for the new identities.
