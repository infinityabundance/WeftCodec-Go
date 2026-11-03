# WeftCodec-Go

Independent, pure-Go implementation of the **WeftCodec v1** wire format
(`WFT1` frames, WCB canonical bytes, WeftQL selectors, WFA1 archives) defined by
`spec/WEFTCODEC-FORMAT.md` and `spec/WEFT-SELECTORS.md`.

This is **not** a binding, subprocess wrapper, FFI call, or code generator for
the Rust implementation. Every byte is produced and checked in Go. The only
relationship to Rust is the frozen conformance corpus and black-box
cross-language gates: Rust encodes and Go decodes, then Go encodes and Rust
decodes.

## Status

Honest status as of 2026-11-03 on branch `staging`:

- Implemented: 96-byte header, 48-byte dependency and chunk entries, header and
  trailer CRC32C, minimal LEB128, all six modes (RAW, ZSTD, REFERENCE, DELTA,
  DICTIONARY, STRUCTURED), chunked RAW/ZSTD, dependency DAG resolution with
  cycle/depth/fetch limits, WFA1 archives, WCB canonicalization, WeftQL parsing
  and `SelectorId`, and the CLI.
- Conformance: all 71 frozen Rust vectors replay (29 codec, 20 structured, 22
  selectors). Byte-identical re-encoding is asserted for RAW (chunked and
  unchunked), REFERENCE, DELTA, and STRUCTURED.
- Cross-language: all 8 fixture modes pass Rust→Go and Go→Rust decode
  equivalence. Go uses `github.com/klauspost/compress/zstd`; its ZSTD and
  DICTIONARY frame bytes differ from Rust's as permitted by spec §12, but both
  directions decode exactly.
- Not run: fuzzing campaigns, benchmarks, or crash-injection courts. No
  performance claims are made.

See [NOTES.md](NOTES.md) for the feature inventory, limits, deviations, and the
full command/output verification record.

## Packages

| Package | Purpose |
|---|---|
| `hash` | Domain-separated BLAKE3-256 (`u32le(len) \|\| domain \|\| data`) |
| `varint` | Minimal LEB128, zigzag |
| `wcb` | Weft Canonical Bytes encode/decode and canonical validation |
| `frame` | WFT1 header/tables/trailer encode/parse, CRCs, `io.Reader` frame reads |
| `modes` | RAW/ZSTD/REFERENCE/DELTA/DICTIONARY/STRUCTURED, chunk decode, DAG decode |
| `dependency` | `Resolver`, `MapResolver`, `DirectoryResolver`, `FrameSource` |
| `archive` | WFA1 create/parse/decode with deterministic topological ordering |
| `selector` | WeftQL lexer/parser, normalization, canonical WCB, `SelectorId` |
| `stream` | Chunked encoding and random-range reads |
| `limits` | Normative v1 bounds and checked arithmetic helpers |
| `werrors` | Normative error classes; `Class() string` on every error |
| `cmd/weftcodec` | CLI: `encode`, `decode`, `inspect --json`, `verify`, `archive create/decode`, `selector id/canonicalize` |

## Build and test

Go 1.23 or newer. `CGO_ENABLED=0` works; the implementation is pure Go.

All builds and tests must run inside the OOM-protected container court from the
program root (`/mnt/1tb_kingston/WeftFS`), never on the host.

```sh
# Unit tests + conformance replay
scripts/docker-court.sh go "cd WeftCodec-Go && GOTMPDIR=/work/.docker-cache/go-tmp GOFLAGS=-buildvcs=false go test ./..."

# Race detector
scripts/docker-court.sh go "cd WeftCodec-Go && GOTMPDIR=/work/.docker-cache/go-tmp GOFLAGS=-buildvcs=false go test -race ./..."

# Vet and cgo-free build
scripts/docker-court.sh go "cd WeftCodec-Go && GOFLAGS=-buildvcs=false go vet ./... && CGO_ENABLED=0 GOFLAGS=-buildvcs=false go build ./..."
```

Two court-specific environment adjustments are required and are **not**
correctness workarounds:

- `GOTMPDIR=/work/.docker-cache/go-tmp`: this host's Docker `/tmp` tmpfs is
  mounted `noexec`, so Go's default test-binary location cannot be executed.
- `GOFLAGS=-buildvcs=false`: the court runs as a remapped root while the nested
  Git repository is owned by the host user, so Go's VCS stamping cannot inspect
  it. Disabling VCS stamping changes nothing in the binary's codec behavior.

The exact requested form `go test ./...` was attempted first; the observed
failure and the successful rerun are recorded verbatim in `NOTES.md`.

## CLI

```sh
weftcodec encode --input FILE --output FILE --mode MODE \
    [--base FILE] [--dict FILE] [--level N] [--chunk-size N]
weftcodec decode --input FILE --output FILE [--deps DIR]
weftcodec inspect FILE --json
weftcodec verify FILE [--deps DIR]
weftcodec archive create --output FILE FRAME...
weftcodec archive decode --input FILE --out-dir DIR
weftcodec selector id --input FILE [--json]
weftcodec selector canonicalize --input FILE
```

Modes: `raw`, `zstd`, `reference`, `delta`, `dictionary`, `structured`, and
`auto`. `--deps DIR` resolves dependencies from files named by lowercase hex
digest. Errors print their normative class and exit nonzero.

## Cross-language gates

`hack/cross.sh` is a reproducible driver for the black-box matrix. It writes
only under `/work/.docker-cache/cross/` and never invokes Rust from Go code.

```sh
# 1. Rust CLI encodes fixtures
scripts/docker-court.sh rust "bash WeftCodec-Go/hack/cross.sh rust-gen"

# 2. Build the Go CLI; encode Go fixtures; decode the Rust fixtures
scripts/docker-court.sh go "cd WeftCodec-Go && bash hack/cross.sh go-build && bash hack/cross.sh go-gen && bash hack/cross.sh go-decode"

# 3. Rust CLI decodes the Go fixtures
scripts/docker-court.sh rust "bash WeftCodec-Go/hack/cross.sh rust-decode"

# 4. Compare every decoded file and the deterministic frame bytes
scripts/docker-court.sh go "cd WeftCodec-Go && bash hack/cross.sh compare"
```

Fixture modes: `raw`, empty RAW, chunked RAW, `zstd`, `reference`, `delta`,
`dictionary`, and `structured`. The observed matrix is recorded in
`NOTES.md`; all checks pass.

## Conformance corpus

`conformance/v1/` is a byte-for-byte copy of the frozen Rust v1 vectors.
Provenance, the source commit, and an aggregate checksum are recorded in
`conformance/PROVENANCE.md`. The vectors are append-only and are never edited
or regenerated by this repository.

## License

Apache-2.0; see [LICENSE](LICENSE).
