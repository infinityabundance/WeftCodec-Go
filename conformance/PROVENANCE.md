# Conformance vector provenance

The files under `conformance/v1/` are a byte-for-byte copy of the frozen v1
vectors produced by the tested Rust implementation.

- Source repository: `https://github.com/infinityabundance/WeftCodec`
- Source commit: `272054febe89592d77ec3f80015db17b96a6565e` (`staging`)
- Copy date: 2026-11-03
- Files copied: `codec/`, `structured/`, `selectors/`, and `README.md`
- Aggregate SHA-256 over `sha256sum` lines of all 71 `*.json` files, sorted with
  `LC_ALL=C sort`:
  `66701851a1502d9c00d7451d4520c2b1dc8e19ceadba7d826c2a0f53221fafe3`
- `conformance/README.md` SHA-256:
  `327b395f41d3b9ca4846871770c816a3d0773ceb5f9f16766d5254d4f53e5676`

These vectors are **append-only**. They must never be regenerated, rewritten,
reordered, or "fixed" to match a later implementation change. If the Go
implementation disagrees with a committed vector, the Go implementation is
wrong (or a new schema version must add new files). New vectors may only be
added alongside a version bump; existing files stay untouched.

The Go tests in `conformance/` replay every vector and do not modify them.
