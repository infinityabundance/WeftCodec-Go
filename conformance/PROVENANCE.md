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

---

## F1 Context Root and causal vectors (append-only addition)

- Source repository: `https://github.com/infinityabundance/WeftCodec`
- Source commit: `cdd579c85fa76d4bff79ac957b69a0d6b7cd5960` (`staging`)
- Copy date: 2026-11-03
- Files copied: `v1/context/` (16 files) and `v1/causal/` (10 files) from the
  tested Rust implementation; `conformance/README.md` was updated with their
  layouts in the same change.
- Aggregate SHA-256 over `sha256sum` lines of the 26 new `*.json` files, sorted
  with `LC_ALL=C sort`:
  `f52b42580b4608b813be0bf8847826726c1b214282fb176918692bf85186652e`
- `conformance/README.md` SHA-256:
  `25bbed7860003939a5e78052109e8d2d60efb14ce6127b6194f92bdc6386f634`

The 16 context files are 6 canonical records plus 10 rejection vectors; the 10
causal files are 4 canonical records plus 6 rejection vectors. The Go tests
replay them without modification. Existing `codec/`, `structured/`, and
`selectors/` vectors were **not** regenerated or changed; their provenance above
still applies.

---

## F6 AI-mode vectors (append-only addition)

- Source repository: `https://github.com/infinityabundance/WeftCodec`
- Source commit: `c632aa103509e601d916ef57230e6dc7d74fdb5f` (`staging`)
- Copy date: 2026-11-03
- Files copied: 20 new `codec/` vectors:
  - valid: `100_tensor_f32_2x3`, `101_tensor_f16_scalar`,
    `102_tensor_u8_zero_element`, `103_tokens_delta_boundaries`,
    `104_tokens_empty`;
  - rejects: `120`–`127` (tensor) and `130`–`135` (tokens);
  - `017_unknown_mode_reserved8` (mode 8 remains reserved).
- Aggregate SHA-256 over `sha256sum` lines of those 20 files, sorted with
  `LC_ALL=C sort`:
  `9025a7fd1d889d1ab8071ab2701e23a453f5e251cf5037f3f84f806d6711ffae`
- `conformance/README.md` SHA-256:
  `ef9cb9c0d3c9a08412005861cffd624d02db981a4c56e9cf1eb95a9555a6d07a`

The historical `017_unknown_mode.json` is retained byte-for-byte with an added
`superseded_by` marker; the Go replay suite skips superseded vectors. All
previously committed vectors keep their original bytes. The Go suite asserts
byte-identical re-encoding for the five valid AI vectors.

---

## QUANTIZED_TENSOR mode-8 vectors (append-only addition, 2026-11-04)

- Source repository: `https://github.com/infinityabundance/WeftCodec`
- Source: Rust conformance generator at the WeftCodec revision recorded by
  `bench/receipts/verify-*.json` for the F8 verification.
- Files added: `105_quantized_i8_2x3`, `106_quantized_u8_zero_element`,
  `107_quantized_i8_scalar` (valid); `030`–`037` (rejects for dtype, NaN scale,
  zero-point range/type, payload length, payload hash, scale length, layout);
  `017_unknown_mode_reserved9` (mode 9 remains reserved).
- `017_unknown_mode_reserved8` keeps its original bytes and adds
  `"superseded_by": "017_unknown_mode_reserved9"`.
- The Go suite asserts byte-identical re-encoding for the three valid
  QUANTIZED_TENSOR vectors and replays every reject vector.
