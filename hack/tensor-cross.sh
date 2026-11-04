#!/usr/bin/env bash
# Cross-language WEFT-TENSOR gate.
#
# Usage (inside the Docker courts, with /work as the program home):
#   hack/tensor-cross.sh payload     # write the deterministic element bytes
#   hack/tensor-cross.sh rust-gen    # Rust CLI: root, split, reassemble
#   hack/tensor-cross.sh go-build    # build the Go CLI
#   hack/tensor-cross.sh go-gen      # Go CLI: root, split, reassemble
#   hack/tensor-cross.sh cross       # each language reads the other's shards
#   hack/tensor-cross.sh compare     # compare roots, containers, and bytes
#
# The exchanged artifacts are the logical root, the shard containers, and the
# reassembled canonical row-major bytes. No source is shared.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CROSS="/work/.docker-cache/tensor-cross"
RUST="/work/.docker-cache/target-rust/release/weftcodec"
GO="/work/.docker-cache/weftcodec-go"

DTYPE="bf16"
SHAPE="4,6,8"
AXIS="0"
SHARDS="4"
# 4 * 6 * 8 elements * 2 bytes.
PAYLOAD_BYTES=$((4 * 6 * 8 * 2))

payload() {
  mkdir -p "$CROSS"
  local path="$CROSS/payload.bin"
  : > "$path"
  local i value
  for ((i = 0; i < PAYLOAD_BYTES; i++)); do
    value=$(( (i * 37 + 11) % 256 ))
    printf "\\$(printf '%03o' "$value")" >> "$path"
  done
  echo "payload: $path ($(wc -c < "$path") bytes)"
}

cli_for() {
  case "$1" in
    rust) echo "$RUST" ;;
    go) echo "$GO" ;;
    *) echo "unknown cli $1" >&2; return 1 ;;
  esac
}

extract_root() {
  sed -n 's/.*"logical_root": *"\([0-9a-f]\{64\}\)".*/\1/p' "$1" | head -n 1
}

gen() {
  local lang="$1" cli out
  cli="$(cli_for "$lang")"
  out="$CROSS/$lang"
  mkdir -p "$out"
  [ -f "$CROSS/payload.bin" ] || payload

  "$cli" tensor root \
    --input "$CROSS/payload.bin" --dtype "$DTYPE" --shape "$SHAPE" --json \
    > "$out/root.json"
  local root
  root="$(extract_root "$out/root.json")"
  if [[ -z "$root" ]]; then
    echo "$lang: root JSON has no logical_root" >&2
    return 1
  fi

  "$cli" tensor split \
    --input "$CROSS/payload.bin" --dtype "$DTYPE" --shape "$SHAPE" \
    --axis "$AXIS" --shards "$SHARDS" --out-dir "$out/shards" --json \
    > "$out/split.json"
  "$cli" tensor root \
    --shards-dir "$out/shards" --dtype "$DTYPE" --shape "$SHAPE" \
    --axis "$AXIS" --shards "$SHARDS" --json \
    > "$out/shard-root.json"
  "$cli" tensor reassemble \
    --shards-dir "$out/shards" --dtype "$DTYPE" --shape "$SHAPE" \
    --axis "$AXIS" --shards "$SHARDS" \
    --output "$out/reassembled.bin" --expect-root "$root" --json \
    > "$out/reassemble.json"

  if ! cmp -s "$CROSS/payload.bin" "$out/reassembled.bin"; then
    echo "$lang: reassembled bytes differ from the payload" >&2
    return 1
  fi
  echo "$lang-gen: root $root, $SHARDS shards in $out"
}

go_build() {
  ( cd "$ROOT" && CGO_ENABLED=0 GOFLAGS=-buildvcs=false go build -o "$GO" ./cmd/weftcodec )
  echo "go-build: wrote $GO"
}

cross() {
  local root
  root="$(extract_root "$CROSS/rust/root.json")"
  if [[ -z "$root" ]]; then
    echo "cross: missing Rust root" >&2
    return 1
  fi
  "$GO" tensor root \
    --shards-dir "$CROSS/rust/shards" --dtype "$DTYPE" --shape "$SHAPE" \
    --axis "$AXIS" --shards "$SHARDS" --json \
    > "$CROSS/go-from-rust-root.json"
  "$GO" tensor reassemble \
    --shards-dir "$CROSS/rust/shards" --dtype "$DTYPE" --shape "$SHAPE" \
    --axis "$AXIS" --shards "$SHARDS" \
    --output "$CROSS/go-from-rust.bin" --expect-root "$root" --json \
    > "$CROSS/go-from-rust-reassemble.json"

  root="$(extract_root "$CROSS/go/root.json")"
  if [[ -z "$root" ]]; then
    echo "cross: missing Go root" >&2
    return 1
  fi
  "$RUST" tensor root \
    --shards-dir "$CROSS/go/shards" --dtype "$DTYPE" --shape "$SHAPE" \
    --axis "$AXIS" --shards "$SHARDS" --json \
    > "$CROSS/rust-from-go-root.json"
  "$RUST" tensor reassemble \
    --shards-dir "$CROSS/go/shards" --dtype "$DTYPE" --shape "$SHAPE" \
    --axis "$AXIS" --shards "$SHARDS" \
    --output "$CROSS/rust-from-go.bin" --expect-root "$root" --json \
    > "$CROSS/rust-from-go-reassemble.json"
  echo "cross: each language read the other's shards"
}

compare() {
  local failures=0
  local rust_root go_root shard_root_root shard_root_go
  rust_root="$(extract_root "$CROSS/rust/root.json")"
  go_root="$(extract_root "$CROSS/go/root.json")"
  if [[ -n "$rust_root" && "$rust_root" == "$go_root" ]]; then
    echo "PASS logical root $rust_root"
  else
    echo "FAIL logical root rust=$rust_root go=$go_root"
    failures=$((failures + 1))
  fi

  for name in root shard-root; do
    if cmp -s "$CROSS/rust/$name.json" "$CROSS/go/$name.json"; then
      echo "PASS $name JSON byte-identical"
    else
      echo "FAIL $name JSON differs"
      failures=$((failures + 1))
    fi
  done
  # split/reassemble JSON carry host paths by design; compare their roots.
  for name in split reassemble; do
    if [[ "$(extract_root "$CROSS/rust/$name.json")" == "$(extract_root "$CROSS/go/$name.json")" ]]; then
      echo "PASS $name root"
    else
      echo "FAIL $name root differs"
      failures=$((failures + 1))
    fi
  done

  for ((i = 0; i < SHARDS; i++)); do
    if cmp -s "$CROSS/rust/shards/shard-$i.bin" "$CROSS/go/shards/shard-$i.bin"; then
      echo "PASS shard-$i container byte-identical"
    else
      echo "FAIL shard-$i container differs"
      failures=$((failures + 1))
    fi
  done

  if cmp -s "$CROSS/go-from-rust.bin" "$CROSS/payload.bin" &&
     cmp -s "$CROSS/rust-from-go.bin" "$CROSS/payload.bin"; then
    echo "PASS cross reassembly reproduces the payload"
  else
    echo "FAIL cross reassembly bytes differ"
    failures=$((failures + 1))
  fi

  if [[ -n "$rust_root" ]] &&
     [[ "$rust_root" == "$(extract_root "$CROSS/go-from-rust-root.json")" ]] &&
     [[ "$rust_root" == "$(extract_root "$CROSS/rust-from-go-root.json")" ]]; then
    echo "PASS cross shard roots agree"
  else
    echo "FAIL cross shard roots differ"
    failures=$((failures + 1))
  fi

  echo "payload sha256 $(sha256sum "$CROSS/payload.bin" | cut -d' ' -f1)"

  if [[ "$failures" -ne 0 ]]; then
    echo "tensor cross-language gate: $failures failure(s)"
    return 1
  fi
  echo "tensor cross-language gate: all checks passed"
}

case "${1:-}" in
  payload) payload ;;
  rust-gen) gen rust ;;
  go-build) go_build ;;
  go-gen) gen go ;;
  cross) cross ;;
  compare) compare ;;
  *)
    echo "usage: $0 <payload|rust-gen|go-build|go-gen|cross|compare>" >&2
    exit 64
    ;;
esac
