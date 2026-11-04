#!/usr/bin/env bash
# Cross-language Context Root gate.
#
# Usage (inside the Docker courts, with /work as the program home):
#   hack/context-cross.sh rust-gen        # Rust CLI writes IDs/URIs/descriptors
#   hack/context-cross.sh go-build        # build the Go CLI
#   hack/context-cross.sh go-gen          # Go CLI writes the same artifacts
#   hack/context-cross.sh go-parse-rust   # Go parses Rust descriptors -> URIs
#   hack/context-cross.sh rust-parse-go   # Rust parses Go descriptors -> URIs
#   hack/context-cross.sh compare         # compare every exchanged artifact
#
# The exchanged "vector" is the descriptor URI text: it carries the selector
# source, snapshot, closure, capability, and budget. Each language must
# reproduce the other's ContextRootId and authoritative URI. No source is
# shared; only the text crosses the boundary.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CROSS="/work/.docker-cache/context-cross"
RUST="/work/.docker-cache/target-rust/release/weftcodec"
GO="/work/.docker-cache/weftcodec-go"

names=(true filter escaping resolver)
snapshots=(1111111111111111111111111111111111111111111111111111111111111111
           2222222222222222222222222222222222222222222222222222222222222222
           3333333333333333333333333333333333333333333333333333333333333333
           4444444444444444444444444444444444444444444444444444444444444444)
causals=(""
         "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
         "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
         "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
budgets=(""
         "max_results=5,max_bytes=4096,max_depth=2"
         "max_work=7"
         "")
caps=(""
         "4444444444444444444444444444444444444444444444444444444444444444"
         ""
         "")
resolvers=(weft-resolver-v1 weft-resolver-v1 weft-resolver-v1 weft-resolver-experimental-1)

setup_sources() {
  mkdir -p "$CROSS/src" "$CROSS/rust" "$CROSS/go" \
    "$CROSS/go-parsed" "$CROSS/rust-parsed"
  printf '%s' 'select true' > "$CROSS/src/true.weftql"
  printf '%s' 'select attrs.project == "demo" limit 10' > "$CROSS/src/filter.weftql"
  printf '%s' 'select attrs.e == "\u00e9\uD83D\uDE00\n\t\"\\/"' > "$CROSS/src/escaping.weftql"
  printf '%s' 'select relation(derived_from, obj:00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff)' > "$CROSS/src/resolver.weftql"
}

cli_for() {
  case "$1" in
    rust) echo "$RUST" ;;
    go) echo "$GO" ;;
    *) echo "unknown cli $1" >&2; return 1 ;;
  esac
}

gen() {
  local lang="$1" cli out
  cli="$(cli_for "$lang")"
  out="$CROSS/$lang"
  setup_sources
  for i in "${!names[@]}"; do
    local name="${names[$i]}"
    local source="$CROSS/src/$name.weftql"
    local args=(
      context id
      --selector "$source"
      --namespace default
      --snapshot "${snapshots[$i]}"
      --resolver "${resolvers[$i]}"
    )
    if [[ -n "${causals[$i]}" ]]; then args+=(--causal "${causals[$i]}"); fi
    if [[ -n "${budgets[$i]}" ]]; then args+=(--budget "${budgets[$i]}"); fi
    if [[ -n "${caps[$i]}" ]]; then args+=(--capability-hash "${caps[$i]}"); fi
    local id_args=("${args[@]}" --json)
    "$cli" "${id_args[@]}" > "$out/$name.id.json"

    local uri_args=("${args[@]}")
    uri_args[1]="uri"
    "$cli" "${uri_args[@]}" > "$out/$name.uri"
    "$cli" "${uri_args[@]}" --descriptor > "$out/$name.descriptor"
  done

  "$cli" causal id \
    --parents "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" \
    --json > "$out/causal.json"
  echo "$lang-gen: wrote ${#names[@]} context cases and one causal case to $out"
}

go_build() {
  ( cd "$ROOT" && CGO_ENABLED=0 GOFLAGS=-buildvcs=false go build -o "$GO" ./cmd/weftcodec )
  echo "go-build: wrote $GO"
}

parse_descriptors() {
  local from="$1" to="$2" parser="$3"
  for i in "${!names[@]}"; do
    local name="${names[$i]}"
    "$parser" context parse "$(cat "$CROSS/$from/$name.descriptor")" > "$CROSS/$to/$name.uri"
  done
  echo "$parser parsed ${#names[@]} $from descriptors -> $to"
}

extract_id() {
  sed -n 's/.*"context_root_id": *"\([0-9a-f]\{64\}\)".*/\1/p' "$1"
}

compare() {
  local failures=0
  echo "== ContextRootId equality (id JSON) =="
  for name in "${names[@]}"; do
    local a b
    a="$(extract_id "$CROSS/rust/$name.id.json")"
    b="$(extract_id "$CROSS/go/$name.id.json")"
    if [[ -n "$a" && "$a" == "$b" ]]; then
      echo "PASS id $name $a"
    else
      echo "FAIL id $name rust=$a go=$b"; failures=$((failures + 1))
    fi
  done

  echo "== authoritative URI equality =="
  for name in "${names[@]}"; do
    if cmp -s "$CROSS/rust/$name.uri" "$CROSS/go/$name.uri"; then
      echo "PASS uri $name"
    else
      echo "FAIL uri $name"; failures=$((failures + 1))
    fi
  done

  echo "== descriptor equality =="
  for name in "${names[@]}"; do
    if cmp -s "$CROSS/rust/$name.descriptor" "$CROSS/go/$name.descriptor"; then
      echo "PASS descriptor $name"
    else
      echo "FAIL descriptor $name"; failures=$((failures + 1))
    fi
  done

  echo "== cross parse: Go canonicalizes Rust descriptors =="
  for name in "${names[@]}"; do
    if cmp -s "$CROSS/rust/$name.uri" "$CROSS/go-parsed/$name.uri"; then
      echo "PASS go-parse-rust $name"
    else
      echo "FAIL go-parse-rust $name"; failures=$((failures + 1))
    fi
  done

  echo "== cross parse: Rust canonicalizes Go descriptors =="
  for name in "${names[@]}"; do
    if cmp -s "$CROSS/go/$name.uri" "$CROSS/rust-parsed/$name.uri"; then
      echo "PASS rust-parse-go $name"
    else
      echo "FAIL rust-parse-go $name"; failures=$((failures + 1))
    fi
  done

  echo "== causal closure equality =="
  local rust_causal go_causal
  rust_causal="$(sed -n 's/.*"closure_hash": *"\([0-9a-f]\{64\}\)".*/\1/p' "$CROSS/rust/causal.json")"
  go_causal="$(sed -n 's/.*"closure_hash": *"\([0-9a-f]\{64\}\)".*/\1/p' "$CROSS/go/causal.json")"
  if [[ -n "$rust_causal" && "$rust_causal" == "$go_causal" ]]; then
    echo "PASS causal $rust_causal"
  else
    echo "FAIL causal rust=$rust_causal go=$go_causal"; failures=$((failures + 1))
  fi
  local rust_hex go_hex
  rust_hex="$(sed -n 's/.*"canonical_hex": *"\([0-9a-f]*\)".*/\1/p' "$CROSS/rust/causal.json")"
  go_hex="$(sed -n 's/.*"canonical_hex": *"\([0-9a-f]*\)".*/\1/p' "$CROSS/go/causal.json")"
  if [[ -n "$rust_hex" && "$rust_hex" == "$go_hex" ]]; then
    echo "PASS causal canonical bytes"
  else
    echo "FAIL causal canonical bytes"; failures=$((failures + 1))
  fi

  if [[ "$failures" -ne 0 ]]; then
    echo "context cross-language gate: $failures failure(s)"
    return 1
  fi
  echo "context cross-language gate: all checks passed"
}

case "${1:-}" in
  rust-gen) setup_sources; gen rust ;;
  go-build) go_build ;;
  go-gen) gen go ;;
  go-parse-rust) parse_descriptors rust go-parsed "$GO" ;;
  rust-parse-go) parse_descriptors go rust-parsed "$RUST" ;;
  compare) compare ;;
  *)
    echo "usage: $0 <rust-gen|go-build|go-gen|go-parse-rust|rust-parse-go|compare>" >&2
    exit 64
    ;;
esac
