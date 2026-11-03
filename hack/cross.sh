#!/usr/bin/env bash
# Cross-language gates for WeftCodec-Go.
#
# Usage (inside the Docker courts, from /work/WeftCodec-Go):
#   hack/cross.sh rust-gen     # Rust CLI encodes fixtures into .docker-cache/cross/rust
#   hack/cross.sh go-build     # build the Go CLI into .docker-cache/weftcodec-go
#   hack/cross.sh go-gen       # Go CLI encodes fixtures into .docker-cache/cross/go
#   hack/cross.sh go-decode    # Go CLI decodes the Rust fixtures
#   hack/cross.sh rust-decode  # Rust CLI decodes the Go fixtures
#   hack/cross.sh compare      # cmp decode results and deterministic frames
#
# The script never touches the Rust source tree and only writes under
# /work/.docker-cache/cross.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="/work/.docker-cache"
CROSS="$CACHE/cross"
RUST="$CACHE/target-rust/release/weftcodec"
GO="$CACHE/weftcodec-go"

make_src() {
  mkdir -p "$CROSS/src"
  if [[ -f "$CROSS/src/payload.bin" ]]; then
    return
  fi
  : > "$CROSS/src/empty.bin"
  dd if=/dev/urandom of="$CROSS/src/payload.bin" bs=1024 count=8 status=none
  dd if=/dev/urandom of="$CROSS/src/base.bin" bs=1024 count=4 status=none
  cp "$CROSS/src/base.bin" "$CROSS/src/target.bin"
  printf 'INSERTED-REVISION-2' | dd of="$CROSS/src/target.bin" bs=1 seek=100 conv=notrunc status=none
  printf 'REPLACED-BLOCK' | dd of="$CROSS/src/target.bin" bs=1 seek=700 conv=notrunc status=none
  awk 'BEGIN { s = "weft codec dictionary line alpha beta gamma\n"; out = ""; while (length(out) < 1024) out = out s; printf "%s", substr(out, 1, 1024) }' > "$CROSS/src/dict.bin"
  awk 'BEGIN { s = "weft codec dictionary line alpha beta gamma\n"; out = ""; while (length(out) < 9000) out = out s; printf "%s", substr(out, 1, 9000) }' > "$CROSS/src/dict_data.bin"
  printf '\x08\x03\x05\x01a\x03\x01\x05\x01b\x05\x03two\x05\x01c\x00' > "$CROSS/src/structured.bin"
}

dep_hash() {
  # $1 = frame path, $2 = CLI. Prints the first dependency hash.
  "$2" inspect "$1" --json | grep -o '"hash": "[0-9a-f]*"' | head -n1 | cut -d'"' -f4
}

frame_modes() {
  printf '%s\n' raw raw_empty raw_chunked zstd reference delta dictionary structured
}

original_for() {
  case "$1" in
    raw|raw_chunked|zstd) echo payload.bin ;;
    raw_empty) echo empty.bin ;;
    reference) echo base.bin ;;
    delta) echo target.bin ;;
    dictionary) echo dict_data.bin ;;
    structured) echo structured.bin ;;
    *) echo "unknown mode $1" >&2; return 1 ;;
  esac
}

rust_gen() {
  make_src
  mkdir -p "$CROSS/rust/deps"
  "$RUST" encode --input "$CROSS/src/payload.bin" --output "$CROSS/rust/raw.wft1" --mode raw
  "$RUST" encode --input "$CROSS/src/empty.bin" --output "$CROSS/rust/raw_empty.wft1" --mode raw
  "$RUST" encode --input "$CROSS/src/payload.bin" --output "$CROSS/rust/raw_chunked.wft1" --mode raw --chunk-size 1024
  "$RUST" encode --input "$CROSS/src/payload.bin" --output "$CROSS/rust/zstd.wft1" --mode zstd --level 3
  "$RUST" encode --input "$CROSS/src/base.bin" --output "$CROSS/rust/reference.wft1" --mode reference --base "$CROSS/src/base.bin"
  "$RUST" encode --input "$CROSS/src/target.bin" --output "$CROSS/rust/delta.wft1" --mode delta --base "$CROSS/src/base.bin"
  "$RUST" encode --input "$CROSS/src/dict_data.bin" --output "$CROSS/rust/dictionary.wft1" --mode dictionary --dict "$CROSS/src/dict.bin" --level 3
  "$RUST" encode --input "$CROSS/src/structured.bin" --output "$CROSS/rust/structured.wft1" --mode structured
  cp "$CROSS/src/base.bin" "$CROSS/rust/deps/$(dep_hash "$CROSS/rust/reference.wft1" "$RUST")"
  cp "$CROSS/src/base.bin" "$CROSS/rust/deps/$(dep_hash "$CROSS/rust/delta.wft1" "$RUST")"
  cp "$CROSS/src/dict.bin" "$CROSS/rust/deps/$(dep_hash "$CROSS/rust/dictionary.wft1" "$RUST")"
  echo "rust-gen: wrote $(find "$CROSS/rust" -name '*.wft1' | wc -l) frames to $CROSS/rust"
}

go_build() {
  ( cd "$ROOT" && CGO_ENABLED=0 GOFLAGS=-buildvcs=false go build -o "$GO" ./cmd/weftcodec )
  echo "go-build: wrote $GO"
}

go_gen() {
  make_src
  mkdir -p "$CROSS/go/deps"
  "$GO" encode --input "$CROSS/src/payload.bin" --output "$CROSS/go/raw.wft1" --mode raw
  "$GO" encode --input "$CROSS/src/empty.bin" --output "$CROSS/go/raw_empty.wft1" --mode raw
  "$GO" encode --input "$CROSS/src/payload.bin" --output "$CROSS/go/raw_chunked.wft1" --mode raw --chunk-size 1024
  "$GO" encode --input "$CROSS/src/payload.bin" --output "$CROSS/go/zstd.wft1" --mode zstd --level 3
  "$GO" encode --input "$CROSS/src/base.bin" --output "$CROSS/go/reference.wft1" --mode reference --base "$CROSS/src/base.bin"
  "$GO" encode --input "$CROSS/src/target.bin" --output "$CROSS/go/delta.wft1" --mode delta --base "$CROSS/src/base.bin"
  "$GO" encode --input "$CROSS/src/dict_data.bin" --output "$CROSS/go/dictionary.wft1" --mode dictionary --dict "$CROSS/src/dict.bin" --level 3
  "$GO" encode --input "$CROSS/src/structured.bin" --output "$CROSS/go/structured.wft1" --mode structured
  cp "$CROSS/src/base.bin" "$CROSS/go/deps/$(dep_hash "$CROSS/go/reference.wft1" "$GO")"
  cp "$CROSS/src/base.bin" "$CROSS/go/deps/$(dep_hash "$CROSS/go/delta.wft1" "$GO")"
  cp "$CROSS/src/dict.bin" "$CROSS/go/deps/$(dep_hash "$CROSS/go/dictionary.wft1" "$GO")"
  echo "go-gen: wrote $(find "$CROSS/go" -name '*.wft1' | wc -l) frames to $CROSS/go"
}

go_decode() {
  mkdir -p "$CROSS/go-decoded"
  for mode in $(frame_modes); do
    args=(decode --input "$CROSS/rust/$mode.wft1" --output "$CROSS/go-decoded/$mode.bin")
    case "$mode" in
      reference|delta|dictionary) args+=(--deps "$CROSS/rust/deps") ;;
    esac
    "$GO" "${args[@]}"
  done
  echo "go-decode: decoded $(find "$CROSS/go-decoded" -name '*.bin' | wc -l) Rust frames"
}

rust_decode() {
  mkdir -p "$CROSS/rust-decoded"
  for mode in $(frame_modes); do
    args=(decode --input "$CROSS/go/$mode.wft1" --output "$CROSS/rust-decoded/$mode.bin")
    case "$mode" in
      reference|delta|dictionary) args+=(--deps "$CROSS/go/deps") ;;
    esac
    "$RUST" "${args[@]}"
  done
  echo "rust-decode: decoded $(find "$CROSS/rust-decoded" -name '*.bin' | wc -l) Go frames"
}

compare() {
  local failures=0
  echo "== decode equivalence =="
  for mode in $(frame_modes); do
    original="$CROSS/src/$(original_for "$mode")"
    if cmp -s "$CROSS/go-decoded/$mode.bin" "$original"; then
      echo "PASS rust->go  $mode"
    else
      echo "FAIL rust->go  $mode"; failures=$((failures + 1))
    fi
    if cmp -s "$CROSS/rust-decoded/$mode.bin" "$original"; then
      echo "PASS go->rust  $mode"
    else
      echo "FAIL go->rust  $mode"; failures=$((failures + 1))
    fi
  done

  echo "== deterministic frame bytes (Go == Rust) =="
  for mode in raw raw_empty raw_chunked reference delta structured; do
    if cmp -s "$CROSS/rust/$mode.wft1" "$CROSS/go/$mode.wft1"; then
      echo "PASS identical $mode"
    else
      echo "FAIL identical $mode"; failures=$((failures + 1))
    fi
  done

  echo "== zstd/dictionary are decode-equivalence modes =="
  for mode in zstd dictionary; do
    if cmp -s "$CROSS/rust/$mode.wft1" "$CROSS/go/$mode.wft1"; then
      echo "NOTE byte-identical $mode (not required)"
    else
      echo "NOTE differing bytes $mode (allowed by spec §12)"
    fi
  done

  if [[ "$failures" -ne 0 ]]; then
    echo "cross-language gate: $failures failure(s)"
    return 1
  fi
  echo "cross-language gate: all checks passed"
}

case "${1:-}" in
  rust-gen) rust_gen ;;
  go-build) go_build ;;
  go-gen) go_gen ;;
  go-decode) go_decode ;;
  rust-decode) rust_decode ;;
  compare) compare ;;
  *)
    echo "usage: $0 <rust-gen|go-build|go-gen|go-decode|rust-decode|compare>" >&2
    exit 64
    ;;
esac
