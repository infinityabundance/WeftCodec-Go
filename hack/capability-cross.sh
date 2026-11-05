#!/usr/bin/env bash
# Cross-language authenticated capability envelope gate.
#
# Usage (inside the Docker courts, with /work as the program home):
#   hack/capability-cross.sh rust-gen         # Rust signs/verifies, writes vectors
#   hack/capability-cross.sh go-build         # build the Go CLI
#   hack/capability-cross.sh go-gen           # Go signs/verifies, writes vectors
#   hack/capability-cross.sh rust-verify-go   # Rust verifies the Go-signed blob
#   hack/capability-cross.sh go-verify-rust   # Go verifies the Rust-signed blob
#   hack/capability-cross.sh compare          # compare bytes, ids, rejections
#
# Both languages must agree on canonical envelope bytes, EnvelopeId, the
# deterministic Ed25519 signature, and every rejection name (bad signature,
# wrong grantee, expired, revoked, non-canonical). Only files cross the
# boundary; no source is shared.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROGRAM="$(cd "$HERE/.." && pwd)"
CROSS="/work/.docker-cache/capability-cross"
RUST="/work/.docker-cache/target-rust/release/weftcodec"
GO="/work/.docker-cache/weftcodec-go"

seed_hex="$(printf '07%.0s' $(seq 1 32))"
grantee="4242424242424242424242424242424242424242424242424242424242424242"
revocation="3333333333333333333333333333333333333333333333333333333333333333"
nonce="4444444444444444444444444444444444444444444444444444444444444444"
root_a="1111111111111111111111111111111111111111111111111111111111111111"
root_b="2222222222222222222222222222222222222222222222222222222222222222"
wrong_grantee="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

setup() {
  mkdir -p "$CROSS/rust" "$CROSS/go"
  printf '%s' "$seed_hex" > "$CROSS/seed.hex"
  printf '%s' "$revocation" > "$CROSS/revocations.txt"
}

write_envelope() {
  local lang="$1" public="$2" out="$CROSS/$lang/envelope.json"
  cat > "$out" <<EOF
{
  "schema": "weft.capability.envelope.v1",
  "grantor": "$public",
  "grantee": "$grantee",
  "tenant": "cross-tenant",
  "roots": ["$root_a", "$root_b"],
  "namespaces": ["default"],
  "predicates": ["derived_from"],
  "expires_at_ns": 2000000000000000000,
  "budget": {"max_results": 10, "max_work": 100},
  "revocation_id": "$revocation",
  "issued_at_ns": 1700000000000000000,
  "nonce": "$nonce"
}
EOF
}

# Flip the last byte of a canonical blob. The trailing bytes are the Ed25519
# signature, so the WCB document stays well formed and the failure is exactly
# BadSignature.
tamper_last_byte() {
  local in="$1" out="$2" size last new
  size=$(wc -c < "$in")
  last=$(od -An -tu1 -j $((size - 1)) -N1 "$in" | tr -d ' ')
  new=$((last ^ 255))
  head -c $((size - 1)) "$in" > "$out"
  printf "\\$(printf '%03o' "$new")" >> "$out"
}

run_rejection() {
  local cli="$1" outfile="$2" name="$3"
  shift 3
  local out rejection
  out=$("$cli" capability verify "$@" 2>&1 || true)
  rejection=""
  for known in BadSignature WrongGrantee Expired Revoked NonCanonical UnknownSchema Malformed; do
    if printf '%s' "$out" | grep -q "$known"; then
      rejection="$known"
      break
    fi
  done
  printf '%s=%s\n' "$name" "$rejection" >> "$outfile"
}

rust_build() {
  ( cd "$PROGRAM/WeftCodec" && cargo build --release --quiet )
}

go_build() {
  ( cd "$HERE" && CGO_ENABLED=0 GOFLAGS=-buildvcs=false go build -o "$GO" ./cmd/weftcodec )
}

gen() {
  local lang="$1" cli="$2"
  local dir="$CROSS/$lang"
  setup
  local public
  public=$("$cli" capability key --key "$CROSS/seed.hex")
  write_envelope "$lang" "$public"
  printf '%s' "$public" > "$dir/public.hex"
  "$cli" capability id --envelope "$dir/envelope.json" --json > "$dir/id.json"
  "$cli" capability sign --envelope "$dir/envelope.json" --key "$CROSS/seed.hex" \
    --out "$dir/envelope.blob" --json > "$dir/sign.json"
  "$cli" capability verify --blob "$dir/envelope.blob" \
    --now-ns 1700000000000000001 --grantee "$grantee" --json > "$dir/verify.json"

  local rej="$dir/rejections.txt"
  : > "$rej"
  tamper_last_byte "$dir/envelope.blob" "$dir/bad-signature.blob"
  cp "$dir/envelope.blob" "$dir/non-canonical.blob"
  printf '\000' >> "$dir/non-canonical.blob"
  printf '%s' "$revocation" > "$dir/revocations.txt"

  run_rejection "$cli" "$rej" bad_signature \
    --blob "$dir/bad-signature.blob" --now-ns 1700000000000000001
  run_rejection "$cli" "$rej" wrong_grantee \
    --blob "$dir/envelope.blob" --now-ns 1700000000000000001 --grantee "$wrong_grantee"
  run_rejection "$cli" "$rej" expired \
    --blob "$dir/envelope.blob" --now-ns 2000000000000000001
  run_rejection "$cli" "$rej" revoked \
    --blob "$dir/envelope.blob" --now-ns 1700000000000000001 --revocations "$dir/revocations.txt"
  run_rejection "$cli" "$rej" non_canonical \
    --blob "$dir/non-canonical.blob" --now-ns 1700000000000000001
  echo "$lang-gen: wrote canonical id, signature, blob, and 5 rejections to $dir"
}

rust_verify_go() {
  setup
  "$RUST" capability verify --blob "$CROSS/go/envelope.blob" \
    --now-ns 1700000000000000001 --grantee "$grantee" --json > "$CROSS/rust/verify-go.json"
  echo "rust-verify-go: OK"
}

go_verify_rust() {
  setup
  "$GO" capability verify --blob "$CROSS/rust/envelope.blob" \
    --now-ns 1700000000000000001 --grantee "$grantee" --json > "$CROSS/go/verify-rust.json"
  echo "go-verify-rust: OK"
}

extract() {
  sed -n "s/.*\"$2\": *\"\([^\"]*\)\".*/\1/p" "$1" | head -n1
}

failures=0
compare() {
  setup
  local a b
  echo "== canonical public key equality =="
  a=$(cat "$CROSS/rust/public.hex")
  b=$(cat "$CROSS/go/public.hex")
  if [[ -n "$a" && "$a" == "$b" ]]; then echo "PASS public $a"; else
    echo "FAIL public rust=$a go=$b"; failures=$((failures + 1)); fi

  echo "== envelope id and sign input equality =="
  for field in envelope_id sign_input; do
    a=$(extract "$CROSS/rust/id.json" "$field")
    b=$(extract "$CROSS/go/id.json" "$field")
    if [[ -n "$a" && "$a" == "$b" ]]; then echo "PASS $field $a"; else
      echo "FAIL $field rust=$a go=$b"; failures=$((failures + 1)); fi
  done

  echo "== byte-identical signed blob =="
  if cmp -s "$CROSS/rust/envelope.blob" "$CROSS/go/envelope.blob"; then
    echo "PASS blob $(wc -c < "$CROSS/rust/envelope.blob") bytes"
  else
    echo "FAIL blob bytes differ"; failures=$((failures + 1))
  fi

  echo "== cross verification =="
  if [[ -s "$CROSS/rust/verify-go.json" && -s "$CROSS/go/verify-rust.json" ]]; then
    echo "PASS rust-verifies-go and go-verifies-rust"
  else
    echo "FAIL cross verification output missing"; failures=$((failures + 1))
  fi

  echo "== rejection matrix equality =="
  while IFS='=' read -r name rejection; do
    local other
    other=$(sed -n "s/^$name=//p" "$CROSS/go/rejections.txt")
    if [[ -n "$rejection" && "$rejection" == "$other" ]]; then
      echo "PASS reject $name $rejection"
    else
      echo "FAIL reject $name rust=$rejection go=$other"; failures=$((failures + 1))
    fi
  done < "$CROSS/rust/rejections.txt"
  # Every expected rejection must be a real rejection, not an empty parse.
  for expected in bad_signature wrong_grantee expired revoked non_canonical; do
    case "$expected" in
      bad_signature) want=BadSignature ;;
      wrong_grantee) want=WrongGrantee ;;
      expired) want=Expired ;;
      revoked) want=Revoked ;;
      non_canonical) want=NonCanonical ;;
    esac
    got=$(sed -n "s/^$expected=//p" "$CROSS/rust/rejections.txt")
    if [[ "$got" != "$want" ]]; then
      echo "FAIL expected $expected=$want got $got"; failures=$((failures + 1))
    fi
  done

  if [[ "$failures" -ne 0 ]]; then
    echo "capability-cross: FAIL ($failures failures)" >&2
    return 1
  fi
  echo "capability-cross: OK"
}

case "${1:-}" in
  rust-gen) rust_build; gen rust "$RUST" ;;
  go-build) go_build ;;
  go-gen) go_build; gen go "$GO" ;;
  rust-verify-go) rust_verify_go ;;
  go-verify-rust) go_verify_rust ;;
  compare) compare ;;
  *)
    echo "usage: $0 <rust-gen|go-build|go-gen|rust-verify-go|go-verify-rust|compare>" >&2
    exit 64
    ;;
esac
