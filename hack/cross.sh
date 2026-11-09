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
  if [[ ! -f "$CROSS/src/payload.bin" ]]; then
    : > "$CROSS/src/empty.bin"
    dd if=/dev/urandom of="$CROSS/src/payload.bin" bs=1024 count=8 status=none
    dd if=/dev/urandom of="$CROSS/src/base.bin" bs=1024 count=4 status=none
    cp "$CROSS/src/base.bin" "$CROSS/src/target.bin"
    printf 'INSERTED-REVISION-2' | dd of="$CROSS/src/target.bin" bs=1 seek=100 conv=notrunc status=none
    printf 'REPLACED-BLOCK' | dd of="$CROSS/src/target.bin" bs=1 seek=700 conv=notrunc status=none
    awk 'BEGIN { s = "weft codec dictionary line alpha beta gamma\n"; out = ""; while (length(out) < 1024) out = out s; printf "%s", substr(out, 1, 1024) }' > "$CROSS/src/dict.bin"
    awk 'BEGIN { s = "weft codec dictionary line alpha beta gamma\n"; out = ""; while (length(out) < 9000) out = out s; printf "%s", substr(out, 1, 9000) }' > "$CROSS/src/dict_data.bin"
    printf '\x08\x03\x05\x01a\x03\x01\x05\x01b\x05\x03two\x05\x01c\x00' > "$CROSS/src/structured.bin"
  fi
  # AI-mode fixtures: a 24-byte f32 2x3 tensor and the token sequence
  # [1,2,3]. The canonical token payload is 01 02 02 (absolute 1, then two
  # +1 zigzag deltas).
  head -c 24 "$CROSS/src/payload.bin" > "$CROSS/src/tensor.bin"
  head -c 6 "$CROSS/src/payload.bin" > "$CROSS/src/quantized.bin"
  # FLOAT_TENSOR fixtures: a skewed exponent plane, an all-zero (single-symbol)
  # plane, and a near-uniform plane with many distinct exponents.
  printf '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\xc0\x7f' > "$CROSS/src/float_tensor.bin"
  head -c 4096 /dev/zero > "$CROSS/src/float_tensor_single.bin"
  head -c 24576 "$CROSS/src/payload.bin" > "$CROSS/src/float_tensor_uniform.bin"
  printf '{"vocab_schema":"weft.test.vocab.v1","token_ids":[1,2,3]}' > "$CROSS/src/tokens.json"
  printf '\x01\x02\x02' > "$CROSS/src/tokens_payload.bin"
  # KV_CACHE (mode 12) and ACTIVATION_STREAM (mode 13) fixtures: two bf16
  # pages of 4x2x4 elements (64 B each) and two bf16 chunks of 2x4 elements
  # (16 B each).
  dd if="$CROSS/src/payload.bin" of="$CROSS/src/kv_page0.bin" bs=1 count=64 status=none
  dd if="$CROSS/src/payload.bin" of="$CROSS/src/kv_page1.bin" bs=1 skip=64 count=64 status=none
  dd if="$CROSS/src/payload.bin" of="$CROSS/src/activation_chunk0.bin" bs=1 skip=128 count=16 status=none
  dd if="$CROSS/src/payload.bin" of="$CROSS/src/activation_chunk1.bin" bs=1 skip=144 count=16 status=none
  printf '{"model":"llama-cross","layer":3,"session":"sess-1","page_index":0,"page_count":2,"seq_start":0,"seq_end":4,"dtype":"bf16","head_count":2,"head_dim":4}' > "$CROSS/src/kv_page0.json"
  printf '{"model":"llama-cross","layer":3,"session":"sess-1","page_index":1,"page_count":2,"seq_start":4,"seq_end":8,"dtype":"bf16","head_count":2,"head_dim":4}' > "$CROSS/src/kv_page1.json"
  # KV_CACHE_PROV (mode 19): the same page bound to provenance. Adapter roots
  # are deliberately listed unsorted so both encoders must canonicalize.
  cp "$CROSS/src/kv_page0.bin" "$CROSS/src/kv_prov_page0.bin"
  cp "$CROSS/src/kv_page0.json" "$CROSS/src/kv_prov_page0.json"
  printf '{"model_root":"1111111111111111111111111111111111111111111111111111111111111111","adapter_roots":["2222222222222222222222222222222222222222222222222222222222222222","2121212121212121212121212121212121212121212121212121212121212121"],"tokenizer":{"scheme":"qwen-bpe","vocab_size":151936},"attention":{"head_count":2,"head_dim":4,"sliding_window":0,"rope_theta_bits":4667827607267037184,"rope_scale_bits":4607182418800017408},"positional":"rope","layout":{"dtype":"bf16","page_rows":4,"layer_count":8},"salt":"a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0","extensions":[{"name":"weft.test.runtime","value":"v2"}]}' > "$CROSS/src/kv_prov_prov.json"
  printf '{"stream":"activation","step":7,"rank":0,"world":1,"chunk_index":0,"chunk_count":2,"dtype":"bf16","shape":[2,4]}' > "$CROSS/src/activation_chunk0.json"
  printf '{"stream":"activation","step":7,"rank":0,"world":1,"chunk_index":1,"chunk_count":2,"dtype":"bf16","shape":[2,4]}' > "$CROSS/src/activation_chunk1.json"
  # TENSOR_SHARDED (mode 10) word inputs: one byte per quantized value.
  sharded_words 32 q8 > "$CROSS/src/sharded_q8.bin"
  sharded_words 64 q4 > "$CROSS/src/sharded_q4.bin"
  sharded_words 16 q8 > "$CROSS/src/sharded_q8_single.bin"
  # WEIGHT_DELTA (mode 11) fixtures: bf16 with one changed word, fp32 with an
  # identical base (zero delta), and q8_0 with mixed changed bytes.
  head -c 12 "$CROSS/src/payload.bin" > "$CROSS/src/weight_base_bf16.bin"
  cp "$CROSS/src/weight_base_bf16.bin" "$CROSS/src/weight_target_bf16.bin"
  printf '\x7f' | dd of="$CROSS/src/weight_target_bf16.bin" bs=1 seek=2 conv=notrunc status=none
  head -c 16 "$CROSS/src/payload.bin" > "$CROSS/src/weight_base_fp32.bin"
  cp "$CROSS/src/weight_base_fp32.bin" "$CROSS/src/weight_target_fp32.bin"
  head -c 32 "$CROSS/src/payload.bin" > "$CROSS/src/weight_base_q8.bin"
  cp "$CROSS/src/weight_base_q8.bin" "$CROSS/src/weight_target_q8.bin"
  printf '\x01\x02\x03' | dd of="$CROSS/src/weight_target_q8.bin" bs=1 seek=5 conv=notrunc status=none
  # SEMANTIC_DELTA (mode 14) fixtures: low-bit-only changes that keep the
  # mantissa_truncate/base semantic root, plus a q8_0 exact pair.
  head -c 12 "$CROSS/src/payload.bin" > "$CROSS/src/semantic_base_bf16.bin"
  printf '\x05' | dd of="$CROSS/src/semantic_base_bf16.bin" bs=1 seek=2 conv=notrunc status=none
  cp "$CROSS/src/semantic_base_bf16.bin" "$CROSS/src/semantic_target_bf16.bin"
  printf '\x07' | dd of="$CROSS/src/semantic_target_bf16.bin" bs=1 seek=2 conv=notrunc status=none
  head -c 16 "$CROSS/src/payload.bin" > "$CROSS/src/semantic_base_fp32.bin"
  printf '\x10' | dd of="$CROSS/src/semantic_base_fp32.bin" bs=1 seek=8 conv=notrunc status=none
  cp "$CROSS/src/semantic_base_fp32.bin" "$CROSS/src/semantic_target_fp32.bin"
  printf '\x1f' | dd of="$CROSS/src/semantic_target_fp32.bin" bs=1 seek=8 conv=notrunc status=none
  head -c 32 "$CROSS/src/payload.bin" > "$CROSS/src/semantic_base_q8.bin"
  cp "$CROSS/src/semantic_base_q8.bin" "$CROSS/src/semantic_target_q8.bin"
  printf '\x05\x06\x07' | dd of="$CROSS/src/semantic_target_q8.bin" bs=1 seek=5 conv=notrunc status=none
  # TENSOR_TILED (mode 15) fixtures: a 4x4 bf16 tensor in 2x2 tiles and a
  # 2x2x2 fp32 tensor in 1x2x2 tiles.
  head -c 32 "$CROSS/src/payload.bin" > "$CROSS/src/tensor_tiled_bf16.bin"
  head -c 32 "$CROSS/src/payload.bin" > "$CROSS/src/tensor_tiled_fp32.bin"
  head -c 32 "$CROSS/src/payload.bin" > "$CROSS/src/float_tiled_bf16.bin"
  head -c 32 "$CROSS/src/payload.bin" > "$CROSS/src/float_tiled_fp32.bin"
  # SPARSE_DELTA (mode 16) fixtures: bf16 with one changed word, q8_0 with
  # three separate runs, and an fp32 pair that is byte-identical (run_count 0).
  head -c 12 "$CROSS/src/payload.bin" > "$CROSS/src/sparse_base_bf16.bin"
  cp "$CROSS/src/sparse_base_bf16.bin" "$CROSS/src/sparse_target_bf16.bin"
  printf '\x7f' | dd of="$CROSS/src/sparse_target_bf16.bin" bs=1 seek=6 conv=notrunc status=none
  head -c 32 "$CROSS/src/payload.bin" > "$CROSS/src/sparse_base_q8.bin"
  cp "$CROSS/src/sparse_base_q8.bin" "$CROSS/src/sparse_target_q8.bin"
  printf '\x01\x02\x03' | dd of="$CROSS/src/sparse_target_q8.bin" bs=1 seek=1 conv=notrunc status=none
  head -c 16 "$CROSS/src/payload.bin" > "$CROSS/src/sparse_base_fp32.bin"
  cp "$CROSS/src/sparse_base_fp32.bin" "$CROSS/src/sparse_target_fp32.bin"
}

sharded_words() {
  local count="$1"
  local kind="$2"
  local i=0
  while [[ "$i" -lt "$count" ]]; do
    local value
    if [[ "$kind" == "q4" ]]; then
      value=$(( i % 16 - 8 ))
    else
      value=$(( (i * 7 + 3) % 251 - 125 ))
    fi
    printf "\\$(printf '%03o' $(( value & 255 )))"
    i=$((i + 1))
  done
}

dep_hash() {
  # $1 = frame path, $2 = CLI. Prints the first dependency hash.
  "$2" inspect "$1" --json | grep -o '"hash": "[0-9a-f]*"' | head -n1 | cut -d'"' -f4
}

frame_modes() {
  printf '%s\n' raw raw_empty raw_chunked zstd reference delta dictionary structured tensor tokens quantized float_tensor float_tensor_single float_tensor_uniform weight_delta_bf16 weight_delta_fp32 weight_delta_q8 semantic_delta_bf16 semantic_delta_fp32 semantic_delta_q8 kv_cache_p0 kv_cache_p1 kv_prov_p0 activation_c0 activation_c1 tensor_tiled_bf16 tensor_tiled_fp32 float_tiled_bf16 float_tiled_fp32 sparse_delta_bf16 sparse_delta_q8 sparse_delta_fp32
}

original_for() {
  case "$1" in
    raw|raw_chunked|zstd) echo payload.bin ;;
    raw_empty) echo empty.bin ;;
    reference) echo base.bin ;;
    delta) echo target.bin ;;
    dictionary) echo dict_data.bin ;;
    structured) echo structured.bin ;;
    tensor) echo tensor.bin ;;
    quantized) echo quantized.bin ;;
    float_tensor) echo float_tensor.bin ;;
    float_tensor_single) echo float_tensor_single.bin ;;
    float_tensor_uniform) echo float_tensor_uniform.bin ;;
    weight_delta_bf16) echo weight_target_bf16.bin ;;
    weight_delta_fp32) echo weight_target_fp32.bin ;;
    weight_delta_q8) echo weight_target_q8.bin ;;
    semantic_delta_bf16) echo semantic_target_bf16.bin ;;
    semantic_delta_fp32) echo semantic_target_fp32.bin ;;
    semantic_delta_q8) echo semantic_target_q8.bin ;;
    kv_cache_p0) echo kv_page0.bin ;;
    kv_cache_p1) echo kv_page1.bin ;;
    kv_prov_p0) echo kv_page0.bin ;;
    activation_c0) echo activation_chunk0.bin ;;
    activation_c1) echo activation_chunk1.bin ;;
    tensor_tiled_bf16) echo tensor_tiled_bf16.bin ;;
    tensor_tiled_fp32) echo tensor_tiled_fp32.bin ;;
    float_tiled_bf16) echo float_tiled_bf16.bin ;;
    float_tiled_fp32) echo float_tiled_fp32.bin ;;
    sparse_delta_bf16) echo sparse_target_bf16.bin ;;
    sparse_delta_q8) echo sparse_target_q8.bin ;;
    sparse_delta_fp32) echo sparse_target_fp32.bin ;;
    tokens) echo tokens_payload.bin ;;
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
  "$RUST" encode --input "$CROSS/src/tensor.bin" --output "$CROSS/rust/tensor.wft1" --mode tensor --dtype f32 --shape 2,3
  "$RUST" encode --input "$CROSS/src/quantized.bin" --output "$CROSS/rust/quantized.wft1" --mode quantized --dtype i8 --shape 2,3 --scale 0.25 --zero-point=-3
  "$RUST" encode --input "$CROSS/src/float_tensor.bin" --output "$CROSS/rust/float_tensor.wft1" --mode float-tensor --dtype fp32 --shape 2,3
  "$RUST" encode --input "$CROSS/src/float_tensor_single.bin" --output "$CROSS/rust/float_tensor_single.wft1" --mode float-tensor --dtype fp32 --shape 1024
  "$RUST" encode --input "$CROSS/src/float_tensor_uniform.bin" --output "$CROSS/rust/float_tensor_uniform.wft1" --mode float-tensor --dtype fp32 --shape 2,1024
  "$RUST" encode --input "$CROSS/src/tokens.json" --output "$CROSS/rust/tokens.wft1" --mode tokens
  "$RUST" encode --mode tensor-sharded --input "$CROSS/src/sharded_q8.bin" --output "$CROSS/rust/sharded_q8" --dtype q8_0 --shape 4,8 --axis 0 --shards 2 --block-size 16 --scales 1056964608,1048576000 --zero-points 0,3
  "$RUST" encode --mode tensor-sharded --input "$CROSS/src/sharded_q4.bin" --output "$CROSS/rust/sharded_q4" --dtype q4_0 --shape 2,4,8 --axis 1 --shards 4 --block-size 16 --scales 1056964608,1048576000,1040187392,1031798784 --zero-points 0,1,-1,2
  "$RUST" encode --mode tensor-sharded --input "$CROSS/src/sharded_q8_single.bin" --output "$CROSS/rust/sharded_q8_single" --dtype q8_0 --shape 16 --axis 0 --shards 1 --block-size 16 --scales 1065353216 --zero-points 0
  for set in sharded_q8 sharded_q4 sharded_q8_single; do
    "$RUST" tensor root --shards-dir "$CROSS/rust/$set" --frames > "$CROSS/rust/$set.root"
  done
  "$RUST" encode --mode tensor --input "$CROSS/src/weight_base_bf16.bin" --output "$CROSS/rust/weight_base_bf16.wft1" --dtype bf16 --shape 2,3
  "$RUST" encode --mode weight-delta --input "$CROSS/src/weight_target_bf16.bin" --output "$CROSS/rust/weight_delta_bf16.wft1" --base "$CROSS/rust/weight_base_bf16.wft1" --dtype bf16 --shape 2,3
  "$RUST" encode --mode tensor --input "$CROSS/src/weight_base_fp32.bin" --output "$CROSS/rust/weight_base_fp32.wft1" --dtype f32 --shape 2,2
  "$RUST" encode --mode weight-delta --input "$CROSS/src/weight_target_fp32.bin" --output "$CROSS/rust/weight_delta_fp32.wft1" --base "$CROSS/rust/weight_base_fp32.wft1" --dtype fp32 --shape 2,2
  "$RUST" encode --mode tensor --input "$CROSS/src/weight_base_q8.bin" --output "$CROSS/rust/weight_base_q8.wft1" --dtype i8 --shape 4,8
  "$RUST" encode --mode weight-delta --input "$CROSS/src/weight_target_q8.bin" --output "$CROSS/rust/weight_delta_q8.wft1" --base "$CROSS/rust/weight_base_q8.wft1" --dtype q8_0 --shape 4,8
  "$RUST" encode --mode tensor --input "$CROSS/src/semantic_base_bf16.bin" --output "$CROSS/rust/semantic_base_bf16.wft1" --dtype bf16 --shape 2,3
  "$RUST" encode --mode semantic-delta --input "$CROSS/src/semantic_target_bf16.bin" --output "$CROSS/rust/semantic_delta_bf16.wft1" --base "$CROSS/rust/semantic_base_bf16.wft1" --dtype bf16 --shape 2,3 --semantic-policy mantissa_truncate:3
  "$RUST" encode --mode tensor --input "$CROSS/src/semantic_base_fp32.bin" --output "$CROSS/rust/semantic_base_fp32.wft1" --dtype f32 --shape 2,2
  "$RUST" encode --mode semantic-delta --input "$CROSS/src/semantic_target_fp32.bin" --output "$CROSS/rust/semantic_delta_fp32.wft1" --base "$CROSS/rust/semantic_base_fp32.wft1" --dtype fp32 --shape 2,2 --semantic-policy mantissa_truncate:8
  "$RUST" encode --mode tensor --input "$CROSS/src/semantic_base_q8.bin" --output "$CROSS/rust/semantic_base_q8.wft1" --dtype i8 --shape 4,8
  "$RUST" encode --mode semantic-delta --input "$CROSS/src/semantic_target_q8.bin" --output "$CROSS/rust/semantic_delta_q8.wft1" --base "$CROSS/rust/semantic_base_q8.wft1" --dtype q8_0 --shape 4,8 --semantic-policy none
  "$RUST" encode --mode kv-cache --input "$CROSS/src/kv_page0.bin" --output "$CROSS/rust/kv_cache_p0.wft1" --metadata "$CROSS/src/kv_page0.json"
  "$RUST" encode --mode kv-cache --input "$CROSS/src/kv_page1.bin" --output "$CROSS/rust/kv_cache_p1.wft1" --metadata "$CROSS/src/kv_page1.json"
  "$RUST" encode --mode kv-prov --input "$CROSS/src/kv_prov_page0.bin" --output "$CROSS/rust/kv_prov_p0.wft1" --metadata "$CROSS/src/kv_prov_page0.json" --provenance "$CROSS/src/kv_prov_prov.json"
  "$RUST" encode --mode activation-stream --input "$CROSS/src/activation_chunk0.bin" --output "$CROSS/rust/activation_c0.wft1" --metadata "$CROSS/src/activation_chunk0.json"
  "$RUST" encode --mode activation-stream --input "$CROSS/src/activation_chunk1.bin" --output "$CROSS/rust/activation_c1.wft1" --metadata "$CROSS/src/activation_chunk1.json"
  "$RUST" encode --mode tensor-tiled --input "$CROSS/src/tensor_tiled_bf16.bin" --output "$CROSS/rust/tensor_tiled_bf16.wft1" --dtype bf16 --shape 4,4 --tile-shape 2,2
  "$RUST" encode --mode tensor-tiled --input "$CROSS/src/tensor_tiled_fp32.bin" --output "$CROSS/rust/tensor_tiled_fp32.wft1" --dtype f32 --shape 2,2,2 --tile-shape 1,2,2
  "$RUST" encode --mode float-tiled --input "$CROSS/src/float_tiled_bf16.bin" --output "$CROSS/rust/float_tiled_bf16.wft1" --dtype bf16 --shape 4,4 --tile-shape 2,2
  "$RUST" encode --mode float-tiled --input "$CROSS/src/float_tiled_fp32.bin" --output "$CROSS/rust/float_tiled_fp32.wft1" --dtype fp32 --shape 2,2,2 --tile-shape 1,2,2
  "$RUST" encode --mode tensor --input "$CROSS/src/sparse_base_bf16.bin" --output "$CROSS/rust/sparse_base_bf16.wft1" --dtype bf16 --shape 2,3
  "$RUST" encode --mode sparse-delta --input "$CROSS/src/sparse_target_bf16.bin" --output "$CROSS/rust/sparse_delta_bf16.wft1" --base "$CROSS/rust/sparse_base_bf16.wft1" --dtype bf16 --shape 2,3
  "$RUST" encode --mode tensor --input "$CROSS/src/sparse_base_q8.bin" --output "$CROSS/rust/sparse_base_q8.wft1" --dtype i8 --shape 4,8
  "$RUST" encode --mode sparse-delta --input "$CROSS/src/sparse_target_q8.bin" --output "$CROSS/rust/sparse_delta_q8.wft1" --base "$CROSS/rust/sparse_base_q8.wft1" --dtype q8_0 --shape 4,8
  "$RUST" encode --mode tensor --input "$CROSS/src/sparse_base_fp32.bin" --output "$CROSS/rust/sparse_base_fp32.wft1" --dtype f32 --shape 2,2
  "$RUST" encode --mode sparse-delta --input "$CROSS/src/sparse_target_fp32.bin" --output "$CROSS/rust/sparse_delta_fp32.wft1" --base "$CROSS/rust/sparse_base_fp32.wft1" --dtype fp32 --shape 2,2
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
  "$GO" encode --input "$CROSS/src/tensor.bin" --output "$CROSS/go/tensor.wft1" --mode tensor --dtype f32 --shape 2,3
  "$GO" encode --input "$CROSS/src/quantized.bin" --output "$CROSS/go/quantized.wft1" --mode quantized --dtype i8 --shape 2,3 --scale 0.25 --zero-point=-3
  "$GO" encode --input "$CROSS/src/float_tensor.bin" --output "$CROSS/go/float_tensor.wft1" --mode float-tensor --dtype fp32 --shape 2,3
  "$GO" encode --input "$CROSS/src/float_tensor_single.bin" --output "$CROSS/go/float_tensor_single.wft1" --mode float-tensor --dtype fp32 --shape 1024
  "$GO" encode --input "$CROSS/src/float_tensor_uniform.bin" --output "$CROSS/go/float_tensor_uniform.wft1" --mode float-tensor --dtype fp32 --shape 2,1024
  "$GO" encode --input "$CROSS/src/tokens.json" --output "$CROSS/go/tokens.wft1" --mode tokens
  "$GO" encode --mode tensor-sharded --input "$CROSS/src/sharded_q8.bin" --output "$CROSS/go/sharded_q8" --dtype q8_0 --shape 4,8 --axis 0 --shards 2 --block-size 16 --scales 1056964608,1048576000 --zero-points 0,3
  "$GO" encode --mode tensor-sharded --input "$CROSS/src/sharded_q4.bin" --output "$CROSS/go/sharded_q4" --dtype q4_0 --shape 2,4,8 --axis 1 --shards 4 --block-size 16 --scales 1056964608,1048576000,1040187392,1031798784 --zero-points 0,1,-1,2
  "$GO" encode --mode tensor-sharded --input "$CROSS/src/sharded_q8_single.bin" --output "$CROSS/go/sharded_q8_single" --dtype q8_0 --shape 16 --axis 0 --shards 1 --block-size 16 --scales 1065353216 --zero-points 0
  for set in sharded_q8 sharded_q4 sharded_q8_single; do
    "$GO" tensor root --shards-dir "$CROSS/go/$set" --frames > "$CROSS/go/$set.root"
  done
  "$GO" encode --mode tensor --input "$CROSS/src/weight_base_bf16.bin" --output "$CROSS/go/weight_base_bf16.wft1" --dtype bf16 --shape 2,3
  "$GO" encode --mode weight-delta --input "$CROSS/src/weight_target_bf16.bin" --output "$CROSS/go/weight_delta_bf16.wft1" --base "$CROSS/go/weight_base_bf16.wft1" --dtype bf16 --shape 2,3
  "$GO" encode --mode tensor --input "$CROSS/src/weight_base_fp32.bin" --output "$CROSS/go/weight_base_fp32.wft1" --dtype f32 --shape 2,2
  "$GO" encode --mode weight-delta --input "$CROSS/src/weight_target_fp32.bin" --output "$CROSS/go/weight_delta_fp32.wft1" --base "$CROSS/go/weight_base_fp32.wft1" --dtype fp32 --shape 2,2
  "$GO" encode --mode tensor --input "$CROSS/src/weight_base_q8.bin" --output "$CROSS/go/weight_base_q8.wft1" --dtype i8 --shape 4,8
  "$GO" encode --mode weight-delta --input "$CROSS/src/weight_target_q8.bin" --output "$CROSS/go/weight_delta_q8.wft1" --base "$CROSS/go/weight_base_q8.wft1" --dtype q8_0 --shape 4,8
  "$GO" encode --mode tensor --input "$CROSS/src/semantic_base_bf16.bin" --output "$CROSS/go/semantic_base_bf16.wft1" --dtype bf16 --shape 2,3
  "$GO" encode --mode semantic-delta --input "$CROSS/src/semantic_target_bf16.bin" --output "$CROSS/go/semantic_delta_bf16.wft1" --base "$CROSS/go/semantic_base_bf16.wft1" --dtype bf16 --shape 2,3 --semantic-policy mantissa_truncate:3
  "$GO" encode --mode tensor --input "$CROSS/src/semantic_base_fp32.bin" --output "$CROSS/go/semantic_base_fp32.wft1" --dtype f32 --shape 2,2
  "$GO" encode --mode semantic-delta --input "$CROSS/src/semantic_target_fp32.bin" --output "$CROSS/go/semantic_delta_fp32.wft1" --base "$CROSS/go/semantic_base_fp32.wft1" --dtype fp32 --shape 2,2 --semantic-policy mantissa_truncate:8
  "$GO" encode --mode tensor --input "$CROSS/src/semantic_base_q8.bin" --output "$CROSS/go/semantic_base_q8.wft1" --dtype i8 --shape 4,8
  "$GO" encode --mode semantic-delta --input "$CROSS/src/semantic_target_q8.bin" --output "$CROSS/go/semantic_delta_q8.wft1" --base "$CROSS/go/semantic_base_q8.wft1" --dtype q8_0 --shape 4,8 --semantic-policy none
  "$GO" encode --mode kv-cache --input "$CROSS/src/kv_page0.bin" --output "$CROSS/go/kv_cache_p0.wft1" --metadata "$CROSS/src/kv_page0.json"
  "$GO" encode --mode kv-cache --input "$CROSS/src/kv_page1.bin" --output "$CROSS/go/kv_cache_p1.wft1" --metadata "$CROSS/src/kv_page1.json"
  "$GO" encode --mode kv-prov --input "$CROSS/src/kv_prov_page0.bin" --output "$CROSS/go/kv_prov_p0.wft1" --metadata "$CROSS/src/kv_prov_page0.json" --provenance "$CROSS/src/kv_prov_prov.json"
  "$GO" encode --mode activation-stream --input "$CROSS/src/activation_chunk0.bin" --output "$CROSS/go/activation_c0.wft1" --metadata "$CROSS/src/activation_chunk0.json"
  "$GO" encode --mode activation-stream --input "$CROSS/src/activation_chunk1.bin" --output "$CROSS/go/activation_c1.wft1" --metadata "$CROSS/src/activation_chunk1.json"
  "$GO" encode --mode tensor-tiled --input "$CROSS/src/tensor_tiled_bf16.bin" --output "$CROSS/go/tensor_tiled_bf16.wft1" --dtype bf16 --shape 4,4 --tile-shape 2,2
  "$GO" encode --mode tensor-tiled --input "$CROSS/src/tensor_tiled_fp32.bin" --output "$CROSS/go/tensor_tiled_fp32.wft1" --dtype f32 --shape 2,2,2 --tile-shape 1,2,2
  "$GO" encode --mode float-tiled --input "$CROSS/src/float_tiled_bf16.bin" --output "$CROSS/go/float_tiled_bf16.wft1" --dtype bf16 --shape 4,4 --tile-shape 2,2
  "$GO" encode --mode float-tiled --input "$CROSS/src/float_tiled_fp32.bin" --output "$CROSS/go/float_tiled_fp32.wft1" --dtype fp32 --shape 2,2,2 --tile-shape 1,2,2
  "$GO" encode --mode tensor --input "$CROSS/src/sparse_base_bf16.bin" --output "$CROSS/go/sparse_base_bf16.wft1" --dtype bf16 --shape 2,3
  "$GO" encode --mode sparse-delta --input "$CROSS/src/sparse_target_bf16.bin" --output "$CROSS/go/sparse_delta_bf16.wft1" --base "$CROSS/go/sparse_base_bf16.wft1" --dtype bf16 --shape 2,3
  "$GO" encode --mode tensor --input "$CROSS/src/sparse_base_q8.bin" --output "$CROSS/go/sparse_base_q8.wft1" --dtype i8 --shape 4,8
  "$GO" encode --mode sparse-delta --input "$CROSS/src/sparse_target_q8.bin" --output "$CROSS/go/sparse_delta_q8.wft1" --base "$CROSS/go/sparse_base_q8.wft1" --dtype q8_0 --shape 4,8
  "$GO" encode --mode tensor --input "$CROSS/src/sparse_base_fp32.bin" --output "$CROSS/go/sparse_base_fp32.wft1" --dtype f32 --shape 2,2
  "$GO" encode --mode sparse-delta --input "$CROSS/src/sparse_target_fp32.bin" --output "$CROSS/go/sparse_delta_fp32.wft1" --base "$CROSS/go/sparse_base_fp32.wft1" --dtype fp32 --shape 2,2
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
      weight_delta_bf16) args+=(--base "$CROSS/rust/weight_base_bf16.wft1") ;;
      weight_delta_fp32) args+=(--base "$CROSS/rust/weight_base_fp32.wft1") ;;
      weight_delta_q8) args+=(--base "$CROSS/rust/weight_base_q8.wft1") ;;
      semantic_delta_bf16) args+=(--base "$CROSS/rust/semantic_base_bf16.wft1") ;;
      semantic_delta_fp32) args+=(--base "$CROSS/rust/semantic_base_fp32.wft1") ;;
      semantic_delta_q8) args+=(--base "$CROSS/rust/semantic_base_q8.wft1") ;;
      sparse_delta_bf16) args+=(--base "$CROSS/rust/sparse_base_bf16.wft1") ;;
      sparse_delta_q8) args+=(--base "$CROSS/rust/sparse_base_q8.wft1") ;;
      sparse_delta_fp32) args+=(--base "$CROSS/rust/sparse_base_fp32.wft1") ;;
    esac
    "$GO" "${args[@]}"
  done
  for set in sharded_q8 sharded_q4 sharded_q8_single; do
    mkdir -p "$CROSS/go-decoded/$set"
    for frame in "$CROSS/rust/$set"/*.wft1; do
      name=$(basename "$frame" .wft1)
      "$GO" decode --input "$frame" --output "$CROSS/go-decoded/$set/$name.bin"
    done
  done
  echo "go-decode: decoded $(find "$CROSS/go-decoded" -name '*.bin' | wc -l) Rust frames (including sharded)"
}

rust_decode() {
  mkdir -p "$CROSS/rust-decoded"
  for mode in $(frame_modes); do
    args=(decode --input "$CROSS/go/$mode.wft1" --output "$CROSS/rust-decoded/$mode.bin")
    case "$mode" in
      reference|delta|dictionary) args+=(--deps "$CROSS/go/deps") ;;
      weight_delta_bf16) args+=(--base "$CROSS/go/weight_base_bf16.wft1") ;;
      weight_delta_fp32) args+=(--base "$CROSS/go/weight_base_fp32.wft1") ;;
      weight_delta_q8) args+=(--base "$CROSS/go/weight_base_q8.wft1") ;;
      semantic_delta_bf16) args+=(--base "$CROSS/go/semantic_base_bf16.wft1") ;;
      semantic_delta_fp32) args+=(--base "$CROSS/go/semantic_base_fp32.wft1") ;;
      semantic_delta_q8) args+=(--base "$CROSS/go/semantic_base_q8.wft1") ;;
      sparse_delta_bf16) args+=(--base "$CROSS/go/sparse_base_bf16.wft1") ;;
      sparse_delta_q8) args+=(--base "$CROSS/go/sparse_base_q8.wft1") ;;
      sparse_delta_fp32) args+=(--base "$CROSS/go/sparse_base_fp32.wft1") ;;
    esac
    "$RUST" "${args[@]}"
  done
  for set in sharded_q8 sharded_q4 sharded_q8_single; do
    mkdir -p "$CROSS/rust-decoded/$set"
    for frame in "$CROSS/go/$set"/*.wft1; do
      name=$(basename "$frame" .wft1)
      "$RUST" decode --input "$frame" --output "$CROSS/rust-decoded/$set/$name.bin"
    done
  done
  echo "rust-decode: decoded $(find "$CROSS/rust-decoded" -name '*.bin' | wc -l) Go frames (including sharded)"
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
  for mode in raw raw_empty raw_chunked reference delta structured tensor tokens quantized float_tensor float_tensor_single float_tensor_uniform weight_delta_bf16 weight_delta_fp32 weight_delta_q8 kv_cache_p0 kv_cache_p1 kv_prov_p0 activation_c0 activation_c1 tensor_tiled_bf16 tensor_tiled_fp32 float_tiled_bf16 float_tiled_fp32 sparse_delta_bf16 sparse_delta_q8 sparse_delta_fp32; do
    if cmp -s "$CROSS/rust/$mode.wft1" "$CROSS/go/$mode.wft1"; then
      echo "PASS identical $mode"
    else
      echo "FAIL identical $mode"; failures=$((failures + 1))
    fi
  done

  echo "== TENSOR_SHARDED (mode 10) shard sets =="
  for set in sharded_q8 sharded_q4 sharded_q8_single; do
    for frame in "$CROSS/rust/$set"/*.wft1; do
      name=$(basename "$frame")
      base=${name%.wft1}
      if cmp -s "$CROSS/rust/$set/$name" "$CROSS/go/$set/$name"; then
        echo "PASS identical tensor_sharded $set/$name"
      else
        echo "FAIL identical tensor_sharded $set/$name"; failures=$((failures + 1))
      fi
      if cmp -s "$CROSS/go-decoded/$set/$base.bin" "$CROSS/rust-decoded/$set/$base.bin"; then
        echo "PASS decode-equivalent tensor_sharded $set/$base"
      else
        echo "FAIL decode-equivalent tensor_sharded $set/$base"; failures=$((failures + 1))
      fi
    done
    if cmp -s "$CROSS/rust/$set.root" "$CROSS/go/$set.root"; then
      echo "PASS identical logical root tensor_sharded $set"
    else
      echo "FAIL identical logical root tensor_sharded $set"; failures=$((failures + 1))
    fi
  done

  echo "== KV_CACHE (mode 12) / ACTIVATION_STREAM (mode 13) byte identity =="
  for mode in kv_cache_p0 kv_cache_p1 activation_c0 activation_c1; do
    if cmp -s "$CROSS/rust/$mode.wft1" "$CROSS/go/$mode.wft1"; then
      echo "YES byte-identical $mode"
    else
      echo "NO  byte-identical $mode"; failures=$((failures + 1))
    fi
  done

  echo "== KV_CACHE_PROV (mode 19) byte identity =="
  for mode in kv_prov_p0; do
    if cmp -s "$CROSS/rust/$mode.wft1" "$CROSS/go/$mode.wft1"; then
      echo "YES byte-identical $mode"
    else
      echo "NO  byte-identical $mode"; failures=$((failures + 1))
    fi
  done

  echo "== SEMANTIC_DELTA (mode 14) byte identity =="
  for mode in semantic_delta_bf16 semantic_delta_fp32 semantic_delta_q8; do
    if cmp -s "$CROSS/rust/$mode.wft1" "$CROSS/go/$mode.wft1"; then
      echo "YES byte-identical $mode"
    else
      echo "NO  byte-identical $mode"; failures=$((failures + 1))
    fi
  done

  echo "== TENSOR_TILED (mode 15) byte identity =="
  for mode in tensor_tiled_bf16 tensor_tiled_fp32; do
    if cmp -s "$CROSS/rust/$mode.wft1" "$CROSS/go/$mode.wft1"; then
      echo "YES byte-identical $mode"
    else
      echo "NO  byte-identical $mode"; failures=$((failures + 1))
    fi
  done

  echo "== FLOAT_TILED (mode 17) byte identity =="
  for mode in float_tiled_bf16 float_tiled_fp32; do
    if cmp -s "$CROSS/rust/$mode.wft1" "$CROSS/go/$mode.wft1"; then
      echo "YES byte-identical $mode"
    else
      echo "NO  byte-identical $mode"; failures=$((failures + 1))
    fi
  done

  echo "== SPARSE_DELTA (mode 16) byte identity =="
  for mode in sparse_delta_bf16 sparse_delta_q8 sparse_delta_fp32; do
    if cmp -s "$CROSS/rust/$mode.wft1" "$CROSS/go/$mode.wft1"; then
      echo "YES byte-identical $mode"
    else
      echo "NO  byte-identical $mode"; failures=$((failures + 1))
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
