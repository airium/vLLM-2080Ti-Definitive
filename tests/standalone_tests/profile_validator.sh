#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
TEST_ROOT=$(mktemp -d)
trap 'rm -rf "$TEST_ROOT"' EXIT

write_valid_profile() {
  local dir=$1
  local name=${2:-nomtp-fp16kv-1x131k-text-only.env}
  mkdir -p "$dir"
  printf '%s\n' \
    'MODEL_FAMILY=qwen35moe' \
    'MODEL_VARIANT=fp8' \
    'QUANTIZATION=fp8' \
    'KV_CACHE_DTYPE=float16' \
    'MAX_MODEL_LEN=131072' \
    'GPU_UTIL=0.96' \
    'MAX_NUM_SEQS=1' \
    'MESSAGE_TYPE=text-only' \
    'SPECULATIVE_METHOD=none' \
    'SPECULATIVE_TOKENS=0' > "$dir/$name"
  printf '%s\n' "$dir/$name"
}

expect_valid() {
  PROFILE_DIR=$1 bash "$ROOT/tools/validate_profiles.sh" >/dev/null
}

expect_invalid() {
  local dir=$1
  local expected=$2
  local output
  if output=$(PROFILE_DIR="$dir" bash "$ROOT/tools/validate_profiles.sh" 2>&1); then
    echo "validator unexpectedly accepted $dir" >&2
    exit 1
  fi
  grep -Fq -- "$expected" <<< "$output" || {
    printf 'expected validator output to contain %q, got:\n%s\n' "$expected" "$output" >&2
    exit 1
  }
}

case_dir="$TEST_ROOT/no-mode"
write_valid_profile "$case_dir" >/dev/null
expect_valid "$case_dir"

for mode in fast normal; do
  case_dir="$TEST_ROOT/mode-$mode"
  profile=$(write_valid_profile "$case_dir")
  printf 'MODE=%s\n' "$mode" >> "$profile"
  expect_valid "$case_dir"
done

for mode in safe aggressive invalid; do
  case_dir="$TEST_ROOT/bad-mode-$mode"
  profile=$(write_valid_profile "$case_dir")
  printf 'MODE=%s\n' "$mode" >> "$profile"
  expect_invalid "$case_dir" 'MODE may be omitted or set to fast/normal'
done

for key in PROFILE_GROUP COMPATIBLE_MODES SERVED_NAME ENABLE_PREFIX_CACHING UNKNOWN_PROFILE_KEY; do
  case_dir="$TEST_ROOT/forbidden-$key"
  profile=$(write_valid_profile "$case_dir")
  printf '%s=1\n' "$key" >> "$profile"
  expect_invalid "$case_dir" "$key is not part of the route-profile schema"
done

case_dir="$TEST_ROOT/duplicate"
profile=$(write_valid_profile "$case_dir")
printf 'MAX_NUM_SEQS=2\n' >> "$profile"
expect_invalid "$case_dir" 'duplicate key MAX_NUM_SEQS'

case_dir="$TEST_ROOT/legacy-dir/fast"
write_valid_profile "$case_dir" >/dev/null
expect_invalid "$TEST_ROOT/legacy-dir" 'fast/normal must be selected by the launcher'

case_dir="$TEST_ROOT/yarn"
profile=$(write_valid_profile "$case_dir" 'yarn-fp16kv-1x131k-text-only.env')
printf 'ENABLE_YARN=1\n' >> "$profile"
expect_valid "$case_dir"

case_dir="$TEST_ROOT/yarn-missing-flag"
write_valid_profile "$case_dir" 'yarn-fp16kv-1x131k-text-only.env' >/dev/null
expect_invalid "$case_dir" 'YaRN profile filename requires ENABLE_YARN=1'

case_dir="$TEST_ROOT/context-mismatch"
profile=$(write_valid_profile "$case_dir")
sed -i 's/MAX_MODEL_LEN=131072/MAX_MODEL_LEN=65536/' "$profile"
expect_invalid "$case_dir" 'filename context does not match MAX_MODEL_LEN'

case_dir="$TEST_ROOT/binary-context-label"
profile=$(write_valid_profile "$case_dir" 'nomtp-fp16kv-1x128k-text-only.env')
expect_invalid "$case_dir" 'filename context does not match MAX_MODEL_LEN'

case_dir="$TEST_ROOT/near-binary-context-label"
profile=$(write_valid_profile "$case_dir" 'nomtp-fp16kv-1x262K-text-only.env')
sed -i 's/MAX_MODEL_LEN=131072/MAX_MODEL_LEN=268288/' "$profile"
expect_invalid "$case_dir" 'filename context does not match MAX_MODEL_LEN'

case_dir="$TEST_ROOT/local-name"
profile=$(write_valid_profile "$case_dir" 'qwen3.8-27b-2x22g-w8a16-fp8kv-131072-mtp4-text-image5.env')
sed -i -e 's/KV_CACHE_DTYPE=float16/KV_CACHE_DTYPE=fp8/' \
  -e 's/MESSAGE_TYPE=text-only/MESSAGE_TYPE=text+image/' \
  -e 's/SPECULATIVE_METHOD=none/SPECULATIVE_METHOD=mtp/' \
  -e 's/SPECULATIVE_TOKENS=0/SPECULATIVE_TOKENS=4/' "$profile"
printf '%s\n' 'MAMBA_CACHE_MODE=align' 'LANGUAGE_MODEL_ONLY=0' \
  'MM_LIMIT_JSON='\''{"image":5,"video":0,"audio":0}'\' >> "$profile"
expect_valid "$case_dir"
sed -i 's/MAX_MODEL_LEN=131072/MAX_MODEL_LEN=65536/' "$profile"
expect_invalid "$case_dir" 'filename exact context does not match MAX_MODEL_LEN'
sed -i 's/MAX_MODEL_LEN=65536/MAX_MODEL_LEN=131072/;s/"image":5/"image":4/' "$profile"
expect_invalid "$case_dir" 'filename image count does not match MM_LIMIT_JSON'
sed -i 's/"image":4/"image":5/;s/SPECULATIVE_METHOD=mtp/SPECULATIVE_METHOD=dflash/' "$profile"
expect_invalid "$case_dir" 'mtp filename requires SPECULATIVE_METHOD=mtp'
mv "$profile" "$case_dir/qwen3.8-27b-2x22g-w8a16-fp8kv-131072-dflash2-text-image5.env"
profile="$case_dir/qwen3.8-27b-2x22g-w8a16-fp8kv-131072-dflash2-text-image5.env"
expect_valid "$case_dir"
sed -i 's/SPECULATIVE_METHOD=dflash/SPECULATIVE_METHOD=mtp/' "$profile"
expect_invalid "$case_dir" 'dflash filename requires SPECULATIVE_METHOD=dflash'

case_dir="$TEST_ROOT/local-dflash-count/airium"
mkdir -p "$case_dir"
cp "$profile" "$case_dir/qwen3.8-27b-2x22g-w8a16-fp8kv-131072-dflash7-text-image5.env"
profile="$case_dir/qwen3.8-27b-2x22g-w8a16-fp8kv-131072-dflash7-text-image5.env"
sed -i 's/SPECULATIVE_METHOD=mtp/SPECULATIVE_METHOD=dflash/;s/SPECULATIVE_TOKENS=4/SPECULATIVE_TOKENS=7/' "$profile"
expect_valid "$TEST_ROOT/local-dflash-count"
sed -i 's/SPECULATIVE_TOKENS=7/SPECULATIVE_TOKENS=6/' "$profile"
expect_invalid "$TEST_ROOT/local-dflash-count" 'filename DFlash token count does not match'

echo "profile_validator_ok"
