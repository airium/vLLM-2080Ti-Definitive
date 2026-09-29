#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/launcher.sh"
fixture=$(mktemp -d)
trap 'rm -rf -- "$fixture"' EXIT
MODEL_DIR="$fixture/model"
mkdir -p "$MODEL_DIR"
printf 'checkpoint bytes\n' > "$MODEL_DIR/model.safetensors"
SPECULATIVE_METHOD=none
KV_ROOT="$fixture/cache"
KV_CPU_GIB=32
KV_PYTHONHASHSEED=0
KV_DISK_MAX_GIB=160
KV_DISK_TARGET_GIB=128
KV_DISK_MIN_FREE_GIB=0
KV_DISK_CLEANUP_INTERVAL_SECONDS=600
ENABLE_PREFIX_CACHING=1
DISABLE_PREFIX_CACHING=0
unset PYTHONHASHSEED KV_DISK_CACHE_DIR KV_DISK_CPU_BYTES KV_PIN_MEMORY
validate_disk_kv_cache_config
[[ "$PYTHONHASHSEED" == 0 ]]
[[ "$KV_DISK_CACHE_DIR" == "$KV_ROOT" ]]
[[ "$KV_DISK_CPU_BYTES" == 34359738368 ]]
prepare_disk_kv_checkpoint_fingerprint
config=$(disk_kv_transfer_config)
python3 - "$config" "$KV_ROOT" "$DISK_KV_CHECKPOINT_FINGERPRINT" <<'PY'
import json
import sys

config = json.loads(sys.argv[1])
extra = config["kv_connector_extra_config"]
assert extra["cpu_bytes_to_use"] == 32 * 1024**3
assert extra["pin_memory"] is False
assert extra["secondary_tiers"] == [{
    "type": "fs",
    "root_dir": sys.argv[2] + "/checkpoint-" + sys.argv[3],
    "max_disk_bytes": 160 * 1024**3,
    "target_disk_bytes": 128 * 1024**3,
    "min_disk_free_bytes": 0,
    "cleanup_interval_seconds": 600,
}]
PY

# Mixed interfaces must not silently select a different root or CPU budget.
if (KV_DISK_CACHE_DIR="$fixture/other"; validate_disk_kv_cache_config) 2>/dev/null; then
  echo 'conflicting cache roots were accepted' >&2; exit 1
fi
if (KV_DISK_CPU_BYTES=4294967296; validate_disk_kv_cache_config) 2>/dev/null; then
  echo 'conflicting CPU capacities were accepted' >&2; exit 1
fi
if (KV_DISK_TARGET_GIB=161; validate_disk_kv_cache_config) 2>/dev/null; then
  echo 'invalid watermarks were accepted' >&2; exit 1
fi
if (PYTHONHASHSEED=17; validate_disk_kv_cache_config) 2>/dev/null; then
  echo 'nonzero disk-cache hash seed was accepted' >&2; exit 1
fi

# There must be exactly one connector, even with both compatible interfaces set.
MODEL_FAMILY=qwen
SERVED_NAME=test
MODE=normal
PORT=8000
TP_SIZE=2
PP_SIZE=1
GPU_UTIL=0.9
MAX_MODEL_LEN=4096
MAX_NUM_SEQS=1
MAX_BATCHED_TOKENS=1024
build_args 127.0.0.1
count=0
for arg in "${VLLM_ARGS[@]}"; do
  [[ "$arg" != --kv-transfer-config ]] || count=$((count + 1))
done
[[ "$count" == 1 ]]

# Interactive disable must clear the old alias as well as the new interface.
save_manager_state() { :; }
menu_select() { printf 'enabled\n'; }
prompt_default() {
  case "$1" in
    'SSD cache directory') printf '%s\n' "$fixture/other" ;;
    'CPU staging bytes') printf '4294967296\n' ;;
  esac
}
edit_disk_kv_cache_menu
[[ -z "$KV_ROOT$KV_CPU_GIB" ]]
[[ "$KV_DISK_CACHE_DIR" == "$fixture/other" ]]
[[ "$KV_DISK_CPU_BYTES" == 4294967296 ]]
validate_disk_kv_cache_config
menu_select() { printf 'disabled\n'; }
edit_disk_kv_cache_menu
[[ -z "$KV_ROOT$KV_CPU_GIB$KV_DISK_CACHE_DIR$KV_DISK_CPU_BYTES" ]]
KV_ROOT="$fixture/cache"
KV_CPU_GIB=32
validate_disk_kv_cache_config
edit_prefix_cache_menu
[[ -z "$KV_ROOT$KV_CPU_GIB$KV_DISK_CACHE_DIR$KV_DISK_CPU_BYTES" ]]

echo launcher_bounded_disk_kv_cache_ok
