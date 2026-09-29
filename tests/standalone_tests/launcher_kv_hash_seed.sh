#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/launcher.sh"
test_dir=$(mktemp -d)
trap 'rm -rf "$test_dir"' EXIT
LOG_DIR="$test_dir"
STATE_FILE="$test_dir/start-manager.state"
KV_ROOT="$test_dir/cache"

# Verify the variable a child actually inherits, not just launcher state.
unset PYTHONHASHSEED
KV_PYTHONHASHSEED=0
configure_kv_hash_seed
[[ "$(bash -c 'printf %s "$PYTHONHASHSEED"')" == 0 ]]

PYTHONHASHSEED=17
configure_kv_hash_seed
[[ "$(bash -c 'printf %s "$PYTHONHASHSEED"')" == 17 ]]
save_manager_state
unset PYTHONHASHSEED KV_PYTHONHASHSEED
load_manager_state
[[ "$(bash -c 'printf %s "$PYTHONHASHSEED"')" == 17 ]]

# Absence must survive a restart too: exporting an empty seed prevents Python
# from starting, and picking a new default would invalidate old cache hashes.
unset PYTHONHASHSEED KV_PYTHONHASHSEED
configure_kv_hash_seed
[[ "$(bash -c 'printf %s "${PYTHONHASHSEED-unset}"')" == unset ]]
save_manager_state
export PYTHONHASHSEED=123
load_manager_state
[[ "$(bash -c 'printf %s "${PYTHONHASHSEED-unset}"')" == unset ]]

KV_ROOT=""
KV_PYTHONHASHSEED=0
configure_kv_hash_seed
[[ "$(bash -c 'printf %s "${PYTHONHASHSEED-unset}"')" == unset ]]
echo launcher_kv_hash_seed_ok
