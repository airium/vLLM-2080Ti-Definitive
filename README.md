# Qwen3.8-27B on two RTX 2080 Ti 22 GiB cards

Language: English | [简体中文](README.zh-CN.md)

This repository is a personal derivative of [vLLM 2080 Ti Definitive
Edition](https://github.com/weicj/vLLM-2080Ti-Definitive) by
[weicj](https://github.com/weicj), itself a hardware-focused fork of
[vLLM](https://github.com/vllm-project/vllm). Read the parent project's
README, hardware guides, and license for its supported routes, build
background, and upstream attribution. This README covers only the additional
local Qwen3.8-27B deployment work.

The changes here add three route profiles under `profiles/airium/` for
[Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) and
[NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) with
[DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2),
allow host-configured GPU -> pageable CPU -> filesystem KV offload, and add a
bounded filesystem cache index with eviction and active-transfer protection.
The offload work and its focused tests are available for others to study or
borrow; it has not been submitted as a pull request to the parent project.
Keep the upstream vLLM license and the parent fork's attribution when
redistributing a derivative.

## Deployment Guide

The tested host has two NVLink-connected RTX 2080 Ti 22 GiB cards, TP=2/PP=1,
Python 3.12, PyTorch 2.13.0+cu130, and a compatible CUDA 13 toolchain. The
profiles are not capacity guarantees for other GPUs or driver/toolkit
combinations. Use the parent project's build instructions to prepare a clean
SM75 environment. Install `uv` and the Hugging Face `hf` CLI before running
the commands below.

### Get The Runtime And Model

Clone this derivative as `vllm-2080ti-definitive`, then download the official
FP8 safetensors checkpoint. The current Hugging Face CLI uses `hf download`;
run `hf auth login` first if your account needs access.

```bash
git clone https://github.com/airium/vLLM-2080Ti-Definitive.git vllm-2080ti-definitive
cd vllm-2080ti-definitive
export REPO_PATH="$PWD"
export MODEL_DIR="$HOME/models/Qwen3.8-27B-FP8"
hf download Qwen/Qwen3.8-27B-FP8 --local-dir "$MODEL_DIR"
```

Set the CUDA toolkit for this host, build the runtime, and check the profiles:

```bash
export CUDA_HOME=/usr/local/cuda-13.3  # Change to your installed toolkit.
./build.sh
bash tools/validate_profiles.sh
```

Rebuild after updating the fork or changing the PyTorch/CUDA ABI. The build
must include the SM75 FlashQLA extension used by the Qwen GDN prefill path.

### Choose A Route

The following is a starting configuration for the full-context TurboQuant
route. Edit the GPU indices, artifact paths, CPU memory, and disk watermarks
for your workstation. Run it in the same shell as the launcher commands below.
The selected profile owns route settings such as KV precision, context length,
speculative decoding, five-image admission, prefix caching, and launch mode.

```bash
export PROFILE=airium/qwen3.8-27b-2x22g-w8a16-tqk8v4-262144-mtp4-text-image5.env
export PROFILE_FILE="$REPO_PATH/profiles/$PROFILE"
export GPU_DEVICES=0,1                 # Physical NVLink-connected GPU pair.
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TP_SIZE=2 PP_SIZE=1
export PORT=8000 SERVICE_SCOPE=local   # Use lan only on a trusted network.
export SERVED_NAME=qwen-qwen3.8-27b  # API model id; customize as needed.
unset MODE                             # Keep the profile's validated mode.

# Optional pageable CPU and persistent filesystem KV tiers.
export KV_OFFLOAD_ENABLED=true
export KV_ROOT_BASE="$HOME/vllm-kv"
if [[ "$KV_OFFLOAD_ENABLED" == true ]]; then
  PROFILE_BASENAME=${PROFILE##*/}
  export KV_DISK_CACHE_DIR="$KV_ROOT_BASE/${PROFILE_BASENAME%.env}"
else
  export KV_DISK_CACHE_DIR=""
fi
export KV_DISK_CPU_BYTES=$((32 * 1024 * 1024 * 1024)) # 32 GiB CPU staging/hot cache
unset KV_ROOT KV_CPU_GIB KV_PYTHONHASHSEED # Clear legacy aliases from earlier shells.
export KV_PIN_MEMORY=false
export PYTHONHASHSEED=0
export KV_DISK_MAX_GIB=160
export KV_DISK_TARGET_GIB=128
export KV_DISK_MIN_FREE_GIB=0
export KV_DISK_CLEANUP_INTERVAL_SECONDS=600
export VLLM_TQ_RESERVE_PREFILL_WORKSPACE=0

# Keep runtime outputs outside the source tree.
export LOG_DIR="$REPO_PATH/../vllm-artifacts/run-logs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_PATH/../vllm-artifacts/torchinductor-cache"
export TRITON_CACHE_DIR="$REPO_PATH/../vllm-artifacts/triton-cache"
```

A nonempty `KV_DISK_CACHE_DIR` enables the CPU/filesystem connector; it does **not**
increase the active GPU context capacity. The example watermarks are
host-specific, not minimum requirements. The CPU tier consumes substantial
RAM and `/dev/shm`; check `MemAvailable`, `df -h /dev/shm`, and free disk
space before starting. `KV_PIN_MEMORY=false` avoids registering a large
shared mmap with CUDA on the tested dual-rank host. Use a separate cache root
per route, and use `PYTHONHASHSEED=0` across restarts that should share
cache blocks. Set the actual process variable `PYTHONHASHSEED`, not a
profile-only alias. The launcher also accepts the legacy `KV_PYTHONHASHSEED`
alias when `PYTHONHASHSEED` is unset, and saves the actual seed for managed
restarts. Changing the seed makes existing block hashes incompatible, so the
first request after a seed change must populate a new prefix.

The runtime base is the parent fork's `v0.2.2-post3`. `KV_ROOT` and
`KV_CPU_GIB` are compatible aliases for its `KV_DISK_CACHE_DIR` and
`KV_DISK_CPU_BYTES` interface; do not supply conflicting directories or
capacities. Both interfaces retain checkpoint-content fingerprinting and use
one connector with the optional pinning and bounded-storage settings above.
Cache files are stored below `checkpoint-<fingerprint>` within the configured
root, and watermarks apply to that active namespace. Existing unfingerprinted
cache files are not automatically migrated, so the new namespace needs an
initial cold fill. Replacing model or separate draft contents changes the
fingerprint; remove obsolete namespaces only when no server uses them.

For the FP8-KV route, switch the profile and initially test with GPU-only KV:

```bash
export PROFILE=airium/qwen3.8-27b-2x22g-w8a16-fp8kv-225280-mtp3-text-image5.env
export PROFILE_FILE="$REPO_PATH/profiles/$PROFILE"
unset MODE # Use this profile's normal mode.
export KV_OFFLOAD_ENABLED=false
export KV_DISK_CACHE_DIR=""
unset VLLM_TQ_RESERVE_PREFILL_WORKSPACE
```

The FP8-KV profile defaults to `normal`; the TQK8V4 profile defaults to
`fast`. FP8-KV retains MTP3; TQK8V4 uses MTP4. Both have an image limit of five. Filesystem offload is
also possible with FP8 KV, but validate it on the target host with a distinct
cache root and suitable CPU/disk limits before using it as a daily route.
Host settings such as GPU selection, port, model path, and offload policy do
not belong in these route profiles.

For NVFP4/DFlash2, download both checkpoints, select the matching profile,
and set the separate draft path. Model paths remain host settings, not profile
fields. The SM75 runtime uses weight-only Marlin W4A16; this is not native
FP4 activation computation. The checkpoint also contains some FP8 layers.

```bash
export NVFP4_MODEL_DIR="$HOME/models/Qwen3.8-27B-NVFP4"
export DFLASH2_MODEL_DIR="$HOME/models/Qwen3.8-27B-DFlash2"
hf download unsloth/Qwen3.8-27B-NVFP4 --local-dir "$NVFP4_MODEL_DIR"
hf download incoai/Qwen3.8-27B-DFlash2 --local-dir "$DFLASH2_MODEL_DIR"
export MODEL_DIR="$NVFP4_MODEL_DIR"
export SPECULATIVE_MODEL="$DFLASH2_MODEL_DIR"
export PROFILE=airium/qwen3.8-27b-2x22g-w4a16-fp8kv-262144-dflash7-text-image5.env
export PROFILE_FILE="$REPO_PATH/profiles/$PROFILE"
unset MODE VLLM_TQ_RESERVE_PREFILL_WORKSPACE
export KV_OFFLOAD_ENABLED=true
PROFILE_BASENAME=${PROFILE##*/}
export KV_DISK_CACHE_DIR="$KV_ROOT_BASE/${PROFILE_BASENAME%.env}"
```

DFlash2 uses `SPECULATIVE_METHOD=dflash` with seven draft tokens. The
launcher selects Model Runner V2 and FP16 draft KV; target KV remains FP8.
The target understands image inputs; the draft receives text-only inputs.
With independent DFlash2 pools, persistent offload stores and restores target
KV only; transient draft KV is rebuilt for each admission. Upgrading from a
draft-inclusive cache layout selects a new file-mapper layout namespace and
requires one cold fill; existing files are retained.
When returning to an FP8/MTP route, restore `MODEL_DIR` to the FP8 checkpoint
and clear `SPECULATIVE_MODEL`. Route checks below do not establish general
NVFP4/FP8 quality equivalence or reproduce the parent host's 200+ tok/s claim.

`SERVED_NAME` controls the model id exposed by `/v1/models` and the value that
clients send in the API request's `model` field. It does not rename the local
checkpoint directory or change the selected route.

### Inspect And Start

```bash
test -f "$MODEL_DIR/config.json"
test -f "$PROFILE_FILE"
# With 32 GiB CPU staging, reserve at least 48 GiB MemAvailable and 32 GiB free /dev/shm.
# Check these again if the staging allocation changes; also check free space in the cache root.
awk '/^MemAvailable:/ {print "MemAvailable GiB:", int($2 / 1024 / 1024)}' /proc/meminfo
df -h /dev/shm
if [[ -n "$KV_DISK_CACHE_DIR" ]]; then
  mkdir -p "$KV_DISK_CACHE_DIR"
  df -h "$KV_DISK_CACHE_DIR"
fi
./launcher.sh --non-interactive --print-config
./launcher.sh --non-interactive
```

Confirm the resolved model, physical GPU pair, TP/PP layout, KV precision,
context, MTP setting, and mode before the real start. The launcher writes a
log and PID under `LOG_DIR`, waits for `/health`, and runs a small **text**
smoke test. Its optional 4K/128 plus 32K/512 benchmark is synthetic; a pass
does not prove image quality or maximum-context stability. Skip it when you
only need to bring up the service.

With `SERVICE_SCOPE=local`, the API is available at
`http://127.0.0.1:$PORT/v1`. `SERVICE_SCOPE=lan` also binds a LAN
interface; this setup adds no authentication or TLS, so do not expose the
port to the public internet.

### Check An Image Request

The launcher smoke test does not send an image. This separate request uses
the repository's small test image and reports both the answer and token
accounting. `max_tokens` limits generated output, **not** image resolution
or the number of visual tokens.

```bash
python3 - "$PORT" "$REPO_PATH/tests/multimodal/assets/image1.png" <<'PY'
import base64
import json
import sys
import urllib.request

port, image_path = sys.argv[1:]
base_url = f"http://127.0.0.1:{port}/v1"
with urllib.request.urlopen(f"{base_url}/models", timeout=20) as response:
    model = json.load(response)["data"][0]["id"]
with open(image_path, "rb") as image_file:
    image = base64.b64encode(image_file.read()).decode("ascii")
payload = {
    "model": model,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image in one short sentence."},
        {"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{image}"
        }},
    ]}],
    "max_tokens": 64,
    "temperature": 0,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}
request = urllib.request.Request(
    f"{base_url}/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    result = json.load(response)
print(result["choices"][0]["message"]["content"])
print("finish_reason:", result["choices"][0]["finish_reason"])
print("usage:", result["usage"])
PY
```

For routine checks, use `curl -fsS "http://127.0.0.1:$PORT/health"`,
`/v1/models`, `/metrics`, the launch log, and `nvidia-smi`. Use the
interactive `./launcher.sh` service manager to stop the service before
changing routes or restarting. Compiler caches may shorten a restart; GPU
prefix KV and CPU staging contents are process-local; matching filesystem
cache blocks can survive a restart. Never remove an active cache root or
shared mmap while a service is using it.

Management wrappers should reuse the launcher's process-group/token-aware
shutdown rather than killing only the API PID. Load the previous service's
state in a subshell so it cannot overwrite edited settings for the next start;
do not proceed with a restart when shutdown fails. Only remove a stopped
service's recorded CPU staging mmap after confirming no process maps or opens
it. Keep the filesystem KV cache and compiler caches intact.

## Route Rules

A profile is a route preset, not a complete deployment or a promise that
every similarly named checkpoint works. Keep host settings outside the
profile. Promote a route only after real HTTP requests finish and pass
quality checks; startup, health, an empty stream, or a tiny smoke request
alone is not capacity or throughput evidence. Record the exact checkpoint,
weight and KV precision, MTP setting, context, graph mode, TP/PP layout, and
measurement method with each benchmark. Historical CUDA 12.8 / PyTorch 2.11
results do not validate the CUDA 13 route.

`normal` is the default mode once its route is validated. `fast` also
requires quality and non-eager CUDA Graph checks; `safe` is a diagnostic
fallback, not a performance recommendation. FP16/default KV is a quality
reference, while FP8 and TurboQuant KV need their own correctness and
capacity evidence. Hybrid Qwen3.8 Mamba `align` prefix caching and the
offload connector remain experimental interfaces; retest after changing the
model, runtime, topology, filesystem, or hash seed.

## Validation Boundary

These are local dual-22-GiB host results on the post3 source base, using TP2,
Mamba `align`, one active sequence, five-image admission, seed zero, and a
32 GiB pageable CPU/filesystem tier. FP8 route checks used
`orcarouter/Qwen3.8-27B-Uncensored-FP8`. NVFP4 checks distinguish the
unsloth target from `orcarouter/Qwen3.8-27B-Uncensored-NVFP4`, both with the
incoai DFlash2 draft. Public download examples select official Qwen FP8 and
unsloth NVFP4; the private wrapper may select the uncensored variants.

| Profile under `profiles/airium/` | Mode | Batch tokens | GPU util | GPU KV tokens |
| --- | --- | ---: | ---: | ---: |
| `qwen3.8-27b-2x22g-w8a16-tqk8v4-262144-mtp4-text-image5.env` | fast | 2048 | 0.95 | 263,990 |
| `qwen3.8-27b-2x22g-w8a16-fp8kv-225280-mtp3-text-image5.env` | normal | 2048 | 0.9825 | 228,168 |
| `qwen3.8-27b-2x22g-w4a16-fp8kv-262144-dflash7-text-image5.env` | fast | 1024 | 0.98 | 270,336 |

TQ MTP4 failed admission at `0.94`, so `0.95` is the minimum at 0.01
resolution. For the uncensored NVFP4 checkpoint, `0.9735` failed and
`0.9736` passed (0.0001 resolution, repeated startup). The earlier unsloth
NVFP4 threshold was `0.9461`, not a universal NVFP4 setting: its LM head is
FP8, while the uncensored checkpoint leaves that head in BF16. The latter
adds about 0.59 GiB per GPU under TP2 (startup weights/draft footprint
13.82 versus 13.23 GiB), leaving less KV memory at the same utilization.
The shipped preset rounds this to `GPU_UTIL=0.98` at 0.01 precision, as
requested for deployment; the finer boundary above is investigation evidence.
Revalidate after changing checkpoints. FP8 MTP4/batch1024 failed both fast and
normal admission at `0.9825`, including warm compiler-cache retries; that
profile retains the previous MTP3/batch2048/normal configuration.

TQ and NVFP4 each completed a 261,632-token prompt plus all 512 fixed output
tokens (262,144 total), HTTP 200 with no OOM. FP8's unchanged route passed
fresh startup and quality checks; its 224,768+512 near-limit result belongs
to the earlier validation record. All three routes passed arithmetic,
reasoning, instruction, Python-function, five-image, and image-semantics smoke
checks. Images were tested separately, not inside the near-limit requests. The
uncensored NVFP4 full-context run used the measured boundary `0.9736`; the
rounded `0.98` preset separately passed the actual private-wrapper startup
(270,336 KV tokens), text/vision/reasoning checks, and image smoke test.
These bounded checks do not establish general model quality equivalence.

| Route | 4K/128 prefill / decode (tok/s) | 32K/512 prefill / decode (tok/s) |
| --- | ---: | ---: |
| FP8 weights / TQK8V4 / MTP4 | 1601.59 / 103.47 | 1396.62 / 72.18 |
| Unsloth NVFP4 / FP8 KV / DFlash2-7 (earlier 0.9461 run) | 1352.40 / 215.90 | 1211.26 / 204.36 |
| Uncensored NVFP4 / FP8 KV / DFlash2-7 (0.9736) | 1345.37 / 204.17 | 1205.90 / 193.98 |

The reference lane temporarily disabled prefix caching and the connector,
excluded warm-up, used the median of three sequential exact 4,096/128
requests, and completed one exact 32,768/512 request. Prompts were filler and
outputs were restricted to the single ` the` token; these synthetic decode
rates are not ordinary-chat throughput guarantees. Capacity/persistence
checks used the connector and prefix caching enabled.

Fresh-process TQ reuse restored 46,816 of 49,152 tokens, read 1,459,847,168
filesystem bytes, and returned in 4.35s. The earlier unsloth NVFP4 restart restored
46,144 of 49,152 tokens in 10.35s with filesystem reads and zero GPU prefix
hits; this is tied to that checkpoint and test configuration.
Quota/cleaner regression tests passed; watermark eviction was not stressed
in these live runs. The CPU/filesystem tier preserves matching prefixes,
not conversation history, and does not expand active GPU context.

### Diagnosing apparently missing disk reuse

File creation proves that stores ran, not that a later request reused them. For a restart test, keep the model, route, hash seed, and tokenized prefix unchanged, let pending stores complete, and stop all server workers before relaunching. Check `usage.prompt_tokens_details.cached_tokens` alongside `vllm:kv_offload_tiering_read_bytes_total` for the filesystem tier; a fast same-process repeat alone could be a GPU or CPU hit. Engine statistics must be enabled to expose connector counters.

Client changes to system instructions, tool definitions or ordering, cache salt, or earlier history invalidate subsequent prefix hashes. Existing disk files cannot restore KV for a different prompt. Compare request prefixes before diagnosing an offload failure. Connector DEBUG logging reports prefix-block fingerprints without prompt text or token IDs.

This derivative keeps the parent project's license and project identity.
For supported hardware and other profiles, refer to
[vLLM 2080 Ti Definitive Edition](https://github.com/weicj/vLLM-2080Ti-Definitive);
for base engine behavior, refer to [vLLM](https://github.com/vllm-project/vllm).
