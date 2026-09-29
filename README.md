# Qwen3.8-27B on two RTX 2080 Ti 22 GiB cards

Language: English | [简体中文](README.zh-CN.md)

This repository is a personal derivative of [vLLM 2080 Ti Definitive
Edition](https://github.com/weicj/vLLM-2080Ti-Definitive) by
[weicj](https://github.com/weicj), itself a hardware-focused fork of
[vLLM](https://github.com/vllm-project/vllm). Read the parent project's
README, hardware guides, and license for its supported routes, build
background, and upstream attribution. This README covers only the additional
local Qwen3.8-27B deployment work.

The changes here add two route profiles for the official
[Qwen3.8-27B-FP8 checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B-FP8),
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
MTP3, five-image admission, prefix caching, and launch mode.

```bash
export PROFILE=qwen3.8-27b-2x22g-w8a16-tqk8v4-262144-mtp3-text-image5.env
export PROFILE_FILE="$REPO_PATH/profiles/$PROFILE"
export GPU_DEVICES=0,1                 # Physical NVLink-connected GPU pair.
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TP_SIZE=2 PP_SIZE=1
export PORT=8000 SERVICE_SCOPE=local   # Use lan only on a trusted network.
export SERVED_NAME=qwen-qwen3.8-27b  # API model id; customize as needed.
unset MODE                             # Keep the profile's validated mode.

# Optional pageable CPU and persistent filesystem KV tiers.
export KV_ROOT="$HOME/vllm-kv/$PROFILE"
export KV_CPU_GIB=32
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

A nonempty `KV_ROOT` enables the CPU/filesystem connector; it does **not**
increase the active GPU context capacity. The example watermarks are
host-specific, not minimum requirements. The CPU tier consumes substantial
RAM and `/dev/shm`; check `MemAvailable`, `df -h /dev/shm`, and free disk
space before starting. `KV_PIN_MEMORY=false` avoids registering a large
shared mmap with CUDA on the tested dual-rank host. Use a separate `KV_ROOT`
per route, and keep `PYTHONHASHSEED` fixed across restarts that should share
cache blocks. Set the actual process variable `PYTHONHASHSEED`, not a
profile-only alias.

For the FP8-KV route, switch the profile and initially test with GPU-only KV:

```bash
export PROFILE=qwen3.8-27b-2x22g-w8a16-fp8kv-225280-mtp3-text-image5.env
export PROFILE_FILE="$REPO_PATH/profiles/$PROFILE"
export KV_ROOT=""
unset VLLM_TQ_RESERVE_PREFILL_WORKSPACE
```

The FP8-KV profile defaults to `normal`; the TQK8V4 profile defaults to
`fast`. Both retain MTP3 and an image limit of five. Filesystem offload is
also possible with FP8 KV, but validate it on the target host with a distinct
cache root and suitable CPU/disk limits before using it as a daily route.
Host settings such as GPU selection, port, model path, and offload policy do
not belong in these route profiles.

`SERVED_NAME` controls the model id exposed by `/v1/models` and the value that
clients send in the API request's `model` field. It does not rename the local
checkpoint directory or change the selected route.

### Inspect And Start

```bash
test -f "$MODEL_DIR/config.json"
test -f "$PROFILE_FILE"
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
prefix KV is process-local, while matching CPU/filesystem cache blocks can
survive a restart. Never remove an active KV root or shared mmap while a
service is using it.

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

These are **local host results**, not universal profile guarantees. The
near-limit runs below used deterministic filler prompts and restricted output
tokens to test memory behavior. They do not establish model quality. They
were measured on the earlier `0.2.2.dev6` checkout before this repository
was rebased onto the parent fork's `v0.2.2-post2` tip; repeat them on a
different checkout before publishing new capacity claims.

| Profile route | KV / mode | Host result on GPUs 2/3, TP=2 |
| --- | --- | --- |
| `qwen3.8-27b-2x22g-w8a16-tqk8v4-262144-mtp3-text-image5.env` | TQK8V4 / fast | At `GPU_UTIL=0.948`, startup allocated 264,016 GPU KV tokens. A 261,632-token prompt returned 509 fixed tokens (262,141 total) with HTTP 200 and no OOM. `0.947` failed startup admission with an estimated 261,888-token maximum. |
| `qwen3.8-27b-2x22g-w8a16-fp8kv-225280-mtp3-text-image5.env` | FP8 / normal | At `GPU_UTIL=0.9825`, startup allocated 228,168 GPU KV tokens. A 224,768-token prompt returned all 512 fixed tokens (225,280 total) with HTTP 200 and no OOM. |

Both profiles specify MTP3, one active sequence, Mamba `align` prefix
caching, and admission for up to five images. The separate five-image,
reasoning, and prefix-repeat checks were performed on earlier route
configurations; the near-limit fixed-token runs above did not include image
input. In particular, 509 generated tokens is **not** a full 512-token
completion at the TQ boundary. The earlier FP8-KV three-tier test used
`KV_CPU_GIB=28` and 40/32 GiB filesystem watermarks and exercised cold
population, external promotion, restart reuse, and cleaner eviction. The
current-main rebase exercised FP8 filesystem reuse but did not stress cleaner
eviction at its watermark. The CPU/filesystem overlay preserves matching
prefixes, not conversation history, and does not expand active GPU context.

This derivative keeps the parent project's license and project identity.
For supported hardware and other profiles, refer to
[vLLM 2080 Ti Definitive Edition](https://github.com/weicj/vLLM-2080Ti-Definitive);
for base engine behavior, refer to [vLLM](https://github.com/vllm-project/vllm).
