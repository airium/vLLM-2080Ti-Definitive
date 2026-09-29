# AGENTS.md

This file governs the whole `vLLM 2080 Ti Definitive Edition` repository.

## Project Identity And Credit

This repository is a hardware-focused fork for dual RTX 2080 Ti / SM75 vLLM
serving. It builds on upstream vLLM and preserves the local runtime work,
profiles, documentation, and benchmark evidence needed to reproduce the 2080 Ti
stack.

If you publish, redistribute, repackage, benchmark, or build a derivative from
this repository, keep clear credit to:

- Upstream vLLM and its original license.
- `vLLM 2080 Ti Definitive Edition`.
- The repository author: `github.com/weicj`.

Do not remove existing attribution, license notices, benchmark provenance, or
project identity text. If you maintain a public derivative, state that it is
based on this project unless the code has been independently replaced.

## Upstream Compatibility

This project remains a fork of upstream vLLM. When changing source files that
come from upstream vLLM:

- Preserve upstream license and copyright notices.
- Prefer small, reviewable patches over broad rewrites.
- Keep SM75/Turing-specific behavior guarded or clearly documented.
- Do not present fork-specific behavior as upstream vLLM behavior.
- If an upstream `AGENTS.md` or contribution instruction applies in a copied
  upstream subtree, follow it as well.

## Runtime And Profile Rules

This repository is organized around validated runtime routes, not generic
benchmark guesses.

- Do not invent context-size, throughput, or support claims without evidence.
- Keep profile files focused on route parameters only. Do not store global
  service settings such as GPU selection, port, chat template, or reasoning
  defaults inside route profiles.
- Use the hardware-specific README files under `profiles/` as the source of
  truth for shipped profiles.
- If adding or promoting a profile, include capacity evidence and throughput
  evidence using the repository's documented benchmark口径.
- Do not keep tiny smoke-only profiles as recommended deployment presets.

## Validation Before Publishing

Before committing or publishing changes, run the relevant subset of:

```bash
bash -n build.sh launcher.sh tools/validate_profiles.sh
bash tools/validate_profiles.sh
python3 -m py_compile <changed-python-files>
git diff --check
```

For launcher/profile changes, also verify `launcher.sh --print-config` for the
affected route and mode. For runtime kernel or graph-policy changes, include a
real benchmark or smoke result that proves the changed path still works.

## Documentation Discipline

- Keep English and Simplified Chinese documentation consistent when both exist.
- Keep benchmark numbers tied to the exact model, KV precision, MTP setting,
  context, and benchmark method.
- Restore or update linked assets when moving documentation. Broken benchmark
  figures are treated as documentation regressions.
- Avoid overstating support. Use precise wording such as `validated`,
  `supported`, `experimental`, or `not promoted` according to the evidence.

## Repository Hygiene

- Do not commit local caches, model weights, logs, temporary workspace state,
  run outputs, or generated native build artifacts.
- Keep `README.md`, `README.zh-CN.md`, `CHANGELOG.md`, `VERSION`, and
  `pyproject.toml` version fallback aligned for releases.
- Release tags and GitHub Releases are separate. Pushing a tag is not enough to
  update the GitHub Release page.

## Current Local Continuation Notes

The repository is currently rebased on `origin/main` at the `v0.2.2-post2`
runtime base. The root `README.md` and `README.zh-CN.md` are the public,
deployment-first guides for the local Qwen3.8 work; do not recreate a separate
workspace setup guide. Keep the English and Chinese READMEs aligned when
changing commands, profile names, host settings, or validation claims.
The public repository URL for this derivative is
`https://github.com/airium/vLLM-2080Ti-Definitive`; do not replace it with the
parent `weicj` repository in local clone or publishing instructions.

The local deployment target is the official `Qwen/Qwen3.8-27B-FP8` safetensors
checkpoint on two NVLink-connected RTX 2080 Ti 22 GiB cards, TP=2/PP=1,
Python 3.12, PyTorch 2.13.0+cu130, and CUDA 13.x. The two root-level custom
profiles are:

- `profiles/qwen3.8-27b-2x22g-w8a16-tqk8v4-262144-mtp3-text-image5.env`: FP8
  weights, TurboQuant K8V4 KV, MTP3, Mamba `align`, five-image admission,
  `MAX_MODEL_LEN=262144`, `GPU_UTIL=0.948`, `MAX_NUM_SEQS=1`, `MODE=fast`.
- `profiles/qwen3.8-27b-2x22g-w8a16-fp8kv-225280-mtp3-text-image5.env`: FP8
  weights and KV, MTP3, Mamba `align`, five-image admission,
  `MAX_MODEL_LEN=225280`, `GPU_UTIL=0.9825`, `MAX_NUM_SEQS=1`,
  `MODE=normal`.

Profiles contain route parameters only. GPU selection, model path, port,
service scope, CPU/filesystem KV storage, CUDA toolkit, and compiler-cache
paths belong to the launcher or host environment. Validate the complete
profile matrix with `bash tools/validate_profiles.sh`; it includes the current
upstream matrix plus these two custom profiles.

### KV Offload Contract

The launcher translates these host variables into the vLLM
`OffloadingConnector`/`TieringOffloadingSpec` configuration when `KV_ROOT` is
nonempty:

- `KV_ROOT`: filesystem cache root; use a separate root per model/route.
- `KV_CPU_GIB`: pageable CPU primary-tier capacity.
- `KV_PIN_MEMORY=false`: required by the validated dual-rank host to avoid
  registering the large shared mmap with CUDA.
- `KV_PYTHONHASHSEED` / process `PYTHONHASHSEED`: keep fixed across restarts
  that should share cache blocks.
- `KV_DISK_MAX_GIB`, `KV_DISK_TARGET_GIB`, `KV_DISK_MIN_FREE_GIB`, and
  `KV_DISK_CLEANUP_INTERVAL_SECONDS`: optional filesystem cleanup policy.

The filesystem tier uses a persistent SQLite index, content-addressed block
groups, a probationary/protected eviction policy, cross-process cleanup locking,
and leases for active lookups and asynchronous transfers. Cleanup evicts all
rank/group files belonging to a content hash together. The index and cache are
generated runtime state and must not be committed.

`KV_PIN_MEMORY` is propagated from the tiering spec to every CPU offload worker;
do not regress this propagation. The filesystem tier must continue to support
partial-load failures, stale-file reconciliation, restart reuse, and orderly
shutdown of its cleanup thread and I/O pool.

### Validation Baseline

Before publishing code changes, run:

```bash
bash -n build.sh launcher.sh tools/validate_profiles.sh
bash tools/validate_profiles.sh
python3 -m py_compile <changed-python-files>
git diff --check
```

For KV-offload changes, the focused baseline is:

```bash
.venv/bin/python -m pytest -q tests/v1/kv_offload/tiering
.venv/bin/python -m pytest -q tests/v1/kv_offload/cpu/test_gpu_worker.py
```

The local baseline has 434 passing tests and one expected skip across those
focused suites. The filesystem-index tests cover group/rank reconciliation,
probationary eviction, active and expired leases, stale-file cleanup, quota
watermarks, minimum-free-space cleanup, and disabled-cleanup behavior.

The exact-total capacity evidence was measured on the pre-rebase `0.2.2.dev6`
runtime and is a reference, not a universal guarantee. TQK8V4 passed
`261632 + 509` tokens at `GPU_UTIL=0.948`; `0.947` failed admission at an
estimated 261888-token maximum. FP8-KV passed `224768 + 512` at `GPU_UTIL=0.9825`.
These are deterministic fixed-token memory checks, not quality tests. The
launcher’s optional synthetic `32K/512` benchmark can exhaust transient GDN
workspace and kill the service on the TQ route; do not treat that lane as a
required startup gate without an isolated process or a route-specific memory
policy.

### Publishing Scope

The intended publishable local change set is the KV offload implementation,
its focused tests, launcher wiring, the two Qwen profiles, and the root bilingual
README pair. Do not add caches, model weights, logs, PIDs, shared-memory mmap
files, generated compiler artifacts, or archived workspace material. Review
the complete diff against `origin/main` before committing, and preserve the
upstream fork credit and Apache-2.0 notices.
