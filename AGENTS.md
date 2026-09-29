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
For documentation-only changes, review the affected text and links and run
`git diff --check`; a rebuild or GPU experiment is not required.

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

## Local Derivative Workflow

The root `README.md` and `README.zh-CN.md` are the public,
deployment-first guides for the local Qwen3.8 work; do not recreate a separate
workspace setup guide. Keep the English and Chinese READMEs aligned when
changing commands, profile names, host settings, or validation claims.
The public repository URL for this derivative is
`https://github.com/airium/vLLM-2080Ti-Definitive`; do not replace it with the
parent `weicj` repository in local clone or publishing instructions.

The public guides use `Qwen/Qwen3.8-27B-FP8` or
`unsloth/Qwen3.8-27B-NVFP4` with `incoai/Qwen3.8-27B-DFlash2` on two
NVLink-connected RTX 2080 Ti 22 GiB cards. The private
workstation wrapper may select a different checkpoint or host configuration;
read its current settings rather than substituting the public example or an
old archive snapshot. Toolchain requirements belong in the deployment guides.
Custom profiles live under `profiles/airium/`. Their filenames record the
actual weight/KV route, exact context length, speculative method, and image
admission. Keep filenames, the validation matrix, public guides, and private
wrapper selections aligned when moving or changing a profile. A mode-less
profile uses the launcher default (`fast`); retain `MODE=normal` when that is
the route's validated graph policy.

Profiles contain route parameters only. GPU selection, model path, port,
service scope, CPU/filesystem KV storage, CUDA toolkit, and compiler-cache
paths belong to the launcher or host environment. Validate the complete
profile matrix with `bash tools/validate_profiles.sh`; it includes the current
upstream matrix plus the custom profiles under `airium/`.

### Private Wrapper And Workspace Records

When this checkout is in the local LLM workspace, also follow `../AGENTS.md`.
The sibling `../vllm-2080ti-definitive.sh` is the user's private management
wrapper, not a publishable repository file. Keep shared setup and launch logic
synchronized with both public READMEs when changing the launcher interface.
Public examples must remain portable and must not acquire personal paths or
private network/download requirements.

Keep host-editable settings together and use variables in later commands.
Preserve current user configuration, honor the selected profile's default mode
unless explicitly overridden, and clear incompatible inherited aliases.
Managed stop/status operations may inspect saved launch state, but that state
must not overwrite edited configuration for the next start. Failed shutdown
must prevent a replacement start. Cleanup must not remove unrelated or still
mapped/open shared-memory buffers, or confuse CPU staging with persistent KV
files.

Investigation notes, rebase records, measured results, and selected evidence
belong in a descriptive subfolder of
`../vllm-2080ti-definitive-archives/`, with an entry in its root `README.md`.
Active logs, PIDs, manager state, and compiler caches belong in the workspace's
runtime artifact directory. Keep dated incident sections out of `AGENTS.md`;
this file records durable contracts, not session history.

### Git Workflow

- `origin/main` is the parent `weicj` fork's rebase baseline; the `airium`
  remote is the public derivative. Branch tracking status against `airium/main`
  is not a measure of distance from the parent baseline.
- Preserve local committed, unstaged, and relevant untracked work before a
  rebase, using backup refs and appropriate file/patch snapshots.
- Keep the local KV-offload derivative as **one commit on top of the latest
  fetched `origin/main`**, including authorized tracked changes. Its message
  remains exactly `feat(kv-offload): add bounded filesystem tier and Qwen routes`.
- Sign that commit with GPG and verify a good signature. Do not silently fall
  back to an unsigned commit. Amending or soft-resetting the local commit is
  authorized for this workflow; preserve user work and verify the final diff.
- After a rebase/amend, check the exact subject, signature, clean tracked
  worktree, and `git rev-list --left-right --count origin/main...HEAD` (`0 1`).
- Do not push branches, tags, or releases without an explicit user request.

### KV Offload Contract

The launcher translates these host variables into the vLLM
`OffloadingConnector`/`TieringOffloadingSpec` configuration when
`KV_DISK_CACHE_DIR` (or its legacy root alias) is nonempty:

- `KV_DISK_CACHE_DIR`: filesystem cache root; use a separate root per model/route.
- `KV_DISK_CPU_BYTES`: CPU primary-tier capacity in bytes.
- `KV_PIN_MEMORY=false`: required by the validated dual-rank host to avoid
  registering the large shared mmap with CUDA.
- `PYTHONHASHSEED=0`: required for disk persistence; the legacy
  `KV_PYTHONHASHSEED` alias is translated only when the real variable is absent.
  The real seed is exported and saved for managed restarts.
- `KV_DISK_MAX_GIB`, `KV_DISK_TARGET_GIB`, `KV_DISK_MIN_FREE_GIB`, and
  `KV_DISK_CLEANUP_INTERVAL_SECONDS`: optional filesystem cleanup policy.

`KV_ROOT`/`KV_CPU_GIB` are compatible aliases for upstream's
`KV_DISK_CACHE_DIR`/`KV_DISK_CPU_BYTES`. Conflicting roots or capacities are
rejected, not silently preferred. Both interfaces use one connector with
upstream checkpoint-content fingerprinting; separate draft weights participate
when applicable. The active filesystem namespace is beneath
`checkpoint-<fingerprint>`. Old unfingerprinted files are not auto-migrated.
Quota/index cleanup applies to that active checkpoint namespace. Interactive
offload/prefix-cache disable clears both interfaces so a saved alias cannot
silently re-enable persistence.

The filesystem tier uses a persistent SQLite index, content-addressed block
groups, a probationary/protected eviction policy, cross-process cleanup locking,
and leases for active lookups and asynchronous transfers. Cleanup evicts all
rank/group files belonging to a content hash together. The index and cache are
generated runtime state and must not be committed.

Independent draft pools rebuilt by `IsolatedKVCacheCoordinator` are not
reusable prefix artifacts. Exclude them consistently from scheduler/worker
offload groups and from bounds on target-cache hits; preserve bounds for
non-offloaded cacheable target groups such as HiSparse sources.

`KV_PIN_MEMORY` is propagated from the tiering spec to every CPU offload worker;
do not regress this propagation. The filesystem tier must continue to support
partial-load failures, stale-file reconciliation, restart reuse, and orderly
shutdown of its cleanup thread and I/O pool.

Cache reuse requires the same tokenized prefix, including chat serialization,
system prompts, tool definitions, and relevant cache-key inputs. Client-side
tool selection can change an early prefix without a backend fault. Cache file
growth and fast same-process retries alone do not prove disk persistence;
validate an unchanged request after a fresh server process, and correlate
external-cache hit telemetry with disk I/O and TTFT. Record the checkpoint,
route, request, and configuration in the archive rather than turning one
client-setting experiment into a guaranteed workaround.

### Validation Baseline

Use the relevant static checks under **Validation Before Publishing**.

For KV-offload changes, the focused baseline is:

```bash
.venv/bin/python -m pytest -q tests/v1/kv_offload/tiering
.venv/bin/python -m pytest -q tests/v1/kv_offload/cpu/test_gpu_worker.py
```

The focused suites exercise CPU-worker transfers and tier lifecycle behavior.
The filesystem-index tests cover group/rank reconciliation,
probationary eviction, active and expired leases, stale-file cleanup, quota
watermarks, minimum-free-space cleanup, and disabled-cleanup behavior.

For launcher offload changes, also run the relevant shell checks under
`tests/standalone_tests/`, including native/legacy variable mapping,
bounded-cache policy, and hash-seed handling. For changes to the private wrapper,
verify lifecycle behavior and configuration preservation as well as
syntax/rendered arguments.

Use the deployment guides and workspace archives for measured capacity and
runtime results, not historical numbers copied into agent instructions.
Fixed-token admission checks establish capacity, not quality. Optional
synthetic benchmarks can exhaust transient GDN workspace on tightly fitted
routes; they are not mandatory startup gates. Run potentially service-killing
lanes in an isolated process with a route-appropriate memory policy.

### Publishing Scope

The intended publishable local change set is the KV offload implementation,
its focused tests, launcher wiring, the custom Qwen profiles, the root bilingual
README pair, and durable repository instructions. Do not add caches, model
weights, logs, PIDs, shared-memory mmap files, generated compiler artifacts,
or archived workspace material. Review the complete diff against `origin/main`
before committing, and preserve the upstream fork credit and Apache-2.0 notices.
