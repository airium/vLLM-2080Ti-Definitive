# Profile Guide

Language: English | [简体中文](README.zh-CN.md)

Profiles are `.env` presets for route parameters. They do not select the
checkpoint, GPUs, port, chat template, or reasoning defaults; those remain
launcher settings. Select the target with `MODEL_DIR` and the optional DFlash
draft with `SPECULATIVE_MODEL`.

Profiles are grouped by hardware, model family, and weight format. The shipped
layout is flat below each hardware/model/weight directory. The launcher
selects the startup mode, defaulting to `fast`.

```text
profiles/
  2x2080Ti/   # [profile details and reference performance](2x2080Ti/README.md)
  4xT10/      # [profile details and reference performance](4xT10/README.md)
  2xT10/      # [profile details and reference performance](2xT10/README.md)
  airium/     # Local dual-22-GiB Qwen routes; see the root deployment guide.
```

Profile filenames use `<decoder>-<kv>-<concurrency><context>-<message>.env`.
The context label is the decimal-token value (floor of `MAX_MODEL_LEN / 1000`),
not a binary Ki-token conversion; for example, `262144` is labeled `262K`.
Thus `dflash2-tqk8v4-4x220K-text-only.env` is a DFlash2 route using TQK8V4 KV,
four concurrent requests, 220K decimal-token context per request, and text-only
messages. Decode routing uses `SPECULATIVE_METHOD=none|mtp|dflash` and
`SPECULATIVE_TOKENS`; the defaults are `0`, `3`, and `7`, respectively.
Per-request speculative metrics are a launcher setting:
`PER_REQUEST_SPEC_DECODE_METRICS=none|summary|detailed`.

To write a profile by hand, use one `KEY=value` assignment per line. The
required route fields are `MODEL_FAMILY`, `MODEL_VARIANT`, `QUANTIZATION`,
`KV_CACHE_DTYPE`, `MAX_MODEL_LEN`, `GPU_UTIL`, `MAX_NUM_SEQS`, `MESSAGE_TYPE`,
`SPECULATIVE_METHOD`, and `SPECULATIVE_TOKENS`. Optional route fields are
`MODE`, `MAX_BATCHED_TOKENS`, `ENABLE_YARN`, `DISABLE_PREFIX_CACHING`,
`MAMBA_CACHE_MODE`, `MM_LIMIT_JSON`, `LANGUAGE_MODEL_ONLY`, and
`SKIP_MM_PROFILING`. Omitting `MODE` selects the launcher default (`fast`),
not a previously validated `normal` policy. `MODEL_FAMILY=qwen35` identifies
the dense Qwen3.5 architecture also used by Qwen3.8-27B; it is not the
checkpoint name.

For example, save the following as
`qwen27b/w8a16/mtp4-fp8kv-1x262K-text-only.env`:

```dotenv
MODEL_FAMILY=qwen35
MODEL_VARIANT=fp8
QUANTIZATION=fp8
KV_CACHE_DTYPE=fp8
MAX_MODEL_LEN=262144
GPU_UTIL=0.96
MAX_NUM_SEQS=1
SPECULATIVE_METHOD=mtp
SPECULATIVE_TOKENS=4
MESSAGE_TYPE=text-only
```

Keep the filename and values aligned: `nomtp`, `mtpN`, and `dflash` must match
the speculative method and token count; the `x` count and `K` context in the
filename must match `MAX_NUM_SEQS` and `MAX_MODEL_LEN`; and the KV/message
suffixes must match their corresponding fields. Run
`bash tools/validate_profiles.sh` before using a hand-written profile.

Local `airium/` filenames additionally record exact context tokens and image
admission: `2x22g` describes the GPU pair, not request concurrency;
`text-image5` requires `MM_LIMIT_JSON` to allow five images. Local `dflashN`
labels count `SPECULATIVE_TOKENS` (for example, `dflash7` means seven tokens),
with `SPECULATIVE_METHOD=dflash`. Select the draft variant separately with
the host setting `SPECULATIVE_MODEL`; the local seven-token route uses DFlash2.
Official `dflash2-*` filenames identify that draft variant, not two tokens.

After selecting a profile, use `./launcher.sh --print-config` to inspect the
resolved route before starting the service.
