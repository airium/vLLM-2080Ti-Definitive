# 双 RTX 2080 Ti 22 GiB 部署 Qwen3.8-27B

语言：[English](README.md) | 简体中文

本仓库是 [weicj](https://github.com/weicj) 的
[vLLM 2080 Ti Definitive Edition](https://github.com/weicj/vLLM-2080Ti-Definitive)
的个人派生版本；该项目本身是针对 SM75 硬件的
[vLLM](https://github.com/vllm-project/vllm) fork。其他受支持路线、构建背景及原始
项目署名，请先阅读上游项目的 README、硬件指南和许可证。本文只介绍额外的本地
Qwen3.8-27B 部署工作。

这里新增了面向官方 [Qwen3.8-27B-FP8 权重](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
的两条路线 Profile，允许在主机侧配置 GPU -> 可分页 CPU 内存 -> 文件系统的 KV
offload，并增加了带容量限制、淘汰策略和活跃传输保护的文件系统缓存索引。相关源码
和专项测试公开供其他人参考或借鉴；这些改动没有作为 PR 提交到上游项目。再发布派生
版本时，请保留 vLLM 的许可证及原 fork 的项目署名。

## 部署指南

实测主机使用两张经 NVLink 连接的 22 GiB RTX 2080 Ti，TP=2/PP=1、
Python 3.12、PyTorch 2.13.0+cu130 和兼容的 CUDA 13 工具链。Profile 的容量
数字不保证适用于其他显卡、驱动或工具链组合。准备全新的 SM75 环境时，请同时参考
上游项目的构建说明。执行下文命令前，请安装 `uv` 和 Hugging Face `hf` CLI。

### 获取运行时与模型

将本派生仓库克隆为 `vllm-2080ti-definitive`，再下载官方 FP8 safetensors
权重。当前 Hugging Face CLI 使用 `hf download`；如需访问授权，先执行
`hf auth login`。

```bash
git clone https://github.com/airium/vLLM-2080Ti-Definitive.git vllm-2080ti-definitive
cd vllm-2080ti-definitive
export REPO_PATH="$PWD"
export MODEL_DIR="$HOME/models/Qwen3.8-27B-FP8"
hf download Qwen/Qwen3.8-27B-FP8 --local-dir "$MODEL_DIR"
```

按主机实际安装位置设置 CUDA 工具链，构建运行时并校验 Profile：

```bash
export CUDA_HOME=/usr/local/cuda-13.3  # 改成主机上实际安装的工具链路径。
./build.sh
bash tools/validate_profiles.sh
```

更新 fork 或更换 PyTorch/CUDA ABI 后需要重新构建。构建结果必须包含 Qwen GDN
prefill 路径使用的 SM75 FlashQLA 扩展。

### 选择路线

下面的配置以完整上下文的 TurboQuant 路线为起点。请根据自己的工作站调整 GPU
序号、产物目录、CPU 内存容量和磁盘水位。配置块与后面的 launcher 命令必须在同一个
shell 中执行。所选 Profile 负责 KV 精度、上下文长度、MTP3、五图准入、prefix
cache 和启动模式等路线参数。

```bash
export PROFILE=qwen3.8-27b-2x22g-w8a16-tqk8v4-262144-mtp3-text-image5.env
export PROFILE_FILE="$REPO_PATH/profiles/$PROFILE"
export GPU_DEVICES=0,1                 # 选择通过 NVLink 连接的一对物理 GPU。
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TP_SIZE=2 PP_SIZE=1
export PORT=8000 SERVICE_SCOPE=local   # 仅在可信局域网中使用 lan。
export SERVED_NAME=qwen-qwen3.8-27b  # API 模型 ID，可按需修改。
unset MODE                             # 使用 Profile 已验证的模式。

# 可选的可分页 CPU tier 和持久化文件系统 KV tier。
export KV_ROOT="$HOME/vllm-kv/$PROFILE"
export KV_CPU_GIB=32
export KV_PIN_MEMORY=false
export PYTHONHASHSEED=0
export KV_DISK_MAX_GIB=160
export KV_DISK_TARGET_GIB=128
export KV_DISK_MIN_FREE_GIB=0
export KV_DISK_CLEANUP_INTERVAL_SECONDS=600
export VLLM_TQ_RESERVE_PREFILL_WORKSPACE=0

# 运行产物保存在源码目录之外。
export LOG_DIR="$REPO_PATH/../vllm-artifacts/run-logs"
export TORCHINDUCTOR_CACHE_DIR="$REPO_PATH/../vllm-artifacts/torchinductor-cache"
export TRITON_CACHE_DIR="$REPO_PATH/../vllm-artifacts/triton-cache"
```

`KV_ROOT` 非空时会启用 CPU/文件系统 connector，但**不会**增大 GPU 上可同时
处理的活跃上下文容量。示例中的磁盘水位是主机配置，不是最低硬件要求。CPU tier
会占用大量内存及 `/dev/shm`；启动前检查 `MemAvailable`、
`df -h /dev/shm` 和磁盘可用空间。`KV_PIN_MEMORY=false` 可避免在实测双
rank 主机上向 CUDA 注册过大的共享 mmap。不同路线应使用各自的 `KV_ROOT`；
若希望重启后复用缓存，必须保持 `PYTHONHASHSEED` 一致。这里设置的是真正的
进程环境变量 `PYTHONHASHSEED`，不是 Profile 别名。

切换到 FP8-KV 路线时，建议先只使用 GPU KV 做验证：

```bash
export PROFILE=qwen3.8-27b-2x22g-w8a16-fp8kv-225280-mtp3-text-image5.env
export PROFILE_FILE="$REPO_PATH/profiles/$PROFILE"
export KV_ROOT=""
unset VLLM_TQ_RESERVE_PREFILL_WORKSPACE
```

FP8-KV Profile 默认使用 `normal`，TQK8V4 Profile 默认使用 `fast`。两者都
保留 MTP3 和最多五张图片的准入限制。FP8 KV 也可以配合文件系统 offload，但在
作为日常路线使用前，应在目标主机上以独立的缓存目录和合适的 CPU/磁盘限额重新验证。
GPU 选择、端口、模型路径和 offload 策略等主机参数不应写入路线 Profile。

`SERVED_NAME` 控制 `/v1/models` 暴露的模型 ID，以及客户端请求中 `model` 字段应
填写的值。它不会重命名本地 checkpoint 目录，也不会改变所选路线。

### 检查并启动

```bash
test -f "$MODEL_DIR/config.json"
test -f "$PROFILE_FILE"
./launcher.sh --non-interactive --print-config
./launcher.sh --non-interactive
```

正式启动前，请确认最终生效的模型、物理 GPU 组合、TP/PP 布局、KV 精度、上下文、
MTP 设置和运行模式。Launcher 会将日志和 PID 写入 `LOG_DIR`，等待
`/health` 可用，并执行一个简短的**纯文本** smoke 测试。可选的 4K/128 加
32K/512 benchmark 是合成测试；通过它不能证明图像质量或最大上下文的稳定性。
若只是启动服务，可跳过该 benchmark。

`SERVICE_SCOPE=local` 时，API 位于 `http://127.0.0.1:$PORT/v1`。
`SERVICE_SCOPE=lan` 还会绑定局域网接口；此配置不提供身份验证或 TLS，
不要将端口暴露到公网。

### 检查图片请求

Launcher 自带的 smoke 测试不发送图片。下面的独立请求使用仓库内的小型测试图片，
同时显示回复和 token 统计。`max_tokens` 限制的是**生成输出**，不是图片分辨率，
也不是输入图片的视觉 token 数。

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

日常检查可使用 `curl -fsS "http://127.0.0.1:$PORT/health"`、
`/v1/models`、`/metrics`、启动日志和 `nvidia-smi`。更换路线或重启前，
通过交互式 `./launcher.sh` 服务管理菜单停止服务。编译缓存可缩短后续启动，
但 GPU prefix KV 只在当前进程内有效；匹配的 CPU/文件系统缓存块可以跨重启保留。
服务仍在使用时，不要移除 KV 目录或共享 mmap 文件。

## 路线准入与证据

Profile 是路线预设，不是完整部署配置；名称相似的 checkpoint 也不能直接视为
已验证。主机参数应保留在 Profile 之外。正式推荐路线前，需要真实 HTTP 请求
完成并通过质量检查；仅有启动、health、空 stream 或极短 smoke 请求，不足以证明
容量或吞吐。每条 benchmark 都应记录确切的 checkpoint、权重和 KV 精度、MTP
设置、上下文长度、CUDA Graph 模式、TP/PP 布局及测量方法。历史 CUDA 12.8 /
PyTorch 2.11 数据不能直接验证 CUDA 13 路线。

`normal` 路线通过相应验证后可作为默认部署模式。`fast` 还须通过质量检查和
非 eager CUDA Graph 验证；`safe` 是诊断回退，不是性能推荐。FP16/default
KV 是质量参考；FP8 和 TurboQuant KV 应分别提供正确性与容量证据。Qwen3.8
hybrid Mamba `align` prefix cache 与 offload connector 仍是实验性接口；
更换模型、运行时、拓扑、文件系统或 hash seed 后应重新验证。

## 验证边界

下表是**本地主机结果**，并非适用于所有硬件的 Profile 保证。近上限测试使用
确定性的 filler prompt 和受限输出 token，检查的是运行时内存行为与输出交付，
而不是模型回答质量。测试发生在较早的 `0.2.2.dev6` checkout 上，早于本仓库
rebase 到上游 fork 的 `v0.2.2-post2`；在不同 checkout 上发布新的容量结论前，
应重复验证。

| Profile 路线 | KV / 模式 | GPU 2/3、TP=2 的主机实测 |
| --- | --- | --- |
| `qwen3.8-27b-2x22g-w8a16-tqk8v4-262144-mtp3-text-image5.env` | TQK8V4 / fast | `GPU_UTIL=0.948` 时启动分配 264,016 个 GPU KV token。261,632-token prompt 返回 509 个固定 token（合计 262,141），HTTP 200，无 OOM。`0.947` 无法通过启动容量检查，估计最长 261,888 token。 |
| `qwen3.8-27b-2x22g-w8a16-fp8kv-225280-mtp3-text-image5.env` | FP8 / normal | `GPU_UTIL=0.9825` 时启动分配 228,168 个 GPU KV token。224,768-token prompt 完整返回 512 个固定 token（合计 225,280），HTTP 200，无 OOM。 |

两个 Profile 均设置 MTP3、单个活跃 sequence、Mamba `align` prefix cache，
并允许最多五张图片。独立的五图、reasoning 和 prefix 重复请求检查使用的是
较早的路线配置；上表的近上限固定 token 测试没有图片输入。特别是 TQ 上限测试
只返回 509 个 token，**不能**称作完整的 512-token 输出。更早的 FP8-KV 三层
测试使用 `KV_CPU_GIB=28` 和 40/32 GiB 文件系统水位，覆盖 cold population、
external promotion、重启复用和 cleaner 淘汰。当前 `main` rebase 测过 FP8
文件系统缓存复用，但没有在水位上压力测试 cleaner 淘汰。CPU/文件系统 overlay
保留匹配的 prefix，不保存对话历史，也不扩展 GPU 活跃上下文容量。

本派生版本保留上游项目的许可证和项目身份。其他受支持硬件及 Profile 见
[vLLM 2080 Ti Definitive Edition](https://github.com/weicj/vLLM-2080Ti-Definitive)；
基础引擎行为见 [vLLM](https://github.com/vllm-project/vllm)。
