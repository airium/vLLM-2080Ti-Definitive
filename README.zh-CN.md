# 双 RTX 2080 Ti 22 GiB 部署 Qwen3.8-27B

语言：[English](README.md) | 简体中文

本仓库是 [weicj](https://github.com/weicj) 的
[vLLM 2080 Ti Definitive Edition](https://github.com/weicj/vLLM-2080Ti-Definitive)
的个人派生版本；该项目本身是针对 SM75 硬件的
[vLLM](https://github.com/vllm-project/vllm) fork。其他受支持路线、构建背景及原始
项目署名，请先阅读上游项目的 README、硬件指南和许可证。本文只介绍额外的本地
Qwen3.8-27B 部署工作。

这里在 `profiles/airium/` 下新增了面向
[Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) 及
[NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) 配合
[DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) 的三条路线 Profile，允许在主机侧配置 GPU -> 可分页 CPU 内存 -> 文件系统的 KV
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
shell 中执行。所选 Profile 负责 KV 精度、上下文长度、投机解码、五图准入、prefix
cache 和启动模式等路线参数。

```bash
export PROFILE=airium/qwen3.8-27b-2x22g-w8a16-tqk8v4-262144-mtp4-text-image5.env
export PROFILE_FILE="$REPO_PATH/profiles/$PROFILE"
export GPU_DEVICES=0,1                 # 选择通过 NVLink 连接的一对物理 GPU。
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TP_SIZE=2 PP_SIZE=1
export PORT=8000 SERVICE_SCOPE=local   # 仅在可信局域网中使用 lan。
export SERVED_NAME=qwen-qwen3.8-27b  # API 模型 ID，可按需修改。
unset MODE                             # 使用 Profile 已验证的模式。

# 可选的可分页 CPU tier 和持久化文件系统 KV tier。
export KV_OFFLOAD_ENABLED=true
export KV_ROOT_BASE="$HOME/vllm-kv"
if [[ "$KV_OFFLOAD_ENABLED" == true ]]; then
  PROFILE_BASENAME=${PROFILE##*/}
  export KV_DISK_CACHE_DIR="$KV_ROOT_BASE/${PROFILE_BASENAME%.env}"
else
  export KV_DISK_CACHE_DIR=""
fi
export KV_DISK_CPU_BYTES=$((32 * 1024 * 1024 * 1024)) # 32 GiB CPU staging/hot cache
unset KV_ROOT KV_CPU_GIB KV_PYTHONHASHSEED # 清除此前 shell 中的旧别名。
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

`KV_DISK_CACHE_DIR` 非空时会启用 CPU/文件系统 connector，但**不会**增大 GPU 上可同时
处理的活跃上下文容量。示例中的磁盘水位是主机配置，不是最低硬件要求。CPU tier
会占用大量内存及 `/dev/shm`；启动前检查 `MemAvailable`、
`df -h /dev/shm` 和磁盘可用空间。`KV_PIN_MEMORY=false` 可避免在实测双
rank 主机上向 CUDA 注册过大的共享 mmap。不同路线应使用各自的缓存根目录；
若希望重启后复用缓存，应使用 `PYTHONHASHSEED=0`。这里设置的是真正的
进程环境变量 `PYTHONHASHSEED`，不是 Profile 别名。若未设置实际变量，启动器也兼容
旧的 `KV_PYTHONHASHSEED` 别名，并保存实际种子供管理器重启使用。改变种子会改变
缓存块哈希，因此更改后的首次请求需要重新填充前缀缓存。

运行时基于上游 fork 的 `v0.2.2-post3`。`KV_ROOT` 和 `KV_CPU_GIB` 兼容
上游的 `KV_DISK_CACHE_DIR` 和 `KV_DISK_CPU_BYTES` 接口，不应同时提供冲突
的目录或容量。两种接口保留 checkpoint 内容指纹隔离，并使用同一个 connector，
支持前述可选 pinning 与有界存储设置。缓存文件位于配置根目录下的
`checkpoint-<fingerprint>` 中，磁盘水位作用于当前命名空间。已有无指纹缓存不会
自动迁移，新命名空间需进行首次冷填充。替换模型或独立 draft 的内容会改变指纹；
只有在没有服务器使用旧命名空间时才可清理它们。

切换到 FP8-KV 路线时，建议先只使用 GPU KV 做验证：

```bash
export PROFILE=airium/qwen3.8-27b-2x22g-w8a16-fp8kv-225280-mtp3-text-image5.env
export PROFILE_FILE="$REPO_PATH/profiles/$PROFILE"
unset MODE # 使用此 Profile 的 normal 模式。
export KV_OFFLOAD_ENABLED=false
export KV_DISK_CACHE_DIR=""
unset VLLM_TQ_RESERVE_PREFILL_WORKSPACE
```

FP8-KV Profile 默认使用 `normal`，TQK8V4 Profile 默认使用 `fast`。两者都
允许最多五张图片；FP8-KV 保留 MTP3，TQK8V4 使用 MTP4。FP8 KV 也可以配合文件系统 offload，但在
作为日常路线使用前，应在目标主机上以独立的缓存目录和合适的 CPU/磁盘限额重新验证。
GPU 选择、端口、模型路径和 offload 策略等主机参数不应写入路线 Profile。

使用 NVFP4/DFlash2 时，下载两个 checkpoint，选择对应 Profile，并指定独立 draft
路径。模型路径属于主机设置，不写入 Profile。SM75 使用 Marlin W4A16 权重压缩，
不是原生 FP4 激活计算；checkpoint 中也有部分 FP8 层。

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

DFlash2 使用 `SPECULATIVE_METHOD=dflash` 和七个 draft token。启动器选择 Model
Runner V2，draft KV 为 FP16，target KV 仍为 FP8。target 处理图片，draft
仅接收文本。独立 DFlash2 pools 的持久化 offload 只保存和恢复 target KV，
每次准入重建临时 draft KV。从包含 draft 的旧缓存布局升级，会使用新的
file-mapper 布局命名空间，需要首次冷填充；已有文件保留。切回 FP8/MTP 路线时，应将 `MODEL_DIR` 恢复为 FP8 checkpoint 并清除
`SPECULATIVE_MODEL`。下文路线检查不证明 NVFP4 与 FP8 的普遍质量等价，也不代表
复现了上游主机的 200+ tok/s 数据。

`SERVED_NAME` 控制 `/v1/models` 暴露的模型 ID，以及客户端请求中 `model` 字段应
填写的值。它不会重命名本地 checkpoint 目录，也不会改变所选路线。

### 检查并启动

```bash
test -f "$MODEL_DIR/config.json"
test -f "$PROFILE_FILE"
# 32 GiB CPU staging 应预留至少 48 GiB MemAvailable 和 32 GiB 空闲 /dev/shm。
# 更改 staging 容量后重新检查，并检查缓存根目录所在文件系统的剩余空间。
awk '/^MemAvailable:/ {print "MemAvailable GiB:", int($2 / 1024 / 1024)}' /proc/meminfo
df -h /dev/shm
if [[ -n "$KV_DISK_CACHE_DIR" ]]; then
  mkdir -p "$KV_DISK_CACHE_DIR"
  df -h "$KV_DISK_CACHE_DIR"
fi
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
但 GPU prefix KV 和 CPU staging 内容只在当前进程内有效；匹配的文件系统缓存块
可以跨重启保留。服务仍在使用时，不要移除缓存根目录或共享 mmap 文件。

管理包装脚本应复用启动器基于进程组和 token 的停机逻辑，而不是仅终止 API PID。
在子 shell 中加载上一次服务的状态，避免覆盖为下一次启动编辑的配置；停机失败时
不要继续启动。只有确认没有进程映射或打开已停止服务记录的 CPU staging mmap 时，
才可删除它。保留文件系统 KV 缓存和编译缓存。

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

下表是 post3 源码基线上的本地双 22 GiB 主机结果，使用 TP2、Mamba `align`、
单个活跃 sequence、五图准入、哈希种子零，以及 32 GiB 可分页 CPU/文件系统 tier。
FP8 路线实测使用 `orcarouter/Qwen3.8-27B-Uncensored-FP8`。NVFP4 区分 unsloth
target 与 `orcarouter/Qwen3.8-27B-Uncensored-NVFP4`，均配合 incoai DFlash2。
公开下载示例选择官方 Qwen FP8 和 unsloth NVFP4；私人包装脚本可选择 uncensored 版本。

| `profiles/airium/` 中的 Profile | 模式 | Batch token | GPU util | GPU KV token |
| --- | --- | ---: | ---: | ---: |
| `qwen3.8-27b-2x22g-w8a16-tqk8v4-262144-mtp4-text-image5.env` | fast | 2048 | 0.95 | 263,990 |
| `qwen3.8-27b-2x22g-w8a16-fp8kv-225280-mtp3-text-image5.env` | normal | 2048 | 0.9825 | 228,168 |
| `qwen3.8-27b-2x22g-w4a16-fp8kv-262144-dflash7-text-image5.env` | fast | 1024 | 0.98 | 270,336 |

TQ MTP4 在 `0.94` 无法通过准入，因此按 0.01 精度最低为 `0.95`。Uncensored
NVFP4 在 `0.9735` 失败、`0.9736` 成功并重复验证启动，精度为 0.0001。较早的
unsloth NVFP4 最低值为 `0.9461`，不是通用 NVFP4 设置：它的 LM head 为 FP8，
而 uncensored checkpoint 保留 BF16 head。后者在 TP2 下每卡多占约 0.59 GiB，
启动模型/draft 占用从 13.23 增至 13.82 GiB，相同 util 留给 KV 的内存更少。
当前部署预设按 0.01 精度向上取 `GPU_UTIL=0.98`；上面的精确边界属于调查证据。
更换 checkpoint 后应重新验证。FP8 MTP4、batch1024
在 `0.9825` 的 fast 和 normal 模式均失败，包括编译缓存预热后的重试；该 Profile
保留先前的 MTP3、batch2048、normal 配置。

TQ 和 NVFP4 均完成 261,632-token prompt 加完整 512 个固定输出 token，合计
262,144，HTTP 200、无 OOM。未改变路线参数的 FP8 通过新的启动和质量检查；
224,768+512 的近上限结果来自较早的验证记录。三条路线均通过算术、reasoning、
指令遵循、Python 函数、五图计数和图像语义 smoke 检查。图片单独测试，没有放入
近上限请求。Uncensored NVFP4 全上下文测试使用精确边界 `0.9736`；
向上取整的 `0.98` 预设另行通过实际私人包装脚本启动（270,336 KV token）、
文本/图像/reasoning 检查和图片 smoke 测试。这些有界检查不证明模型质量普遍等价。

| 路线 | 4K/128 prefill / decode (tok/s) | 32K/512 prefill / decode (tok/s) |
| --- | ---: | ---: |
| FP8 权重 / TQK8V4 / MTP4 | 1601.59 / 103.47 | 1396.62 / 72.18 |
| Unsloth NVFP4 / FP8 KV / DFlash2-7（较早的 0.9461 测试） | 1352.40 / 215.90 | 1211.26 / 204.36 |
| Uncensored NVFP4 / FP8 KV / DFlash2-7（0.9736） | 1345.37 / 204.17 | 1205.90 / 193.98 |

参考性能测试临时关闭 prefix cache 和 connector，排除 warm-up，取三个串行、精确
4,096/128 请求的中位数，并完整运行一个精确 32,768/512 请求。prompt 为 filler，
输出限制为单个 ` the` token；这些合成 decode 数字不是普通聊天的吞吐保证。
容量和持久化检查则启用 connector 与 prefix cache。

TQ 重启后恢复 49,152-token prompt 中的 46,816 token，读取 1,459,847,168 字节
文件系统数据，4.35 秒返回。较早的 unsloth NVFP4 重启测试恢复其中的
46,144 token，10.35 秒返回，并有实际文件系统读取、零 GPU prefix 命中；
该结果绑定于该 checkpoint 和测试配置。
Quota/cleaner 回归测试通过；本轮在线测试没有在水位上压力验证淘汰。
CPU/文件系统 tier 保留匹配的 prefix，不保存对话历史，也不扩大 GPU 活跃上下文。

### 排查看似未生效的磁盘缓存复用

生成缓存文件只能证明写入发生过，不能证明后续请求复用了它们。验证重启复用时，应保持模型、路线、哈希种子和 token 化后的前缀不变，等待已有写入完成，停止全部服务 worker 后再启动。结合 `usage.prompt_tokens_details.cached_tokens` 与文件系统 tier 的 `vllm:kv_offload_tiering_read_bytes_total` 判断；仅同一进程内的快速重复请求，可能只是 GPU 或 CPU 命中。需启用引擎统计才能暴露 connector 计数器。

客户端更改系统指令、工具定义或顺序、cache salt、较早的历史内容，都会使后续前缀哈希失效。已有磁盘文件不能恢复不同 prompt 的 KV。排查 offload 故障前，应先比较请求前缀。Connector 的 DEBUG 日志记录前缀块指纹，不记录 prompt 文本或 token ID。

本派生版本保留上游项目的许可证和项目身份。其他受支持硬件及 Profile 见
[vLLM 2080 Ti Definitive Edition](https://github.com/weicj/vLLM-2080Ti-Definitive)；
基础引擎行为见 [vLLM](https://github.com/vllm-project/vllm)。
