# HumDial × MiniCPM-o 4.5：单机 8 卡 serving 压测

这套入口固定使用随机种子 `20260901`、实时 `200 ms` 音频块和每个请求率约
`300 s` 的开环会话流量。入口默认只打印命令；只有显式添加 `--execute` 才会启动
服务并使用 GPU。

## 拓扑

| 名称 | Thinker | Talker | Code2Wav | 物理放置 |
| --- | ---: | ---: | ---: | --- |
| `separated_422` | 4 | 2 | 2 | 三阶段分离 |
| `downstream_colocated` | 4 | 4 | 4 | Thinker 独占 0–3，下游两阶段共置于 4–7 |
| `thinker6_downstream_colocated` | 6 | 2 | 2 | Thinker 独占 0–5，下游两阶段共置于 6–7 |
| `full_colocated` | 8 | 8 | 8 | 每张卡共置三个阶段 |

`thinker6_downstream_colocated` 是针对 Thinker 计算瓶颈的探测拓扑：相比
`downstream_colocated` 增加 Thinker 副本，同时把 Talker 和 Code2Wav 收缩到两张卡并保持
共置。它同时改变了副本数和物理放置，不是单变量放置对照；正式结论应与相同副本数的
拓扑另行对比。两张下游卡的显存可行性可由此前下游共置每卡约 48–50 GiB 的实测余量
支持，但下游计算吞吐会变成需要重点观察的新瓶颈。

四份拓扑都继承
[`configs/humdial_8gpu/base_h200_b32_fullgraph256.yaml`](configs/humdial_8gpu/base_h200_b32_fullgraph256.yaml)，
不依赖 `intermediate/` 下的历史配置。Stage 0/1 使用生产 CUDA Graph；Stage 2
保留模型要求的外层 eager 路径，HiFT/CFM 内部 CUDA Graph 已启用，CFM cache
上限为 256。因此当前结果属于非 PD 的生产 serving 路径，不能表述为 PD 分离容量。

### Code2Wav CUDA Graph 预热范围

当前 `thinker6_downstream_colocated` 配置的每个 Code2Wav replica 会在启动时固定捕获
HiFT 的 7 个 batch bucket × 2 个 cache 形状，即 14 张；两个 replica 共 28 张。CFM
按完整 tensor shape（包含滚动 attention cache）建图，数量随会话形状变化，并不存在对任意
流量都成立的“全部张数”。基于真实 `humdial_900` trace，启动预热现在以 batch `[1]`
覆盖首包残余 `[9,10,13,18,23,27]`、稳定输入 `[28]`（25 个新 codec frame + 3 个
left-context）和尾包 `[6,9,12,19,22,28]`；同时将默认 reference WAV 按 native-duplex
的 float32/mono/16 kHz 临时 WAV 方式物化，避免 prompt mel shape 偏移。未覆盖 shape
禁止在线 lazy capture，直接 eager fallback。`cfm_max_graphs: 256` 是每个 replica 的缓存
上限，不是启动时必须捕获的 512 张图。

如果把整条当前 `thinker6_downstream_colocated_async_stage1` 链路也计入，已有启动日志中
Stage 0 的每个 replica 是 7 张 PIECEWISE + 5 张 FULL，6 个 replica 共 72 张；Stage 1
的 2 个 replica 共 24 张。因此固定启动图合计是 `72 + 24 + 28 = 124` 张，CFM 动态
缓存另有最多 `256 × 2 = 512` 个 slot，不能把它们等同于必须预先捕获的 512 张。

## 先 dry-run

下面只打印三条命令，不启动服务：

```bash
/app/vllm-omni/.venv/bin/python \
  benchmarks/minicpmo/run_humdial_8gpu_sweeps.py \
  --topology all \
  --campaign humdial_8gpu \
  --attempt attempt_001
```

默认请求率是 `0.025,0.05,0.1` session/s，每个测点 300 秒，热身 60 秒。
每个拓扑内部遇到第一个 SLO 边界即停止更高测点，但 `--topology all` 会继续执行
下一个拓扑。若显存仍未打满且所有预设测点都通过，应新建 attempt 并提高
`--rates`，直到 SLO/OOM/硬件饱和边界。

## 获批后执行

GPU 实验必须先按机器流程申请提权/资源确认，然后在同一条命令末尾添加：

```bash
--execute
```

也可以单独跑一个拓扑，例如：

```bash
/app/vllm-omni/.venv/bin/python \
  benchmarks/minicpmo/run_humdial_8gpu_sweeps.py \
  --topology downstream_colocated \
  --campaign humdial_8gpu \
  --attempt attempt_002 \
  --execute
```

输出目录固定为：

```text
intermediate/humdial_minicpmo_serving/<campaign>/<topology>/<attempt>/
```

默认启动路径在一个进程内创建全部副本。若希望跨 GPU 并行加载，可显式启用
stage-based 启动器：API head 和每个 stage replica 使用独立进程；同一 stage 的
不同 replica 并发加载，共置在同一 GPU 的 stage 按 wave 串行，避免启动期显存
profiling 互相干扰。

```bash
/app/vllm-omni/.venv/bin/python \
  benchmarks/minicpmo/run_humdial_8gpu_sweeps.py \
  --topology thinker6_downstream_colocated \
  --campaign humdial_8gpu \
  --attempt attempt_parallel_001 \
  --parallel-stage-launch \
  --execute
```

并行启动会在 attempt 目录下写入 `stage_logs/`，并在测试结束时合并到
`server.log`。`--omni-master-port` 默认为 `26000`，需要与 API `--port` 不同。

`attempt` 目录只要已经存在（即使为空）就会被拒绝。中断、服务失败或客户端结果
缺失的目录必须原样保留为 partial，后续运行使用新的 attempt，禁止恢复后混写。
正式执行前，底层 benchmark 还会连续检查 8 张 GPU 的空闲状态 30 秒。

## Stage 1 async scheduler A/B

为了只改变 Talker 的调度器，使用独立配置
`configs/humdial_8gpu/thinker6_downstream_colocated_async_stage1.yaml`：Stage 1
开启 `async_scheduling`，Stage 0/2 和其余资源参数保持 baseline 不变。独立脚本默认
跑 `0.025,0.05,0.1` 三个请求率、每点 300 秒，并使用并行 stage 启动：

```bash
DRY_RUN=1 benchmarks/minicpmo/run_humdial_8gpu_async_stage1.sh attempt_async_stage1_preview

benchmarks/minicpmo/run_humdial_8gpu_async_stage1.sh attempt_async_stage1_001
```

脚本会拒绝已存在的 attempt 目录；可通过 `RATES`、`DURATION_S`、`GPUS` 等环境变量
覆盖默认参数。上面的第一条只打印命令，不启动服务。

## Full-duplex E2E 交互测量

`humdial_arrival_rate.py` 仍是 L1 serving 压测：它只上传一段固定音频，不能判断
“播放到哪里以后被打断、第二轮是否沿用上下文”。需要 full-duplex E2E 时，使用独立的
`humdial_e2e.py`。它从 manifest 采样 case，并为每个 session 执行：初始用户音频 →
播放 ACK 锚点（L2）或外部时钟（L1）→ `barge_in` → 第二段用户音频 → follow-up
response。每个 session 写一个 `*.events.jsonl`，另写 `sessions.jsonl` 和
`summary.json`；`task_success` 只有在 manifest 提供的 `expected_text_contains` 全部
命中时才为 true，缺少标签会计为 unknown，不会伪装成成功。

manifest 最小格式见
[`humdial_e2e_manifest.example.json`](humdial_e2e_manifest.example.json)。先只生成计划：

```bash
/app/vllm-omni/.venv/bin/python \
  benchmarks/minicpmo/humdial_e2e.py \
  --manifest /path/to/humdial_e2e.json \
  --feedback-contract L2 \
  --request-rate 0.025 \
  --duration-s 300 \
  --output-dir intermediate/humdial_e2e/attempt_001 \
  --schedule-only
```

服务已启动后再去掉 `--schedule-only`。L2 的 `playback_anchor_ms` 是模拟浏览器播放
光标（不是物理扬声器测量）；如果要做仅外部时钟的对照，将 contract 改成 `L1`，此时
使用 case 的 `external_interrupt_ms`。summary 中的 `session_slo_pass_rate` 是逐 session
E2E 达成率，`goodput_session_window_fraction` 是按 case 的 session window 加权的
goodput；它们与 token 级 latency/RTF 分开报告。

### 从 HumDial 标注生成 paired manifest

HumDial 的每个非-clean WAV 都有同名 JSON，`speech_segments` 提供带时间的用户语音段。
生成器会从每个至少包含两段语音的 item 截取前两段：第一段作为 initial turn，第二段
作为 interrupt turn；不会把不同 item 拼成一条 session，也不会加速或重采样原始音频。
单段的 `pause` item 会被跳过，并在生成结果的 `skipped_source_rows` 中记录。

下面的命令生成按 scenario 每类 100 条、语言尽量均衡的固定子集（当前数据可生成 9 类、
共 900 条），同时写出裁剪后的 WAV 和 manifest：

```bash
/app/vllm-omni/.venv/bin/python \
  benchmarks/minicpmo/generate_humdial_e2e_manifest.py \
  --dataset-root /mnt/nvme1n1/ml_research/linbinbin1/src-omni-modal/Humdial-Track2-Test \
  --output-dir intermediate/humdial_e2e/humdial_900_seed20260901 \
  --per-scenario 100 \
  --seed 20260901
```

HumDial 原始标注不包含 assistant 的标准答案，所以生成的 manifest 默认将
`expected_text_contains` 留空；这类 case 会正常测量 transport/interaction/playout，但
task correctness 会是 `unknown`。如果有独立的语义评测标签，可通过 `--labels labels.json`
注入，JSON 格式为 `{ "en/ask/0007_0011": ["expected phrase"] }`，key 也可以使用
manifest 中的 `dataset_relative_path`。

Native duplex 开启 `auto_response` 后，模型可能在整段输入上传完成前就开始输出；此时
E2E 计时器不会把较晚到达的 input commit 当作起点，而会回退到该 response 的
`response.created`。每个 response metric 的 `input_commit_used_as_timing_origin` 会标明
采用了哪一种口径，避免产生负的 TTFP/TTFT。

如果希望沿用 8 卡 async-stage1 的服务启动、遥测和 attempt 管理，可以直接切换该
launcher 的 client：

```bash
CLIENT_MODE=e2e \
E2E_MANIFEST=/path/to/humdial_e2e.json \
FEEDBACK_CONTRACT=L2 \
RATES=0.025 DURATION_S=300 \
benchmarks/minicpmo/run_humdial_8gpu_async_stage1.sh attempt_async_stage1_e2e_001
```

`CLIENT_MODE` 默认仍为 `arrival`，所以既有 async/no-async A/B 命令的 workload 不变。

## 只读汇总

汇总器既不启动 GPU，也不修改实验目录：

```bash
/app/vllm-omni/.venv/bin/python \
  benchmarks/minicpmo/summarize_humdial_sweeps.py \
  intermediate/humdial_minicpmo_serving/humdial_8gpu
```

需要保留全部 per-GPU 字段时使用 `--format json`。`complete + slo_fail` 表示一次
完整测量正常触发 SLO，是有效容量边界；`partial` 表示中断、run error、客户端
非零退出或缺少完整请求计数，不能用于拓扑比较。

所有工具默认不计算文件、日志、数据集或状态的 SHA/hash，不验证 hash chain，
也不在启动、恢复、运行或结束时重建历史状态做 hash 校验。
