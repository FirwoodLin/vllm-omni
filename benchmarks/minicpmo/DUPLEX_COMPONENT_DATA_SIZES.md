# MiniCPM-o 4.5 full-duplex component data sizes

Per-component input formats and data volumes in native full-duplex serving,
one 1 s unit at a time. This is the reference for sizing component-level
benchmarks and interpreting their results. Shapes are derived from the
production code paths referenced inline; the bs=8 column is filled by the
service-level run (`vllm_omni/deploy/minicpmo_4_5_bs8.yaml` +
`benchmark_duplex_service_saturation.py --session-counts 8`).

## Cadence

The duplex unit is **1 second** (`duplex/input.py` coalesces client chunks to
whole model units; `chunk_period_ms` default 1000). 100 ms is only the
embedding granularity inside a unit.

## Per-unit data flow (bs=1, steady unit with one camera frame)

| Edge | Format | Size | Source |
|---|---|---|---|
| client → engine | pcm16 @ 16 kHz, 16000 samples | 32 KB (85 KB base64 on WS) | `duplex/policy.py:24-26` |
| engine → stage0 preprocess | pcm_f32le, 16000 samples | 64 KB | `entrypoints/duplex/serving.py:908` |
| streaming mel → apm | `(1, 80, 102)` first, `(1, 80, 104)` steady, fp32 | 33 KB | `duplex/stage0.py:102-112` |
| apm encoder KV growth | 50 frames × 1024 × 2 (K/V) × 24 layers, bf16 | ~4.8 MB/s cumulative | `minicpmo_4_5_omni_llm.py:4447-4456` |
| audio embeds → LLM | 10 × 4096 bf16 (10 tokens/s) | 80 KB/s | `minicpmo_4_5_omni_llm.py:3934-3936` |
| vision embeds → LLM | 64 × 4096 bf16 per frame (66 tokens/frame) | 512 KB/frame | `minicpmo_4_5_omni_llm.py:3921-3927` |
| thinker → talker handoff | N token ids + N × 4096 fp32 hidden rows | N × 16.4 KB (N ≤ ~28/segment) | `stage_input_processors/minicpmo_4_5_omni.py:882-940` |
| talker decode | 1 codec id/step over 6562-way vocab; 26 steps/chunk | 26 KB logits/step | `minicpmo_4_5_omni_tts.py:750-779` |
| talker → code2wav | 25 codec ids + 3 left-context, int64 | 224 B/chunk (25 tokens/s speech) | `stage_input_processors/minicpmo_4_5_omni.py:348-377` |
| code2wav → client | 24000 samples @ 24 kHz | 96 KB fp32 → 48 KB pcm16 | `minicpmo_4_5_code2wav.py:600,836-841` |

Thinker per-second token budget: a scheduler slot is one token position reserved
by vLLM for append scheduling and KV-cache allocation (not a session slot or a
fixed amount of GPU memory). The steady budget is 79 scheduler slots with a
camera frame (13 audio-only); the first append is larger because it also
reserves session context and optional reference-audio embeddings. The Thinker
produces ~10 decode tokens/s while speaking
(`duplex/runtime.py:80-140`).

## Sequence-length dependencies

- Audio encoder KV grows 50 frames/s and hard-resets at the 1500-frame
  (30 s) boundary — a per-session latency spike every 30 s
  (`minicpmo_4_5_omni_llm.py:4447-4456`).
- Talker re-prepends the previous chunk's codec ids + condition on each
  sliding recompute (`minicpmo_4_5_omni_tts.py:269-393`); context rolls over
  to a one-chunk window at its 4096-token limit.
- Code2Wav flushes a vocoder window per 25 accumulated codec tokens; first
  window gets 3 silence-code left-context tokens.

## Batching model

- Thinker: duplex sessions are resumable vLLM requests batched by the
  scheduler; the audio-encoder path inside is batch-1 per session
  (`minicpmo_4_5_omni_llm.py:4437`).
- Talker: continuous batching across rows; codec repetition penalty is a
  batched kernel (`minicpmo_4_5_omni_tts.py:80-129`).
- Code2Wav: explicit batch dimension, equal-length buckets through
  `decode_batch`, mixed lengths through `decode_ragged_batch`
  (`minicpmo_4_5_code2wav.py:774-787`).

The shipped default deploy admits **4 sessions** (`minicpmo_4_5.yaml`:
`max_sessions: 4`, `max_num_seqs: 4` per stage). `minicpmo_4_5_bs8.yaml`
raises every knob to 8 for saturation benchmarking.

## bs=8 per-unit totals

Per-edge volumes scale linearly with session count (×8 vs the bs=1 column).
Service-level validation (`minicpmo_4_5_bs8.yaml`, H200):

- Closed-loop (1 turn × 2 repeats): 8/8 sessions completed, no OOM, peak GPU
  memory 117.8 GiB (82% of 143.7 GiB), util 100%. TTFT p50 1.69 s / 0.87 s
  (repeat 1 / 2), RTF p50 1.21 / 1.06 (p99 1.17). Stage0 `num_tokens_in`
  mean ~735-740 per append (first turn incl. context + ref audio, above the
  79-slot steady budget), stage1 out ~190 codec tokens per session (≈8 chunks
  × 26). Artifacts: `intermediate/minicpmo45_bs8_duplex/bs8_closed/`.
- Open-loop (natural speak duty, audio-only, single H200 colocated 3-stage):
  the 1 s unit deadline boundary, measured as Stage0 completion lag:

  | sessions @ 1 Hz | Stage0 DDL miss | lag p50/p95 (ms) | steady slope (ms/tick) |
  |---:|---:|---:|---:|
  | 2 | 22% | 485 / 1587 | +51 |
  | 3 | 47% | 804 / 2742 | +12 |
  | 4 | 76% | 1315 / 2600 | +43 |
  | 6 | 82% | 1557 / 4529 | +313 |
  | 8 | 88% | — / 4471 | +325 |
  | 8 @ 0.5 Hz | 8% | — / 1602 | -38 (pass) |

  Throughput backlog starts between 4 and 6 sessions (slope jump +43 →
  +313). But the sharper limit is tail latency: with natural speak duty on a
  colocated single GPU, p95 Stage0 completion lag exceeds the 1 s deadline
  from 2 sessions onward (cross-stage contention: thinker + talker + code2wav
  share one GPU). Listen-only input at B8 @ 1 Hz is clean (p99 ~157 ms,
  08-25 campaign), so the speak path is the differentiator.
  Artifacts: `intermediate/minicpmo45_bs8_duplex/bs{2,3,4,6}_openloop/`,
  `bs8_openloop/`, `bs8_openloop_05hz/`.
- Long session (8 sessions × 30 turns, closed loop): functionally stable —
  8/8 sessions, 240 responses, peak memory 119.6 GiB, no OOM, CFM graph
  cache keeps filling — but real-time playback SLO fails: RTF p50 1.32,
  p95 2.03, p99 2.50; Stage1 gen p99 ~10 s (talker queueing).
  Artifacts: `intermediate/minicpmo45_bs8_duplex/bs8_long30/`.

## Boundary attribution (service-level vs component-level)

The service-level boundaries above and the component solo benchmarks below
pinpoint where the duplex capacity wall actually is:

1. Every component has ≥7x compute headroom at the bs=8 duplex demand, so
   neither the backlog at ≥6 sessions @ 1 Hz nor the tail-DDL failures from
   2 sessions on are component compute walls.
2. The backlog jump between 4 and 6 sessions (slope +43 → +313 ms/tick)
   coincides with Thinker/Talker unit latencies (390-440 ms / 215-235 ms)
   exceeding the per-session share of the 1 s budget once scheduler queueing
   and cross-stage contention on the colocated GPU are added.
3. The tail-DDL boundary from 2 sessions on is a colocated-GPU contention
   effect: listen-only traffic at B8 @ 1 Hz has p99 ~157 ms, so the speak
   path (thinker decode + talker + code2wav sharing the GPU) creates the
   fat tail.
4. Scaling levers, in order of evidence: Stage0 replicas (4:2:2 topology
   reached B8 streaming RTF p99 1.311 in the 08-29 campaign), separating
   Stage1/2 from the colocated GPU, or speak-duty-aware admission.

## Audio encoder streaming measurements (H200, bf16, eager)

`benchmark_encoder_saturation.py --component audio --audio-mode streaming
--stream-seconds 35 --stream-count 8 --warmup 3 --repeats 10`
(`intermediate/minicpmo45_encoder_bench/streaming_b8_35s.json`):

| metric | value |
|---|---|
| steady chunk (1 s unit) p50 / p95 / p99 | 15.3 / 27.9 / 31.0 ms |
| 30 s reset chunk p50 / p95 / p99 | 45.1 / 51.8 / 61.3 ms |
| first chunk median | 11.0 ms |
| aggregate throughput | 57.4 audio-s/s (8-session demand is 8 audio-s/s) |
| peak allocated (8 × 30 s KV + weights) | 23.8 GiB |

The 30 s KV reset spike is ~3x the steady chunk but stays two orders of
magnitude below the 1 s unit deadline.

## Component solo benchmarks (H200, 2026-09-11)

All five components measured in isolation on one H200 each
(`intermediate/minicpmo45_component_solo_20260911/`), against the bs=8
duplex demand of 8 audio-s/s per component:

| Component | Measured | bs=8 demand | Headroom |
|---|---|---|---|
| Vision (vpm+resampler) | saturated at B16: 144.4 slices/s (microbatch-16 platform; B8 = 140.0) | ≤8 slices/s | ~18x |
| Audio encoder (streaming) | steady p99 31.0 ms/chunk, 57.4 audio-s/s, 30 s reset spike p99 61.3 ms | 8 audio-s/s | ~7x |
| Thinker (eager microbench) | unit ~390-440 ms flat across B1-8 and ctx 13/79/256 (decode-bound ~14 ms/token; prefill 28-70 ms). Eager KV is the memory wall: OOM at B4+/kv4096 and everything at kv16384 | compute OK; eager KV does not represent vLLM paged capacity | compute |
| Thinker (production engine, graph) | piecewise prefill + FULL-decode CUDA graphs + paged KV 88.26 GiB (642,672 tokens). Decode 5.14 ms/token B1 → 5.25 ms B16 flat (2.7x faster than eager; 195→3012 tok/s). Prefill scales to max_model_len: B1 207 ms @8k (39.6k tok/s) → 1739 ms @40940 (23.5k tok/s); batched prefill serializes through chunked prefill (B16 @40940 ≈ 14.9 s). KV concurrency bound 15.7x @40k: B8@40960 fits (327k tokens, decode 16 ms/step); B16@40940 exceeds KV → queue/preemption churn (ITL p50 498 ms), still no OOM (`intermediate/minicpmo45_thinker_engine_20260911/thinker_graph.json`, `thinker_graph_long.json`) | 8 units/s: decode 26 tok × ~5.5 ms ≈ 143 ms + 8 prefills ≤52 ms each — well inside 1 s | compute |
| Talker (eager) | unit ~215-235 ms flat across the whole B/cond/KV matrix (serial-step bound, not FLOP bound); 125→1000 tok/s linear in B; B8/kv1024 = 21.6 GiB | 208 tok/s (B2 already satisfies) | compute |
| Code2Wav (HiFT+CFM graphs) | B1 69.3 ms (RTF 0.069) → B8 123.1 ms / 65.0 audio-s/s (p99 170.8) → B16 189.7 ms / 84.3 audio-s/s; graph hits 49 / miss 1 per shape | 8 audio-s/s | ample |

**Interpretation**: no single component saturates at the bs=8 duplex demand
(the tightest is the audio encoder at ~7x headroom). The service-level
backlog at ≥6 sessions @ 1 Hz and the tail-DDL failures from 2 sessions on
come from scheduling/queueing plus three-stage GPU contention, not from any
component's compute wall. Adding Stage0 replicas or splitting Stage1/2 off
the colocated GPU is more effective than optimizing any single kernel.

## Script mapping

| Component | Microbenchmark | Service-level attribution |
|---|---|---|
| Vision encoder | `benchmark_encoder_saturation.py --component vision` | stage 0 metrics via `summarize_duplex_saturation.py` |
| Audio encoder (offline) | `benchmark_encoder_saturation.py --component audio` | stage 0 metrics |
| Audio encoder (duplex streaming) | `benchmark_encoder_saturation.py --component audio --audio-mode streaming` | stage 0 metrics |
| Thinker prefill/decode | `benchmark_thinker_engine.py` (production engine: piecewise prefill, FULL-decode CUDA graphs, paged KV) | stage 0 `vllm_ttft_ms` / `vllm_itls_ms`, `num_tokens_in/out` |
| Talker | `benchmark_talker_saturation.py` (eager; directional lower bound) | stage 1 metrics |
| Code2Wav | `benchmark_code2wav_saturation.py` (HiFT/CFM CUDA graphs optional) | stage 2 metrics + CFM graph hit stats |
| End-to-end duplex | — | `benchmark_duplex_service_saturation.py` (closed/open loop, GPU telemetry) |

Talker microbenchmark runs eager by design: the production CUDA graphs are
vLLM-engine-managed PIECEWISE/FULL captures that a standalone HF path cannot
faithfully reproduce. For the thinker, `benchmark_thinker_engine.py` now
measures the production path directly (thinker-only pipeline
`minicpmo_4_5_thinker_only.yaml`: piecewise-compiled prefill, FULL-decode
CUDA graphs, paged KV) and supersedes the eager numbers above; the Talker
keeps the eager caveat and production-path conclusions still come from the
service-level per-stage attribution.
