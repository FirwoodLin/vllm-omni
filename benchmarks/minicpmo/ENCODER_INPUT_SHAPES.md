# MiniCPM-o 4.5 encoder input shapes under full-duplex video input

Real-shape analysis for the full-duplex data plane (`vllm_omni/model_executor/models/minicpmo_4_5/duplex/`),
derived from the official `StreamingMelProcessorExact` math (`processing_minicpmo.py` in the
checkpoint remote code) and the vLLM-Omni duplex code. Companion to
`benchmark_encoder_saturation.py`.

Cadence: full duplex emits one model unit per second. In the deployed steady
state each append carries one second of audio plus one camera frame (two
frames when an official stacked pair occurs).

## Audio encoder (apm, Whisper-medium)

Config: 24 encoder layers, d_model=1024, 16 heads, encoder_ffn_dim=4096,
80 mel bins, `max_source_positions` (embed_positions cap) = 1500.

### Mel shapes fed to `get_audio_embedding_streaming`

Streaming mel processor configured by `duplex/stage0.py:102-112` with
`mode="exact"`, `chunk_ms=1000`, `first_chunk_ms=1035`, `cnn_redundancy_ms=20`
(sliding window: 30 s trigger, 10 s stride).

With hop=160 at 16 kHz (100 mel frames/s, 10 ms/frame):

| chunk | mel shape | composition |
|---|---|---|
| chunk 0 | `(1, 80, 102)` | 100 core frames + 2 right redundancy (left redundancy clamped at 0) |
| chunk k>0 | `(1, 80, 104)` | 100 core + 2 left redundancy + 2 right redundancy |

Derivation: `chunk_frames = 16000 // 160 = 100`; the emitter expands the core
window by `cnn_redundancy_frames = 20 ms * 100 fps/1000 = 2` frames on each
side; the first emission starts at frame 0 so the left redundancy is clamped.
The 1035 ms first chunk exists so STFT stable-frame count (`(L - 200)/160 + 1`)
covers `core_end + 2` for the first emission.

### Inside the encoder (`minicpmo_4_5_omni_llm.py:4404-4496`)

- conv1/conv2 (stride 2): 102 -> 51 conv frames (chunk 0), 104 -> 52 conv
  frames (later chunks) via `(L - 1) // 2 + 1`.
- `use_extra_context=True` strips redundant context frames
  (`minicpmo_4_5_omni_llm.py:2770-2778`): chunk 0 strips suffix 1
  (`prefix_extra_frames=0`, `suffix_extra_frames=2`), later chunks strip
  prefix 1 + suffix 1 -> **exactly 50 frames enter the 24 layers** in both
  cases (51 - 1 and 52 - 2).
- Encoder KV grows exactly 50 frames/s (what the layers see per second). The
  1500 `embed_positions` cap is reached at the 30 s append
  (past 1500 + current 50 >= cap); `minicpmo_4_5_omni_llm.py:4447-4456`
  auto-resets `audio_past_key_values`, so that append re-encodes from an
  empty cache. The mel side slides its window (30 s trigger / 10 s stride).
  **The 30 s boundary is a latency spike point.**
- `output_hidden_states=True` returns all 25 hidden states; only the final
  layer is consumed (`audio_encoder_layer=-1`) - extra memory/bandwidth
  overhead per chunk.

### Output invariant

Projection (1024 -> 4096) then `audio_avg_pooler` (AvgPool1d stride 5) yields
**10 embeddings per second** (one per 100 ms). This is the
`SAMPLES_PER_AUDIO_TOKEN = 1600` contract (`duplex/policy.py:24-26`); scheduler
token budgets must match it exactly or listen/speak behavior degrades
(`duplex/policy.py:17-23`). Cross-check with the paper: Whisper produces
"50 feature tokens per second" - our net rate is exactly 50 (52 conv frames
minus 2 stripped context frames, uniform across chunks) - consistent.

## Vision encoder (SigLIP + resampler)

Config: 27 layers, hidden 1152, patch 14; resampler emits `query_num=64`
queries. Frames enter through the serving PCM queue at **at most one frame
per 1 s unit** (`duplex/input.py:336-345`); the official protocol additionally
allows a stacked pair (two frames) in one unit - see below.

### What a "slice" is

A slice is one 448-normalized image tile and the unit of SigLIP computation.
Each slice goes through SigLIP then the adaptive resampler (64 fixed queries
cross-attending over the slice's patches after `kv_proj` 1152->4096), so
**every slice costs the same ~1030-patch SigLIP forward and yields exactly 64
embeds -> token block `<image> + 64 + </image>` = 66 tokens**. Encoder cost
scales linearly with the number of slices per second, not with camera
resolution.

### The mechanism, per the MiniCPM-o 4.5 paper

The MiniCPM-o 4.5 technical report (arXiv:2604.27393, "MiniCPM-o 4.5: Towards
Real-Time Full-Duplex Omni-Modal Interaction") pins down the visual encoding:

> "MiniCPM-o 4.5 adopts the LLaVA-UHD image partitioning strategy to encode
> any aspect high-resolution images and improve compression rate with a
> resampler module. We adopt a max resolution of **448x448 for the full-duplex
> streaming mode** and otherwise 2240x2240. Specifically, each image is first
> divided into slices, and each slice is then encoded into 1024 tokens by a
> SigLIP ViT (0.4B) and compressed into 64 tokens by the resampler module.
> This yields a 16x token compression ratio."

And the audio side (confirming the shapes in the section above):

> "A Whisper Medium encoder (0.3B) encodes input audio in a chunk-based
> streaming fashion, producing **50 feature tokens per second**. We then use
> a two-layer MLP projector to conduct a 5x temporal compression, resulting
> in **10 audio tokens per second** for the LLM backbone."

The unified serialization `g_k = [v_k; a_k; o_k]` (visual tokens first, then
audio, then output, per 1 s chunk with explicit boundary tokens and the
Listen-Speak control token) matches the vLLM-Omni unit layout implemented in
`duplex/stage0.py:265-305` (frames before audio inside a unit).

### The two knobs (official code, not paper-specified)

The paper caps full-duplex streaming at 448x448 (one slice per frame). The
checkpoint code and README expose two orthogonal extensions:

| knob | dimension | what it does | cost |
|---|---|---|---|
| `max_slice_nums` | **spatial** | HD slicing: frame becomes 1 source tile (global view) + HD patches (local views at higher effective resolution) | (1 + N) x 66 tokens for that frame |
| `stack_frames` | **temporal** | 1 main frame + N-1 sub-frames composited into one grid image | +1 image block (+66 tokens) per second |

Duplex serving convention (checkpoint README `streaming_prefill` example:
`max_slice_nums=1, # Increase for HD mode (e.g., [2, 1] for stacked frames)`):
HD is **off by default**; when a unit carries a stacked pair, the official
policy is `[2, 1]` - HD only on the current frame. The `[2, 1]` value is a
demo-code convention; the paper does not state it.

### Real shapes

Common camera resolutions with `max_slice_nums=1` (no slicing; single tile via
`find_best_resize` at scale_resolution=448, `ensure_divide` patch 14):

| camera | resized tile W x H | tgt_sizes (grid_w, grid_h) | patches |
|---|---|---|---|
| 960x540 | 602 x 336 | (43, 24) | 1032 |
| 1280x720 | 602 x 336 | (43, 24) | 1032 |
| 640x480 | 518 x 392 | (37, 28) | 1036 |
| 1080x1920 (portrait) | 336 x 602 | (24, 43) | 1032 |

Stacked pair (`max_slice_nums=[2, 1]`): the current frame is HD-sliced
(1 source + 2 patches; official reference 960x540: source tile 602x336 ->
1032 patches, each HD patch 420x476 -> 1020 patches) and the composite is a
single tile -> **4 slices = 4 x 66 = 264 tokens** in that unit
(`duplex/runtime.py:27,50-51`: `(3 + (count - 1)) * 66`). SigLIP runs one
call per frame: batch=3 slices, then batch=1.

## Per-append thinker token budget

这里的 **scheduler slot** 是 vLLM scheduler 为一次 append 预留的 token 位置数，
用于本轮请求的调度和 KV cache 分配。它包含 unit 控制 token、音频和视觉 embedding
对应的 token 位置，以及必要的 terminator 修正；它不是并发会话数，也不表示固定的
显存容量。首个 append 还需为会话上下文和可选的 reference audio 预留额外位置，
因此通常高于稳态 unit。

Exact scheduler slots per append (`duplex/runtime.py:80-89,130-140`:
`units x (2 unit tokens + 10 audio) + vision`, plus corrections):

| append | composition | slots |
|---|---|---|
| first (seq<=1) | session context reserve (48 base + ref audio embeddings) + 11 unit tokens (`<unit>` + 10 audio, no `</unit>` yet) + 66 vision | reserve + 77 |
| steady per-second (seq>1) | 12 base slots (`<unit>` + `</unit>` + 10 audio) + 1 terminator slot (seq>1 exact-chunk correction, `runtime.py:137-138`) + 66 vision | **79** |
| steady, stacked pair | same unit overhead + 264 vision (4 slices) | **277** |

The +1 terminator slot is consumed by stage0's re-injection of the previous
unit's sampled terminator ahead of `</unit>` (`duplex/stage0.py:272-278`); it
is recorded when a segment ends on a terminator token such as `<|listen|>`
(`minicpmo_4_5_omni.py:966-968`). The `final` append adds a further +12 slots
(one extra unit for the turn closure, `runtime.py:139-140`). The stage0
construction that these budgets must match exactly is
`duplex/stage0.py:265-305` (terminator re-injection, `</unit>` close,
`<unit>` open, `<image>`/`<slice>` blocks, audio embeds).

## Mapping to `benchmark_encoder_saturation.py`

- Vision: `--vision-patches 1024` (the script default) is already the
  worst-case real shape (real tiles are 1020-1036 patches). Batch size
  corresponds to the slices arriving in one step: 1 (steady state) to
  3-4 (stacked pair, two calls of batch=3 then batch=1).
- Audio, non-streaming: `--audio-mel-frames 100` (1 s) and `3000` (30 s
  full clip) cover both ends of the turn-mode path.
- Audio, streaming: `--audio-mode streaming` reproduces the real duplex
  load: 1 s chunks through the checkpoint's `StreamingMelProcessorExact`
  (chunk_ms=1000, first_chunk_ms=1035, cnn_redundancy_ms=20) into the
  production `get_audio_embedding_streaming` path with a growing per-session
  audio KV cache and its 30 s reset. Concurrency is modeled as independent
  interleaved sessions (`--stream-count`), because the production encoder
  path is batch-1 per session. Reported categories: first chunk, steady
  chunks, and the reset chunk at the 30 s boundary.

## Limitations

- Both paths run eager in production (encoder calls happen inside the runner
  preprocess hook once per second, with per-chunk shape variation from extra
  context frames and the growing encoder KV), so an eager microbenchmark is
  representative of the duplex encoder path; no CUDA graph is available for
  the growing-cache path today.
