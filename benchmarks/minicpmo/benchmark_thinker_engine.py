# SPDX-License-Identifier: Apache-2.0
"""Measure the MiniCPM-o 4.5 Thinker through the production vLLM-Omni engine.

Unlike ``benchmark_thinker_saturation.py`` (standalone HF Qwen3 microbench,
eager + non-paged KV), this benchmark boots the thinker-only pipeline
(``vllm_omni/deploy/minicpmo_4_5_thinker_only.yaml``) so prefill runs through
the production piecewise torch.compile path and decode through the engine's
full CUDA graphs with paged KV cache — no standalone analogues.

Each measurement submits ``batch_size`` text requests with exact token
lengths and fixed ``generated_tokens`` (``ignore_eos``) and reads the vLLM
native per-request metrics surfaced on ``OmniRequestOutput``:
``vllm_ttft_ms`` (prefill) and ``vllm_itls_ms`` (per-decode-step latency).
Batching happens inside the model runner exactly as in duplex serving.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy

DEFAULT_MODEL = Path("/mnt/shared-storage-user/gpfs2-shared-public/huggingface/zskj-hub/models--OpenBMB--MiniCPM-o-4_5")
DEFAULT_DEPLOY = Path("vllm_omni/deploy/minicpmo_4_5_thinker_only.yaml")
DEFAULT_CONTEXTS = "64,256,4096,16384"
DEFAULT_BATCH = "1,2,4,8,16"

# Same budget one duplex unit spends in decode (26 speak tokens per chunk).
NATIVE_SPEAK_TOKENS = MiniCPMO45DuplexPolicy.DEFAULT_MAX_NEW_SPEAK_TOKENS_PER_CHUNK

SYSTEM = (
    "You are MiniCPM-o, a helpful multimodal assistant that can "
    "understand images, audio and video, and respond in text and speech."
)
PROMPT_HEAD = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
PROMPT_TAIL = "<|im_end|>\n<|im_start|>assistant\n"
FILLER_UNITS = [
    "The quick brown fox jumps over the lazy dog. ",
    "Pack my box with five dozen liquor jugs. ",
    "How vexingly quick daft zebras jump. ",
    "Sphinx of black quartz, judge my vow. ",
]

RequestRow = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--deploy-config", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (applied via CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--batch-sizes", type=str, default=DEFAULT_BATCH)
    parser.add_argument("--context-lengths", type=str, default=DEFAULT_CONTEXTS)
    parser.add_argument("--generated-tokens", type=int, default=NATIVE_SPEAK_TOKENS)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-timeout", type=int, default=1800)
    parser.add_argument("--stage-init-timeout", type=int, default=1500)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--enforce-eager", action="store_true", help="A/B control: disable compile + CUDA graphs")
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Override engine max_num_seqs (e.g. to test batch sizes above the deploy config's default)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Override engine max_num_batched_tokens (e.g. to rule out chunked-prefill scheduling artifacts)",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--engine-log", type=Path, help="Redirected stdout log of this run; parsed for graph evidence")
    args = parser.parse_args()
    if args.generated_tokens <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("generated tokens and repeats must be positive; warmup must be non-negative")
    if args.max_num_seqs is not None and args.max_num_seqs <= 0:
        parser.error("--max-num-seqs must be positive")
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens <= 0:
        parser.error("--max-num-batched-tokens must be positive")
    return args


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _build_prompt_ids(tokenizer, context_length: int, filler_index: int) -> list[int] | None:
    """Chat-template prompt padded with filler tokens to exactly ``context_length`` ids."""
    head_ids = _encode(tokenizer, PROMPT_HEAD)
    tail_ids = _encode(tokenizer, PROMPT_TAIL)
    budget = context_length - len(head_ids) - len(tail_ids)
    if budget < 0:
        return None
    unit = _encode(tokenizer, FILLER_UNITS[filler_index % len(FILLER_UNITS)])
    filler = (unit * (budget // len(unit) + 1))[:budget]
    return head_ids + filler + tail_ids


@dataclass
class EngineThinkerResult:
    batch_size: int
    context_length: int
    generated_tokens: int
    repeats: list[dict[str, Any]] = field(default_factory=list)
    prefill_ms_p50: float | None = None
    prefill_ms_p95: float | None = None
    prefill_ms_p99: float | None = None
    decode_step_ms_p50: float | None = None
    decode_step_ms_p95: float | None = None
    decode_step_ms_p99: float | None = None
    decode_batch_tokens_per_second_p50: float | None = None
    unit_gen_time_ms_p95: float | None = None
    status: str = "ok"
    error: str | None = None


def _aggregate(result: EngineThinkerResult, timed_repeats: list[list[RequestRow]]) -> None:
    ttfts: list[float] = []
    itls: list[float] = []
    unit_gen_ms: list[float] = []
    batch_rates: list[float] = []
    for requests in timed_repeats:
        ttfts.extend(r["ttft_ms"] for r in requests)
        for r in requests:
            itls.extend(r["itls_ms"])
            unit_gen_ms.append(r["stage_gen_time_ms"])
        span_s = max(sum(r["itls_ms"]) for r in requests) / 1000.0
        if span_s > 0:
            batch_rates.append(result.batch_size * (result.generated_tokens - 1) / span_s)
    if ttfts:
        result.prefill_ms_p50 = _percentile(ttfts, 0.50)
        result.prefill_ms_p95 = _percentile(ttfts, 0.95)
        result.prefill_ms_p99 = _percentile(ttfts, 0.99)
    if itls:
        result.decode_step_ms_p50 = _percentile(itls, 0.50)
        result.decode_step_ms_p95 = _percentile(itls, 0.95)
        result.decode_step_ms_p99 = _percentile(itls, 0.99)
    if unit_gen_ms:
        result.unit_gen_time_ms_p95 = _percentile(unit_gen_ms, 0.95)
    if batch_rates:
        result.decode_batch_tokens_per_second_p50 = _percentile(batch_rates, 0.50)


def _collect_outputs(outputs, expected_count: int, generated_tokens: int) -> tuple[list[RequestRow], str | None]:
    """Convert finished OmniRequestOutputs into per-request metric rows."""
    by_index: dict[int, RequestRow] = {}
    for out in outputs:
        if out.error:
            return [], repr(out.error).splitlines()[0]
        try:
            stage_m = out.metrics["stage_metrics"]["0"]
        except (KeyError, TypeError) as exc:
            return [], f"missing stage-0 metrics for {out.request_id}: {exc!r}"
        index = int(str(out.request_id).split("_")[0])
        by_index[index] = {
            "request_index": index,
            "request_id": out.request_id,
            "prompt_tokens": int(stage_m.get("num_tokens_in", 0)),
            "generated_tokens": int(stage_m.get("num_tokens_out", 0)),
            "ttft_ms": float(stage_m.get("vllm_ttft_ms", 0.0)),
            "tpot_ms": float(stage_m.get("vllm_tpot_ms", 0.0)),
            "itls_ms": [float(v) for v in stage_m.get("vllm_itls_ms", [])],
            "stage_gen_time_ms": float(stage_m.get("stage_gen_time_ms", 0.0)),
            "finish_reason": stage_m.get("finish_reason"),
        }
    rows = [by_index[i] for i in range(expected_count) if i in by_index]
    if len(rows) != expected_count:
        missing = [i for i in range(expected_count) if i not in by_index]
        return rows, f"missing {len(missing)}/{expected_count} requests: {missing}"
    for row in rows:
        if row["generated_tokens"] != generated_tokens:
            return rows, (
                f"request {row['request_id']} generated {row['generated_tokens']} tokens, expected {generated_tokens}"
            )
        if len(row["itls_ms"]) != generated_tokens - 1:
            return rows, (
                f"request {row['request_id']} has {len(row['itls_ms'])} ITLs, expected {generated_tokens - 1}"
            )
    return rows, None


def _stage_config_snapshot(omni) -> dict[str, Any]:
    try:
        cfg = omni.stage_configs[0]
        engine_args = cfg.engine_args

        def _field(obj, name, default=None):
            try:
                value = obj[name] if not isinstance(obj, (dict,)) else obj.get(name, default)
                if hasattr(value, "get"):
                    return value.get()
                return default if value is None else value
            except Exception:  # noqa: BLE001
                return default

        model_stage = _field(engine_args, "model_stage", _field(cfg, "model_stage", "llm"))
        return {
            "model_stage": str(model_stage),
            "session_mode": str(_field(cfg, "session_mode", "turn")),
            "enforce_eager": bool(_field(engine_args, "enforce_eager", False)),
            "max_num_seqs": int(_field(engine_args, "max_num_seqs", 0)),
            "max_num_batched_tokens": int(_field(engine_args, "max_num_batched_tokens", 0)),
            "gpu_memory_utilization": float(_field(engine_args, "gpu_memory_utilization", 0.0)),
        }
    except Exception as exc:  # noqa: BLE001 - snapshot is best-effort metadata
        return {"error": repr(exc).splitlines()[0]}


_ENGINE_LOG_PATTERNS = (
    "Capturing CUDA graphs",
    "cudagraph",
    "PIECEWISE",
    "FULL_AND_PIECEWISE",
    "torch.compile",
    "GPU KV cache size",
    "Available KV cache memory",
    "model weights take",
)


def _engine_log_evidence(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    matched: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip()
            if any(pattern in line for pattern in _ENGINE_LOG_PATTERNS):
                key = line[:200]
                if key not in seen:
                    seen.add(key)
                    matched.append(line)
    return matched


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from vllm_omni.entrypoints.omni import Omni

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    batch_sizes = [int(v) for v in args.batch_sizes.split(",") if v.strip()]
    context_lengths = [int(v) for v in args.context_lengths.split(",") if v.strip()]
    if not batch_sizes or not context_lengths:
        raise ValueError("batch-sizes and context-lengths must be non-empty comma-separated ints")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    template_tokens = len(_encode(tokenizer, PROMPT_HEAD)) + len(_encode(tokenizer, PROMPT_TAIL))

    init_kwargs: dict[str, Any] = {
        "model": str(args.model),
        "deploy_config": str(args.deploy_config),
        "trust_remote_code": True,
        "init_timeout": args.init_timeout,
        "stage_init_timeout": args.stage_init_timeout,
        "batch_timeout": args.batch_timeout,
        "output_modalities": ["text"],
        # Required for the vLLM native per-request metrics (vllm_ttft_ms /
        # vllm_itls_ms) that this benchmark reads off OmniRequestOutput.
        "log_stats": True,
    }
    if args.enforce_eager or args.max_num_seqs is not None or args.max_num_batched_tokens is not None:
        overrides: dict[str, Any] = {}
        if args.enforce_eager:
            overrides["enforce_eager"] = True
        if args.max_num_seqs is not None:
            overrides["max_num_seqs"] = args.max_num_seqs
        if args.max_num_batched_tokens is not None:
            overrides["max_num_batched_tokens"] = args.max_num_batched_tokens
        init_kwargs["stage_overrides"] = {"0": overrides}

    init_event = {"event": "initializing_engine", "deploy_config": str(args.deploy_config)}
    print(json.dumps(init_event, sort_keys=True), flush=True)
    omni = Omni(**init_kwargs)
    snapshot = _stage_config_snapshot(omni)
    print(json.dumps({"event": "engine_ready", "stage_config": snapshot}, sort_keys=True), flush=True)

    metadata: dict[str, Any] = {
        "timestamp_unix": time.time(),
        "mode": "engine",
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "model": str(args.model),
        "deploy_config": str(args.deploy_config),
        "enforce_eager": args.enforce_eager,
        "max_num_seqs_override": args.max_num_seqs,
        "max_num_batched_tokens_override": args.max_num_batched_tokens,
        "stage_config": snapshot,
        "num_stages": omni.num_stages,
        "chat_template_tokens": template_tokens,
        "batch_sizes": batch_sizes,
        "context_lengths": context_lengths,
        "generated_tokens": args.generated_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "measurement_scope": (
            "Production vLLM-Omni thinker-only engine: piecewise-compiled prefill + full-CUDA-graph decode + paged KV; "
            "TTFT/ITL from vLLM native per-request metrics (idle engine, one batch at a time)"
        ),
        "prompt_scope": (
            "Text-only chat prompts with exact token counts; the chat template is included in each count "
            "(text-only cannot reproduce the duplex audio-embedding continuation)"
        ),
    }

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.generated_tokens,
        ignore_eos=True,
        detokenize=True,
        seed=args.seed,
    )

    results: list[EngineThinkerResult] = []
    engine_dead = False
    try:
        filler_counter = 0
        for context_length in context_lengths:
            for batch_size in batch_sizes:
                result = EngineThinkerResult(
                    batch_size=batch_size,
                    context_length=context_length,
                    generated_tokens=args.generated_tokens,
                )
                if engine_dead:
                    result.status = "engine_dead"
                    results.append(result)
                    print(json.dumps(asdict(result), sort_keys=True), flush=True)
                    continue
                prompts = [_build_prompt_ids(tokenizer, context_length, filler_counter + i) for i in range(batch_size)]
                filler_counter += batch_size
                if any(p is None for p in prompts):
                    result.status = "skip_ctx_below_template"
                    results.append(result)
                    print(json.dumps(asdict(result), sort_keys=True), flush=True)
                    continue
                for repeat_index in range(args.warmup + args.repeats):
                    is_warmup = repeat_index < args.warmup
                    error: str | None = None
                    rows: list[RequestRow] = []
                    wall_ms: float | None = None
                    try:
                        start = time.perf_counter()
                        outputs = omni.generate(prompts, [sampling_params], use_tqdm=False)
                        wall_ms = (time.perf_counter() - start) * 1000.0
                        rows, error = _collect_outputs(outputs, batch_size, args.generated_tokens)
                    except Exception as exc:  # noqa: BLE001
                        error = repr(exc).splitlines()[0]
                        if "dead" in error.lower():
                            engine_dead = True
                    if error is not None:
                        result.status = "engine_dead" if engine_dead else "error"
                        result.error = error
                        break
                    if is_warmup:
                        print(
                            json.dumps(
                                {"event": "warmup", "ctx": context_length, "bs": batch_size, "wall_ms": wall_ms},
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        continue
                    result.repeats.append({"repeat": repeat_index - args.warmup, "wall_ms": wall_ms, "requests": rows})
                    print(
                        json.dumps(
                            {"event": "run", "ctx": context_length, "bs": batch_size, "wall_ms": wall_ms},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                timed = [r["requests"] for r in result.repeats if r["requests"]]
                if timed:
                    _aggregate(result, timed)
                results.append(result)
                print(json.dumps(asdict(result), sort_keys=True), flush=True)
    finally:
        metadata["engine_log_evidence"] = _engine_log_evidence(args.engine_log)
        payload = {"metadata": metadata, "results": [asdict(result) for result in results]}
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with args.output_json.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
        omni.close()
        print(json.dumps({"event": "done", "output_json": str(args.output_json)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
