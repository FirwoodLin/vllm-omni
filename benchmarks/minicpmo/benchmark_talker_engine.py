# SPDX-License-Identifier: Apache-2.0
"""Measure the MiniCPM-o 4.5 Talker through the production vLLM-Omni engine.

Boots the thinker+talker pipeline
(``vllm_omni/deploy/minicpmo_4_5_thinker_talker.yaml``): thinker prefill +
FULL-decode CUDA graphs + paged KV on stage 0, Talker as the final stage on
stage 1 (``llm2tts`` latent handoff, codec-token output). Reads the vLLM
native per-request metrics of BOTH stages off ``OmniRequestOutput`` and
reports the Talker scaling curve (stage-1 TTFT = Talker prefill, stage-1
ITLs = Talker decode steps).

Talker output length is driven by codec EOS, not ``max_tokens``: with
``min_tokens=0`` each request ends at natural codec EOS (variable, ~50-200
codec tokens); with ``min_tokens=N`` the Sampler masks EOS for N tokens so
output reaches ~N (bounded by the offline cap of 2048).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("_thinker_engine_bench", _HERE / "benchmark_thinker_engine.py")
_thinker = importlib.util.module_from_spec(_SPEC)
sys.modules["_thinker_engine_bench"] = _thinker
_SPEC.loader.exec_module(_thinker)

DEFAULT_DEPLOY = Path("vllm_omni/deploy/minicpmo_4_5_thinker_talker.yaml")
DEFAULT_BATCH = "1,2,4,8,16,32"
DEFAULT_TALKER_MIN_TOKENS = "0,1024"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=_thinker.DEFAULT_MODEL)
    parser.add_argument("--deploy-config", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-sizes", type=str, default=DEFAULT_BATCH)
    parser.add_argument(
        "--talker-min-tokens",
        type=str,
        default=DEFAULT_TALKER_MIN_TOKENS,
        help="comma list; 0 = natural codec EOS, N = mask EOS until N tokens",
    )
    parser.add_argument("--talker-max-tokens", type=int, default=4096)
    parser.add_argument("--thinker-output-tokens", type=int, default=32)
    parser.add_argument("--thinker-context-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-timeout", type=int, default=1800)
    parser.add_argument("--stage-init-timeout", type=int, default=1500)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--enforce-eager", action="store_true", help="A/B control: disable compile + CUDA graphs")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--engine-log", type=Path)
    args = parser.parse_args()
    if args.thinker_output_tokens <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("thinker output tokens and repeats must be positive; warmup non-negative")
    return args


@dataclass
class TalkerEngineResult:
    batch_size: int
    talker_min_tokens: int
    thinker_output_tokens: int
    repeats: list[dict[str, Any]] = field(default_factory=list)
    talker_prefill_ms_p50: float | None = None
    talker_prefill_ms_p95: float | None = None
    talker_decode_step_ms_p50: float | None = None
    talker_decode_step_ms_p95: float | None = None
    talker_decode_step_ms_p99: float | None = None
    talker_decode_batch_tokens_per_second_p50: float | None = None
    talker_tokens_out_p50: float | None = None
    thinker_prefill_ms_p50: float | None = None
    status: str = "ok"
    error: str | None = None


def _stage_row(out: Any, stage_id: int) -> dict[str, Any] | None:
    stage_m = out.metrics.get("stage_metrics", {}).get(str(stage_id))
    if stage_m is None:
        return None
    return {
        "prompt_tokens": int(stage_m.get("num_tokens_in", 0)),
        "generated_tokens": int(stage_m.get("num_tokens_out", 0)),
        "ttft_ms": float(stage_m.get("vllm_ttft_ms", 0.0)),
        "tpot_ms": float(stage_m.get("vllm_tpot_ms", 0.0)),
        "itls_ms": [float(v) for v in stage_m.get("vllm_itls_ms", [])],
        "stage_gen_time_ms": float(stage_m.get("stage_gen_time_ms", 0.0)),
        "finish_reason": stage_m.get("finish_reason"),
    }


def _aggregate(result: TalkerEngineResult, timed_repeats: list[list[dict[str, Any]]]) -> None:
    talker_ttfts: list[float] = []
    talker_itls: list[float] = []
    thinker_ttfts: list[float] = []
    tokens_out: list[int] = []
    batch_rates: list[float] = []
    for requests in timed_repeats:
        spans: list[float] = []
        for r in requests:
            talker = r["talker"]
            talker_ttfts.append(talker["ttft_ms"])
            talker_itls.extend(talker["itls_ms"])
            tokens_out.append(talker["generated_tokens"])
            thinker_ttfts.append(r["thinker"]["ttft_ms"])
            spans.append(sum(talker["itls_ms"]))
        span_s = max(spans) / 1000.0
        total_tokens = sum(r["talker"]["generated_tokens"] for r in requests)
        if span_s > 0:
            batch_rates.append(total_tokens / span_s)
    if talker_ttfts:
        result.talker_prefill_ms_p50 = _thinker._percentile(talker_ttfts, 0.50)
        result.talker_prefill_ms_p95 = _thinker._percentile(talker_ttfts, 0.95)
    if talker_itls:
        result.talker_decode_step_ms_p50 = _thinker._percentile(talker_itls, 0.50)
        result.talker_decode_step_ms_p95 = _thinker._percentile(talker_itls, 0.95)
        result.talker_decode_step_ms_p99 = _thinker._percentile(talker_itls, 0.99)
    if tokens_out:
        result.talker_tokens_out_p50 = _thinker._percentile([float(t) for t in tokens_out], 0.50)
    if thinker_ttfts:
        result.thinker_prefill_ms_p50 = _thinker._percentile(thinker_ttfts, 0.50)
    if batch_rates:
        result.talker_decode_batch_tokens_per_second_p50 = _thinker._percentile(batch_rates, 0.50)


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
    talker_min_tokens_list = [int(v) for v in args.talker_min_tokens.split(",") if v.strip()]

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    tts_tail = "<|im_end|>\n<|im_start|>assistant\n<|tts_bos|>"
    template_tokens = len(_thinker._encode(tokenizer, _thinker.PROMPT_HEAD))
    template_tokens += len(_thinker._encode(tokenizer, tts_tail))

    def build_prompt_ids(context_length: int, filler_index: int) -> list[int] | None:
        head_ids = _thinker._encode(tokenizer, _thinker.PROMPT_HEAD)
        tail_ids = _thinker._encode(tokenizer, tts_tail)
        budget = context_length - len(head_ids) - len(tail_ids)
        if budget < 0:
            return None
        unit = _thinker._encode(tokenizer, _thinker.FILLER_UNITS[filler_index % len(_thinker.FILLER_UNITS)])
        filler = (unit * (budget // len(unit) + 1))[:budget]
        return head_ids + filler + tail_ids

    init_kwargs: dict[str, Any] = {
        "model": str(args.model),
        "deploy_config": str(args.deploy_config),
        "trust_remote_code": True,
        "init_timeout": args.init_timeout,
        "stage_init_timeout": args.stage_init_timeout,
        "batch_timeout": args.batch_timeout,
        "log_stats": True,
    }
    if args.enforce_eager:
        init_kwargs["stage_overrides"] = {"0": {"enforce_eager": True}, "1": {"enforce_eager": True}}

    init_event = {"event": "initializing_engine", "deploy_config": str(args.deploy_config)}
    print(json.dumps(init_event, sort_keys=True), flush=True)
    omni = Omni(**init_kwargs)
    snapshot = _thinker._stage_config_snapshot(omni)
    stages_snapshot = [
        _thinker._stage_config_snapshot(type("O", (), {"stage_configs": [cfg]})) for cfg in omni.stage_configs
    ]
    ready_event = {
        "event": "engine_ready",
        "num_stages": omni.num_stages,
        "stage0": snapshot,
        "stages": stages_snapshot,
    }
    print(json.dumps(ready_event, sort_keys=True), flush=True)

    metadata: dict[str, Any] = {
        "timestamp_unix": time.time(),
        "mode": "engine_talker",
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "model": str(args.model),
        "deploy_config": str(args.deploy_config),
        "enforce_eager": args.enforce_eager,
        "stage_config": stages_snapshot,
        "num_stages": omni.num_stages,
        "chat_template_tokens": template_tokens,
        "batch_sizes": batch_sizes,
        "talker_min_tokens": talker_min_tokens_list,
        "talker_max_tokens": args.talker_max_tokens,
        "thinker_output_tokens": args.thinker_output_tokens,
        "thinker_context_tokens": args.thinker_context_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "measurement_scope": (
            "Production vLLM-Omni thinker+talker engine (code2wav dropped): stage-1 Talker = piecewise prefill + "
            "FULL-decode CUDA graph + paged KV; TTFT/ITL from vLLM native per-request metrics"
        ),
        "talker_length_control": (
            "Talker codec output ends at natural codec EOS (variable, min_tokens=0) or is pushed to ~min_tokens "
            "tokens via Sampler min_tokens (offline cap 2048)"
        ),
    }

    def build_params(min_tokens: int) -> list[SamplingParams]:
        thinker_sp = SamplingParams(
            temperature=0.0,
            max_tokens=args.thinker_output_tokens,
            ignore_eos=True,
            detokenize=True,
            seed=args.seed,
        )
        talker_kwargs: dict[str, Any] = {
            "temperature": 0.8,
            "top_p": 0.85,
            "top_k": 25,
            "repetition_penalty": 1.05,
            "max_tokens": args.talker_max_tokens,
            "detokenize": False,
            "seed": args.seed,
        }
        if min_tokens > 0:
            talker_kwargs["min_tokens"] = min_tokens
        return [thinker_sp, SamplingParams(**talker_kwargs)]

    results: list[TalkerEngineResult] = []
    engine_dead = False
    try:
        filler_counter = 0
        for min_tokens in talker_min_tokens_list:
            for batch_size in batch_sizes:
                result = TalkerEngineResult(
                    batch_size=batch_size,
                    talker_min_tokens=min_tokens,
                    thinker_output_tokens=args.thinker_output_tokens,
                )
                if engine_dead:
                    result.status = "engine_dead"
                    results.append(result)
                    print(json.dumps(asdict(result), sort_keys=True), flush=True)
                    continue
                prompts = [build_prompt_ids(args.thinker_context_tokens, filler_counter + i) for i in range(batch_size)]
                filler_counter += batch_size
                if any(p is None for p in prompts):
                    result.status = "skip_ctx_below_template"
                    results.append(result)
                    print(json.dumps(asdict(result), sort_keys=True), flush=True)
                    continue
                params = build_params(min_tokens)
                for repeat_index in range(args.warmup + args.repeats):
                    is_warmup = repeat_index < args.warmup
                    error: str | None = None
                    rows: list[dict[str, Any]] = []
                    wall_ms: float | None = None
                    try:
                        start = time.perf_counter()
                        outputs = omni.generate(prompts, params, use_tqdm=False)
                        wall_ms = (time.perf_counter() - start) * 1000.0
                        by_index: dict[int, dict[str, Any]] = {}
                        for out in outputs:
                            if out.error:
                                error = repr(out.error).splitlines()[0]
                                break
                            thinker_row = _stage_row(out, 0)
                            talker_row = _stage_row(out, 1)
                            if talker_row is None:
                                error = f"missing stage-1 metrics for {out.request_id}"
                                break
                            index = int(str(out.request_id).split("_")[0])
                            by_index[index] = {
                                "request_index": index,
                                "request_id": out.request_id,
                                "thinker": thinker_row or {},
                                "talker": talker_row,
                            }
                        if error is None and len(by_index) != batch_size:
                            missing = [i for i in range(batch_size) if i not in by_index]
                            error = f"missing {len(missing)}/{batch_size} requests: {missing}"
                        rows = [by_index[i] for i in sorted(by_index)]
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
                                {"event": "warmup", "min_tokens": min_tokens, "bs": batch_size, "wall_ms": wall_ms},
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        continue
                    result.repeats.append({"repeat": repeat_index - args.warmup, "wall_ms": wall_ms, "requests": rows})
                    print(
                        json.dumps(
                            {"event": "run", "min_tokens": min_tokens, "bs": batch_size, "wall_ms": wall_ms},
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
        metadata["engine_log_evidence"] = _thinker._engine_log_evidence(args.engine_log)
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
