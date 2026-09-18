# SPDX-License-Identifier: Apache-2.0
"""Measure the MiniCPM-o 4.5 Talker through the production vLLM-Omni engine.

Boots the **Talker-only** pipeline
(``vllm_omni/deploy/minicpmo_4_5_talker_only.yaml``): the Thinker stage is
bypassed entirely and a fake hidden-state handoff whose length is
controlled by the benchmark is fed straight into the Talker. The Talker
runs with FULL-decode CUDA graphs + paged KV (unless ``--enforce-eager``
is set). Reads the vLLM native per-request metrics off
``OmniRequestOutput`` and reports the Talker scaling curve (stage-0 TTFT
= Talker prefill, stage-0 ITLs = Talker decode steps).

The synthesised handoff latent is random Gaussian noise (``seed=0`` for
reproducibility), so codec token quality is meaningless — only latency /
throughput numbers are valid. Sampling terminates on codec EOS
(``6561``).
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
_SPEC = importlib.util.spec_from_file_location(
    "_thinker_engine_bench", _HERE / "benchmark_thinker_engine.py"
)
_thinker = importlib.util.module_from_spec(_SPEC)
sys.modules["_thinker_engine_bench"] = _thinker
_SPEC.loader.exec_module(_thinker)

DEFAULT_DEPLOY = Path("vllm_omni/deploy/minicpmo_4_5_talker_only.yaml")
DEFAULT_BATCH = "1,2,4,8,16,32"
DEFAULT_TALKER_MIN_TOKENS = "0,1024"
# Same 8-point rolling-KV sweep as the eager standalone (§5.4 KV8). Talker's
# max_model_len is 4096; the 2-token condition suffix leaves room for all
# points plus a small safety margin.
DEFAULT_KV_LENGTHS = "0,512,1024,1536,2048,2560,3072,3584"
# Pad each KV-length target by 2 dummies to match the real pipeline's
# ``condition_suffix_length`` semantics.
CONDITION_SUFFIX_LENGTH = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=_thinker.DEFAULT_MODEL)
    parser.add_argument("--deploy-config", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-sizes", type=str, default=DEFAULT_BATCH)
    parser.add_argument("--talker-min-tokens", type=str, default=DEFAULT_TALKER_MIN_TOKENS)
    parser.add_argument("--talker-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--kv-lengths",
        type=str,
        default=DEFAULT_KV_LENGTHS,
        help=(
            "comma list of rolling-KV lengths (same 8-point sweep as "
            "benchmark_talker_saturation.py). 0 = empty handoff, N = N+2 "
            "conditioning tokens (the +2 matches the real pipeline's "
            "condition_suffix_length)."
        ),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-timeout", type=int, default=1800)
    parser.add_argument("--stage-init-timeout", type=int, default=1500)
    parser.add_argument("--batch-timeout", type=int, default=5)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help=(
            "Override engine max_num_seqs (e.g. to test batch sizes above the "
            "deploy config's default). Mirrors benchmark_thinker_engine.py."
        ),
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help=(
            "Override engine max_num_batched_tokens. Useful to rule out "
            "chunked-prefill scheduling artifacts at large batch sizes. "
            "Mirrors benchmark_thinker_engine.py."
        ),
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--engine-log", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("repeats must be positive; warmup non-negative")
    if args.max_num_seqs is not None and args.max_num_seqs <= 0:
        parser.error("--max-num-seqs must be positive")
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens <= 0:
        parser.error("--max-num-batched-tokens must be positive")
    return args


@dataclass
class TalkerOnlyResult:
    batch_size: int
    talker_min_tokens: int
    kv_length: int
    repeats: list[dict[str, Any]] = field(default_factory=list)
    talker_prefill_ms_p50: float | None = None
    talker_prefill_ms_p95: float | None = None
    talker_decode_step_ms_p50: float | None = None
    talker_decode_step_ms_p95: float | None = None
    talker_decode_step_ms_p99: float | None = None
    talker_tokens_out_p50: float | None = None
    talker_decode_batch_tokens_per_second_p50: float | None = None
    status: str = "ok"
    error: str | None = None


def _stage_row(out: Any, stage_index: int) -> dict[str, Any] | None:
    # Newer Omni exposes per-stage metrics under ``out.metrics["stage_metrics"]``
    # as a dict keyed by stage id (string). Returns ``None`` when the entry is
    # missing (matches the ``_stage_row is None`` contract in the loop).
    metrics = getattr(out, "metrics", None)
    if not isinstance(metrics, dict):
        return None
    stage_metrics = metrics.get("stage_metrics")
    if not isinstance(stage_metrics, dict):
        return None
    return stage_metrics.get(str(stage_index))


def _aggregate(result: TalkerOnlyResult, timed: list[list[dict[str, Any]]]) -> None:
    # ``stage_metrics["0"]`` carries vLLM-Omni's published per-stage stats:
    # ``vllm_ttft_ms`` (Talker prefill), ``vllm_itls_ms`` (list of decode
    # ITLs), ``stage_gen_time_ms`` (total stage wall time), ``num_tokens_in``
    # / ``num_tokens_out`` (codec token counts).
    ttfts: list[float] = []
    itls: list[float] = []
    tokens_out: list[int] = []
    batch_rates: list[float] = []
    for repeat in timed:
        if not repeat:
            continue
        span_s = sum(r["talker"].get("stage_gen_time_ms", 0.0) for r in repeat) / 1000.0
        total_tokens = sum(r["talker"].get("num_tokens_out", 0) for r in repeat)
        if span_s > 0:
            batch_rates.append(total_tokens / span_s)
        for r in repeat:
            talker = r.get("talker", {})
            if "vllm_ttft_ms" in talker:
                ttfts.append(float(talker["vllm_ttft_ms"]))
            if "vllm_itls_ms" in talker:
                itls.extend(float(v) for v in talker["vllm_itls_ms"])
            if "num_tokens_out" in talker:
                tokens_out.append(int(talker["num_tokens_out"]))
    if ttfts:
        result.talker_prefill_ms_p50 = _thinker._percentile(ttfts, 0.50)
        result.talker_prefill_ms_p95 = _thinker._percentile(ttfts, 0.95)
    if itls:
        result.talker_decode_step_ms_p50 = _thinker._percentile(itls, 0.50)
        result.talker_decode_step_ms_p95 = _thinker._percentile(itls, 0.95)
        result.talker_decode_step_ms_p99 = _thinker._percentile(itls, 0.99)
    if tokens_out:
        result.talker_tokens_out_p50 = _thinker._percentile([float(t) for t in tokens_out], 0.50)
    if batch_rates:
        result.talker_decode_batch_tokens_per_second_p50 = _thinker._percentile(batch_rates, 0.50)


def main() -> None:
    args = parse_args()
    # Respect an externally-set CUDA_VISIBLE_DEVICES (e.g. ``CUDA_VISIBLE_DEVICES=4
    # python ...``); only fall back to ``--device`` when no env var is present.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

    import torch
    import vllm
    from vllm import SamplingParams

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniTokensPrompt

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    batch_sizes = [int(v) for v in args.batch_sizes.split(",") if v.strip()]
    talker_min_tokens_list = [int(v) for v in args.talker_min_tokens.split(",") if v.strip()]
    kv_lengths = [int(v) for v in args.kv_lengths.split(",") if v.strip()]

    def build_request(kv_length: int) -> OmniTokensPrompt:
        # Bypass the stage-input processor entirely (the talker-only pipeline
        # has no upstream source_outputs): stamp a synthesised Thinker
        # handoff straight onto ``model_intermediate_buffer`` so the runner
        # resolves ``tts_token_ids``/``tts_hidden_states`` on the first
        # ``Talker.preprocess`` call.
        #
        # Shape contract (mirrors ``llm2tts`` + ``projector_semantic 4096->768``):
        #   * tts_token_ids: [N] long
        #   * tts_hidden_states: [N, 4096] float32  (LLM hidden_size; the
        #     Talker's projector maps it down to its 768-dim embedding).
        #   * meta.next_stage_prompt_len: N  (Talker scheduler reserves N KV
        #     slots; the real path uses max(|ids|, |hidden|) + suffix).
        condition_length = kv_length + CONDITION_SUFFIX_LENGTH
        gen = torch.Generator(device="cpu").manual_seed(0)
        handoff_ids = [0] * condition_length
        handoff_hidden = torch.randn(
            (condition_length, 4096), generator=gen, dtype=torch.float32
        ).tolist()
        model_intermediate_buffer = {
            "global_request_id": ["fake_talker_only"],
            "ids": {
                "prompt": [0] * condition_length,
                "tts": list(handoff_ids),
            },
            "hidden_states": {
                "tts": list(handoff_hidden),
            },
            "meta": {
                "next_stage_prompt_len": condition_length,
            },
        }
        return OmniTokensPrompt(
            prompt_token_ids=[0] * condition_length,
            model_intermediate_buffer=model_intermediate_buffer,
        )

    init_kwargs: dict[str, Any] = {
        "model": str(args.model),
        "deploy_config": str(args.deploy_config),
        "trust_remote_code": True,
        "init_timeout": args.init_timeout,
        "stage_init_timeout": args.stage_init_timeout,
        "batch_timeout": args.batch_timeout,
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
        "mode": "engine_talker_only",
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "model": str(args.model),
        "deploy_config": str(args.deploy_config),
        "enforce_eager": args.enforce_eager,
        "max_num_seqs_override": args.max_num_seqs,
        "max_num_batched_tokens_override": args.max_num_batched_tokens,
        "stage_config": stages_snapshot,
        "num_stages": omni.num_stages,
        "batch_sizes": batch_sizes,
        "talker_min_tokens": talker_min_tokens_list,
        "talker_max_tokens": args.talker_max_tokens,
        "kv_lengths": kv_lengths,
        "condition_suffix_length": CONDITION_SUFFIX_LENGTH,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "measurement_scope": (
            "Production vLLM-Omni Talker-only pipeline: stage 0 = Talker with "
            "piecewise prefill + FULL-decode CUDA graph + paged KV. "
            "Synthesised Thinker hidden-state handoff (random Gaussian, seed=0); "
            "codec token quality is meaningless."
        ),
        "talker_length_control": (
            "Talker codec output ends at natural codec EOS (variable, min_tokens=0) "
            "or is pushed to ~min_tokens tokens via Sampler min_tokens (offline cap 2048)"
        ),
    }

    def build_params(min_tokens: int) -> list[SamplingParams]:
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
        return [SamplingParams(**talker_kwargs)]

    results: list[TalkerOnlyResult] = []
    engine_dead = False
    try:
        for min_tokens in talker_min_tokens_list:
            for kv_length in kv_lengths:
                for batch_size in batch_sizes:
                    result = TalkerOnlyResult(
                        batch_size=batch_size,
                        talker_min_tokens=min_tokens,
                        kv_length=kv_length,
                    )
                    if engine_dead:
                        result.status = "engine_dead"
                        results.append(result)
                        print(json.dumps(asdict(result), sort_keys=True), flush=True)
                        continue
                    prompts = [build_request(kv_length) for _ in range(batch_size)]
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
                                talker_row = _stage_row(out, 0)
                                if talker_row is None:
                                    error = f"missing stage-0 metrics for {out.request_id}"
                                    break
                                index = int(str(out.request_id).split("_")[0])
                                by_index[index] = {
                                    "request_index": index,
                                    "request_id": out.request_id,
                                    "talker": {
                                        "num_tokens_in": int(talker_row.get("num_tokens_in", 0)),
                                        "num_tokens_out": int(talker_row.get("num_tokens_out", 0)),
                                        "stage_gen_time_ms": float(talker_row.get("stage_gen_time_ms", 0.0)),
                                        "vllm_ttft_ms": float(talker_row.get("vllm_ttft_ms", 0.0)),
                                        "vllm_tpot_ms": float(talker_row.get("vllm_tpot_ms", 0.0)),
                                        "vllm_itls_ms": [float(v) for v in talker_row.get("vllm_itls_ms", [])],
                                        "finish_reason": talker_row.get("finish_reason"),
                                    },
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
                                    {
                                        "event": "warmup",
                                        "min_tokens": min_tokens,
                                        "kv_length": kv_length,
                                        "bs": batch_size,
                                        "wall_ms": wall_ms,
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                            continue
                        result.repeats.append(
                            {"repeat": repeat_index - args.warmup, "wall_ms": wall_ms, "requests": rows}
                        )
                        print(
                            json.dumps(
                                {
                                    "event": "run",
                                    "min_tokens": min_tokens,
                                    "kv_length": kv_length,
                                    "bs": batch_size,
                                    "wall_ms": wall_ms,
                                },
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