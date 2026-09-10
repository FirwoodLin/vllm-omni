# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Generate a manifest of paired full-duplex cases from HumDial metadata.

HumDial test WAVs contain one or more user speech segments in a single file.
The full-duplex runner consumes two WAVs (an initial turn and an interrupting
turn), so this tool extracts the first two annotated segments from each item
with at least two segments.  The source metadata is retained in each manifest
row for auditability; no audio or timing is synthesized.

HumDial does not provide assistant answer references.  ``--labels`` can supply
optional ``expected_text_contains`` values keyed by generated ``case_id`` or
source ``relative_path``.  Rows without labels remain valid interaction cases
but are reported as task-correctness ``unknown`` by ``humdial_e2e.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from benchmarks.minicpmo.humdial_arrival_rate import (
        DEFAULT_DATASET_ROOT,
        DEFAULT_SEED,
        HumDialCase,
        discover_humdial_cases,
    )
except ModuleNotFoundError:  # direct ``python benchmarks/minicpmo/...``
    from humdial_arrival_rate import (  # type: ignore[no-redef]
        DEFAULT_DATASET_ROOT,
        DEFAULT_SEED,
        HumDialCase,
        discover_humdial_cases,
    )

from vllm_omni.experimental.fullduplex.client import (
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    read_pcm16_wav,
    write_pcm16_wav,
)

MANIFEST_SCHEMA_VERSION = 1
DEFAULT_PER_SCENARIO = 100
DEFAULT_PLAYBACK_ANCHOR_MS = 5_000


@dataclass(frozen=True)
class SpeechSegment:
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class HumDialE2ECandidate:
    case: HumDialCase
    metadata_path: Path
    segments: tuple[SpeechSegment, ...]

    @property
    def case_id(self) -> str:
        return f"{self.case.language}/{self.case.scenario}/{self.case.path.stem}"


def load_speech_segments(metadata_path: Path) -> tuple[SpeechSegment, ...]:
    """Load and validate the time-aligned HumDial speech segments."""
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw_segments = payload.get("speech_segments") if isinstance(payload, dict) else None
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(f"metadata must contain a non-empty speech_segments list: {metadata_path}")
    segments: list[SpeechSegment] = []
    previous_start_s = 0.0
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            raise ValueError(f"speech_segments[{index}] must be an object: {metadata_path}")
        start_s = raw_segment.get("xmin")
        end_s = raw_segment.get("xmax")
        text = raw_segment.get("text", "")
        if (
            isinstance(start_s, bool)
            or not isinstance(start_s, int | float)
            or not math.isfinite(float(start_s))
            or isinstance(end_s, bool)
            or not isinstance(end_s, int | float)
            or not math.isfinite(float(end_s))
            or float(start_s) < 0
            or float(end_s) <= float(start_s)
        ):
            raise ValueError(f"speech_segments[{index}] has invalid xmin/xmax: {metadata_path}")
        # Overlapping segments are intentional in HumDial's
        # ``others_talk_to_user_after`` cases.  We only require starts to be
        # ordered so the first two segments have a deterministic turn role.
        if segments and float(start_s) < previous_start_s:
            raise ValueError(f"speech_segments must be ordered by xmin: {metadata_path}")
        if not isinstance(text, str):
            raise ValueError(f"speech_segments[{index}].text must be a string: {metadata_path}")
        segments.append(SpeechSegment(float(start_s), float(end_s), text.strip()))
        previous_start_s = float(start_s)
    return tuple(segments)


def discover_humdial_e2e_candidates(
    dataset_root: Path,
    *,
    scenarios: set[str] | None = None,
) -> tuple[list[HumDialE2ECandidate], dict[str, int]]:
    """Discover paired items and count source rows skipped for lacking turns."""
    candidates: list[HumDialE2ECandidate] = []
    skipped = {"single_segment": 0, "filtered_scenario": 0}
    for case in discover_humdial_cases(dataset_root):
        if scenarios is not None and case.scenario not in scenarios:
            skipped["filtered_scenario"] += 1
            continue
        metadata_path = case.path.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"HumDial metadata does not exist: {metadata_path}")
        segments = load_speech_segments(metadata_path)
        if len(segments) < 2:
            skipped["single_segment"] += 1
            continue
        candidates.append(HumDialE2ECandidate(case, metadata_path, segments))
    if not candidates:
        requested = ", ".join(sorted(scenarios)) if scenarios else "all scenarios"
        raise ValueError(f"No HumDial items with at least two segments for {requested}")
    return candidates, skipped


def _sample_language_balanced(
    candidates: list[HumDialE2ECandidate],
    count: int,
    *,
    seed: int,
) -> list[HumDialE2ECandidate]:
    if count < 1:
        raise ValueError("per-scenario count must be positive")
    if count > len(candidates):
        raise ValueError(f"requested {count} cases but only {len(candidates)} are available")
    groups: dict[str, list[HumDialE2ECandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.case.language].append(candidate)
    languages = sorted(groups)
    base, remainder = divmod(count, len(languages))
    rng = random.Random(seed)
    selected: list[HumDialE2ECandidate] = []
    for index, language in enumerate(languages):
        group = sorted(groups[language], key=lambda item: item.case.relative_path)
        rng.shuffle(group)
        target = base + (index < remainder)
        if target > len(group):
            raise ValueError(
                f"language-balanced sample needs {target} {language} cases, "
                f"but only {len(group)} are available"
            )
        selected.extend(group[:target])
    rng.shuffle(selected)
    return selected


def sample_e2e_candidates(
    candidates: list[HumDialE2ECandidate],
    *,
    per_scenario: int,
    seed: int,
    scenarios: set[str] | None = None,
) -> list[HumDialE2ECandidate]:
    """Select a deterministic, language-balanced sample per scenario."""
    grouped: dict[str, list[HumDialE2ECandidate]] = defaultdict(list)
    for candidate in candidates:
        if scenarios is None or candidate.case.scenario in scenarios:
            grouped[candidate.case.scenario].append(candidate)
    if not grouped:
        raise ValueError("no scenarios remain after filtering")
    selected: list[HumDialE2ECandidate] = []
    for index, scenario in enumerate(sorted(grouped)):
        selected.extend(
            _sample_language_balanced(
                grouped[scenario],
                per_scenario,
                seed=seed + (index + 1) * 1_000_003,
            )
        )
    return selected


def _load_labels(path: Path | None) -> dict[str, tuple[str, ...]]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("labels file must be a JSON object keyed by case_id or relative_path")
    labels: dict[str, tuple[str, ...]] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError("labels keys must be non-empty strings")
        if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError(f"labels[{key!r}] must be a list of non-empty strings")
        labels[key] = tuple(item.strip() for item in value)
    return labels


def _clip_pcm16(source_pcm: bytes, *, start_s: float, end_s: float) -> bytes:
    start_frame = max(0, math.floor(start_s * PCM16_SAMPLE_RATE))
    end_frame = min(len(source_pcm) // PCM16_BYTES_PER_SAMPLE, math.ceil(end_s * PCM16_SAMPLE_RATE))
    if end_frame <= start_frame:
        raise ValueError(f"empty audio clip for [{start_s}, {end_s}) seconds")
    start_byte = start_frame * PCM16_BYTES_PER_SAMPLE
    end_byte = end_frame * PCM16_BYTES_PER_SAMPLE
    return source_pcm[start_byte:end_byte]


def _clip_name(candidate: HumDialE2ECandidate) -> str:
    return candidate.case_id.replace("/", "__")


def generate_manifest(
    dataset_root: Path,
    output_dir: Path,
    *,
    per_scenario: int = DEFAULT_PER_SCENARIO,
    seed: int = DEFAULT_SEED,
    labels_path: Path | None = None,
    scenarios: set[str] | None = None,
    playback_anchor_ms: int = DEFAULT_PLAYBACK_ANCHOR_MS,
) -> tuple[Path, dict[str, object]]:
    """Generate extracted WAVs and a v1 manifest, refusing partial overwrite."""
    dataset_root = dataset_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix generated artifacts in non-empty directory: {output_dir}")
    if playback_anchor_ms < 0:
        raise ValueError("playback_anchor_ms must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir()
    labels = _load_labels(labels_path)
    candidates, skipped = discover_humdial_e2e_candidates(dataset_root, scenarios=scenarios)
    selected = sample_e2e_candidates(
        candidates,
        per_scenario=per_scenario,
        seed=seed,
        scenarios=scenarios,
    )
    manifest_cases: list[dict[str, object]] = []
    for candidate in selected:
        first, interrupt = candidate.segments[:2]
        source_pcm = read_pcm16_wav(candidate.case.path)
        initial_pcm = _clip_pcm16(source_pcm, start_s=0.0, end_s=first.end_s)
        interrupt_pcm = _clip_pcm16(source_pcm, start_s=interrupt.start_s, end_s=interrupt.end_s)
        name = _clip_name(candidate)
        initial_path = audio_dir / f"{name}.initial.wav"
        interrupt_path = audio_dir / f"{name}.interrupt.wav"
        write_pcm16_wav(initial_path, initial_pcm, sample_rate_hz=PCM16_SAMPLE_RATE)
        write_pcm16_wav(interrupt_path, interrupt_pcm, sample_rate_hz=PCM16_SAMPLE_RATE)
        expected = labels.get(candidate.case_id, labels.get(candidate.case.relative_path, ()))
        gap_ms = max(0, round((interrupt.start_s - first.end_s) * 1000))
        manifest_cases.append(
            {
                "case_id": candidate.case_id,
                "initial_audio": initial_path.relative_to(output_dir).as_posix(),
                "interrupt_audio": interrupt_path.relative_to(output_dir).as_posix(),
                "external_interrupt_ms": gap_ms,
                "playback_anchor_ms": playback_anchor_ms,
                "operation": interrupt.text,
                "expected_text_contains": list(expected),
                "session_window_s": 60.0,
                "instructions": (
                    "Answer the initial user turn naturally. When the user interrupts, "
                    "stop stale output and answer the new turn while preserving useful context."
                ),
                "dataset_relative_path": candidate.case.relative_path,
                "metadata_relative_path": candidate.metadata_path.relative_to(dataset_root).as_posix(),
                "language": candidate.case.language,
                "scenario": candidate.case.scenario,
                "initial_text": first.text,
                "interrupt_text": interrupt.text,
                "initial_source_end_ms": round(first.end_s * 1000, 3),
                "interrupt_source_start_ms": round(interrupt.start_s * 1000, 3),
                "interrupt_source_end_ms": round(interrupt.end_s * 1000, 3),
            }
        )
    scenario_counts: dict[str, int] = defaultdict(int)
    for case in manifest_cases:
        scenario_counts[str(case["scenario"])] += 1
    payload: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "seed": seed,
        "generator": "generate_humdial_e2e_manifest.py",
        "per_scenario": per_scenario,
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "selected_case_count": len(manifest_cases),
        "semantic_labels_provided": sum(bool(case["expected_text_contains"]) for case in manifest_cases),
        "skipped_source_rows": skipped,
        "cases": manifest_cases,
    }
    manifest_path = output_dir / "humdial_e2e_manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-scenario", type=int, default=DEFAULT_PER_SCENARIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--labels", type=Path, help="optional JSON mapping case_id/relative_path to text labels")
    parser.add_argument("--scenario", dest="scenarios", action="append", help="repeat to restrict scenarios")
    parser.add_argument("--playback-anchor-ms", type=int, default=DEFAULT_PLAYBACK_ANCHOR_MS)
    args = parser.parse_args()
    if args.per_scenario <= 0:
        parser.error("--per-scenario must be positive")
    if args.playback_anchor_ms < 0:
        parser.error("--playback-anchor-ms must be non-negative")
    args.scenarios = set(args.scenarios) if args.scenarios else None
    return args


def main() -> None:
    args = parse_args()
    manifest_path, payload = generate_manifest(
        args.dataset_root,
        args.output_dir,
        per_scenario=args.per_scenario,
        seed=args.seed,
        labels_path=args.labels,
        scenarios=args.scenarios,
        playback_anchor_ms=args.playback_anchor_ms,
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "selected_case_count": payload["selected_case_count"],
                "scenario_counts": payload["scenario_counts"],
                "semantic_labels_provided": payload["semantic_labels_provided"],
                "skipped_source_rows": payload["skipped_source_rows"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
