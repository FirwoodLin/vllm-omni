# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
import wave
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SCRIPT = Path(__file__).parents[2] / "benchmarks" / "minicpmo" / "generate_humdial_e2e_manifest.py"
E2E_SCRIPT = SCRIPT.with_name("humdial_e2e.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("humdial_e2e_manifest_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_e2e_module():
    spec = importlib.util.spec_from_file_location("humdial_e2e_loader_test", E2E_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_wav(path: Path, duration_ms: int = 2000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(bytes(16_000 * 2 * duration_ms // 1000))


def _write_item(root: Path, language: str, scenario: str, name: str, *, segments: list[dict[str, object]]) -> None:
    wav = root / "test" / f"{language}_test_nondev" / scenario / f"{name}.wav"
    _write_wav(wav, duration_ms=4000)
    wav.with_suffix(".json").write_text(
        json.dumps({"final_duration": 4.0, "speech_segments": segments}),
        encoding="utf-8",
    )


def test_generator_extracts_first_two_segments_and_load_manifest(tmp_path):
    module = _load_module()
    _write_item(
        tmp_path,
        "en",
        "ask",
        "0001_0001",
        segments=[
            {"xmin": 0.1, "xmax": 1.2, "text": "initial question"},
            {"xmin": 2.2, "xmax": 3.4, "text": "follow up question"},
        ],
    )
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"en/ask/0001_0001": ["answer"]}), encoding="utf-8")
    output_dir = tmp_path / "generated"

    manifest_path, payload = module.generate_manifest(
        tmp_path,
        output_dir,
        per_scenario=1,
        seed=7,
        labels_path=labels,
    )

    assert manifest_path.is_file()
    assert payload["selected_case_count"] == 1
    assert payload["semantic_labels_provided"] == 1
    case = payload["cases"][0]
    assert case["case_id"] == "en/ask/0001_0001"
    assert case["initial_text"] == "initial question"
    assert case["interrupt_text"] == "follow up question"
    assert case["expected_text_contains"] == ["answer"]
    assert case["initial_audio"] == "audio/en__ask__0001_0001.initial.wav"
    assert case["interrupt_audio"] == "audio/en__ask__0001_0001.interrupt.wav"

    e2e = _load_e2e_module()
    loaded = e2e.load_manifest(manifest_path)
    assert loaded[0].initial_audio.is_file()
    assert loaded[0].interrupt_audio.is_file()
    assert loaded[0].expected_text_contains == ("answer",)
    with wave.open(str(output_dir / case["initial_audio"]), "rb") as wav_file:
        assert wav_file.getnframes() == 1_200 * 16
    with wave.open(str(output_dir / case["interrupt_audio"]), "rb") as wav_file:
        assert wav_file.getnframes() == 1_200 * 16


def test_generator_skips_single_segment_rows_and_reports_counts(tmp_path):
    module = _load_module()
    _write_item(
        tmp_path,
        "en",
        "pause",
        "single",
        segments=[{"xmin": 0.1, "xmax": 1.2, "text": "one turn"}],
    )
    _write_item(
        tmp_path,
        "en",
        "ask",
        "paired",
        segments=[
            {"xmin": 0.1, "xmax": 1.2, "text": "first"},
            {"xmin": 2.2, "xmax": 3.4, "text": "second"},
        ],
    )
    _, payload = module.generate_manifest(tmp_path, tmp_path / "generated", per_scenario=1, seed=1)
    assert payload["selected_case_count"] == 1
    assert payload["scenario_counts"] == {"ask": 1}
    assert payload["skipped_source_rows"]["single_segment"] == 1


def test_generator_refuses_overwrite_and_rejects_unordered_segments(tmp_path):
    module = _load_module()
    _write_item(
        tmp_path,
        "en",
        "ask",
        "bad",
        segments=[
            {"xmin": 1.0, "xmax": 2.0, "text": "first"},
            {"xmin": 0.5, "xmax": 3.0, "text": "out of order"},
        ],
    )
    with pytest.raises(ValueError, match="ordered by xmin"):
        module.discover_humdial_e2e_candidates(tmp_path)

    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        module.generate_manifest(tmp_path, output_dir)
