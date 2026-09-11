"""Run-directory creation and JSON/TXT artifact writers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from vad import SpeechSegment


def _sanitize_stem(audio_name: str) -> str:
    stem = Path(audio_name).stem
    return re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._") or "audio"


def create_run_directory(
    output_root: Path,
    audio_name: str,
    *,
    timestamp: str | None = None,
    run_label: str | None = None,
) -> Path:
    """Create a collision-free per-execution directory below output_root/runs."""
    from datetime import datetime

    timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    label = f"{_sanitize_stem(run_label)}_" if run_label else ""
    base = output_root / "runs" / f"{timestamp}_{label}{_sanitize_stem(audio_name)}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{suffix:02d}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")
    temporary.replace(path)


def _format_time(milliseconds: int) -> str:
    if milliseconds < 0:
        raise ValueError("milliseconds must be non-negative")
    minutes, remainder = divmod(milliseconds, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{minutes}:{seconds:02d}.{millis:03d}"


def _segment_payload(segment: SpeechSegment) -> dict[str, Any]:
    return {
        "segment_id": segment.segment_id,
        "time_range": (
            f"{_format_time(segment.start_ms)}-{_format_time(segment.end_ms)}"
        ),
        "duration_ms": segment.duration_ms,
        "duration_class": segment.duration_class,
        "speaker_composition": segment.speaker_composition,
        "cut_left": segment.cut_left,
        "cut_right": segment.cut_right,
        "asr_text": segment.asr_text,
        "speaker_id": segment.speaker_id,
        "previous_segment_similarity": segment.previous_segment_similarity,
        "cluster_assignment_similarity": segment.cluster_assignment_similarity,
        "overlap_regions": [
            {"start_ms": start_ms, "end_ms": end_ms}
            for start_ms, end_ms in segment.overlap_regions
        ],
        "pyannote_mask": segment.pyannote_mask,
    }


def write_results(
    run_dir: Path,
    audio_name: str,
    audio_duration_ms: int,
    segments: Sequence[SpeechSegment],
    timings: dict[str, float],
) -> None:
    """Write machine-readable records and an operator-readable transcript."""
    _write_json(
        run_dir / "result.json",
        {
            "audio_name": audio_name,
            "audio_duration_ms": audio_duration_ms,
            "segments": [_segment_payload(segment) for segment in segments],
            "timings": timings,
        },
    )
    transcript = run_dir / "result.txt"
    with transcript.open("w", encoding="utf-8", newline="\n") as output:
        for segment in segments:
            output.write(
                f"[{_format_time(segment.start_ms)} - {_format_time(segment.end_ms)}] "
                f"{segment.speaker_id}: {segment.asr_text}\n"
            )


def write_metadata(run_dir: Path, payload: dict[str, Any]) -> None:
    """Write reproducibility information for a pipeline run."""
    _write_json(run_dir / "run_metadata.json", payload)
