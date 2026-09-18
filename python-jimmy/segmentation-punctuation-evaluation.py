"""Reproducible pyannote segmentation-punctuation evaluation helpers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(r"C:\Users\admin\Downloads\sherpa-onnx断句优化测试-0917")
MODEL_ROLES = ("asr", "vad", "speaker", "segmentation")
COMMON_ARGUMENTS = (
    "--audio-format", "pcm", "--sample-rate", "16000", "--channels", "1",
    "--sample-width", "2", "--asr-engine", "paraformer", "--num-clusters", "-1",
    "--cluster-threshold", "0.6", "--segmentation-mode", "vad-pyannote",
)
METRIC_ARGUMENTS = ("--collar-ms", "500", "--boundary-tolerance-ms", "500")
METRIC_PARAMETERS = {"collar_ms": 500, "boundary_tolerance_ms": 500}
STREAMING_INVARIANTS = {
    "--segmentation-chunk-ms": "32",
    "--segmentation-num-threads": "4",
    "--min-duration-on": "0.5",
    "--min-duration-off": "0.5",
    "--change-vote-threshold": "0.5",
}
_SPECIAL_VARIANT_OPTIONS = {"--audio", "--run-label", *STREAMING_INVARIANTS}
_REQUIRED_FAIR_OPTIONS = {"--num-clusters": "-1", "--cluster-threshold": "0.6"}
_FIXED_COMMAND_OPTIONS = {
    "--audio-format": "pcm",
    "--sample-rate": "16000",
    "--channels": "1",
    "--sample-width": "2",
    "--asr-engine": "paraformer",
    "--num-clusters": "-1",
    "--cluster-threshold": "0.6",
    "--segmentation-mode": "vad-pyannote",
}


@dataclass(frozen=True)
class TestCase:
    """One immutable user-confirmed audio and label pair."""

    file_id: str
    pcm: Path
    label: Path


@dataclass(frozen=True)
class TimedText:
    """One strict three-column ASR/label record."""

    segment_id: str
    speaker_id: str
    text: str


TEST_CASES = (
    TestCase(
        file_id="23_asr_1782715267098",
        pcm=Path(r"\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\内部会议\双人安静咨询\promax\23_asr_1782715267098.pcm"),
        label=Path(r"\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\内部会议\双人安静咨询\promax\23_asr_1782715267098_label.txt"),
    ),
    TestCase(
        file_id="asr_1788402212076",
        pcm=Path(r"\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\内部会议\信息流投放实习生面试\asr_1788402212076.pcm"),
        label=Path(r"\\10.88.0.243\xainas\ProMax\测试集\音频测试集\中文\内部会议\信息流投放实习生面试\asr_1788402212076_label.txt"),
    ),
)


@dataclass(frozen=True)
class ModelPaths:
    """Model directories for an invocation; raw path strings retain their original syntax."""

    asr_dir: Path | str
    vad_dir: Path | str
    speaker_dir: Path | str
    segmentation_dir: Path | str


def build_common_arguments(paths: ModelPaths, output_root: Path | str) -> list[str]:
    """Return common arguments while preserving raw local or POSIX path strings exactly."""
    return [
        "--output-root", str(output_root), "--audio-format", "pcm",
        "--sample-rate", "16000", "--channels", "1", "--sample-width", "2",
        "--asr-dir", str(paths.asr_dir), "--vad-dir", str(paths.vad_dir),
        "--speaker-dir", str(paths.speaker_dir), "--segmentation-dir", str(paths.segmentation_dir),
        "--asr-num-threads", "1", "--speaker-num-threads", "2",
        "--vad-threshold", "0.5", "--min-silence-duration", "0.8",
        "--min-speech-duration", "0.25", "--max-speech-duration", "25.0",
        "--pre-speech-pad-duration", "0.0", "--cluster-threshold", "0.6",
        "--num-clusters", "-1", "--min-cluster-duration", "1.0",
        "--centroid-assignment-similarity-threshold", "0.5",
        "--diarization-min-duration-on", "0.5", "--diarization-min-duration-off", "0.5",
        "--asr-engine", "paraformer", "--segmentation-mode", "vad-pyannote",
    ]


ENTRY_SCRIPT_FILENAMES = {
    "baseline": "offline-long-audio-pipeline-asr-speaker.py",
    "streaming": "offline-long-audio-pipeline-asr-speaker-segmentation.py",
}
EXPECTED_ENTRY_SCRIPTS = {
    kind: str(SCRIPT_DIR / filename)
    for kind, filename in ENTRY_SCRIPT_FILENAMES.items()
}
COMMON_OPTION_NAMES = frozenset(
    build_common_arguments(
        ModelPaths(Path("asr"), Path("vad"), Path("speaker"), Path("segmentation")),
        Path("output"),
    )[::2]
)


def build_invocation(kind: str, pcm: Path | str, common: list[str]) -> list[str]:
    """Build a local canonical command; Task 4 materializes remote argv separately."""
    if kind not in EXPECTED_ENTRY_SCRIPTS:
        raise ValueError(f"unsupported invocation kind: {kind}")
    invocation = [sys.executable, EXPECTED_ENTRY_SCRIPTS[kind], "--audio", str(pcm), *common]
    if kind == "streaming":
        for option, value in STREAMING_INVARIANTS.items():
            invocation.extend([option, value])
    return invocation


def _option_map(argv: Sequence[str]) -> dict[str, str]:
    """Parse long options that each take exactly one value."""
    options: dict[str, str] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            index += 1
            continue
        if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
            raise ValueError(f"option requires one value: {token}")
        if token in options:
            raise ValueError(f"duplicate option: {token}")
        options[token] = argv[index + 1]
        index += 2
    return options


def option_value(argv: Sequence[str], option: str) -> str:
    """Return an option value or raise a descriptive error when it is absent."""
    try:
        return _option_map(argv)[option]
    except KeyError as error:
        raise ValueError(f"missing option: {option}") from error


def common_option_map(argv: Sequence[str]) -> dict[str, str]:
    """Return the option mapping that must be equal for baseline and streaming."""
    return {option: value for option, value in _option_map(argv).items() if option not in _SPECIAL_VARIANT_OPTIONS}


def _normalize_cross_platform_path(path: Path | str) -> str:
    """Normalize only path separators so local and remote script paths compare exactly."""
    return str(path).replace("\\", "/")


def _script_under_root(script_root: Path | str, filename: str) -> str:
    """Construct one official entrypoint path under an explicitly trusted script root."""
    normalized_root = _normalize_cross_platform_path(script_root).rstrip("/")
    if not normalized_root:
        normalized_root = "/" if _normalize_cross_platform_path(script_root).startswith("/") else ""
    if not normalized_root:
        raise ValueError("expected script root must be nonempty")
    return f"{normalized_root}/{filename}"


def validate_variant_fairness(
    baseline_argv: Sequence[str],
    streaming_argv: Sequence[str],
    *,
    expected_script_root: Path | str = SCRIPT_DIR,
) -> None:
    """Verify approved differences with entrypoints bound to an explicit trusted script root."""
    for variant, argv in (("baseline", baseline_argv), ("streaming", streaming_argv)):
        expected_script = _script_under_root(expected_script_root, ENTRY_SCRIPT_FILENAMES[variant])
        actual_script = argv[1] if len(argv) >= 2 else None
        if not isinstance(actual_script, str) or _normalize_cross_platform_path(actual_script) != expected_script:
            raise ValueError(
                f"{variant} entry script must equal {expected_script!r}, got {actual_script!r}"
            )

    baseline_options = _option_map(baseline_argv)
    streaming_options = _option_map(streaming_argv)
    for option in COMMON_OPTION_NAMES:
        for variant, options in (("baseline", baseline_options), ("streaming", streaming_options)):
            if option not in options:
                raise ValueError(f"{variant} missing required common option: {option}")

    baseline_audio = baseline_options.get("--audio")
    streaming_audio = streaming_options.get("--audio")
    if baseline_audio is None or streaming_audio is None:
        raise ValueError("baseline and streaming invocations must both set --audio")
    if baseline_audio != streaming_audio:
        raise ValueError(f"baseline and streaming --audio values must match: {baseline_audio!r} != {streaming_audio!r}")

    for option, expected_value in STREAMING_INVARIANTS.items():
        if option in baseline_options:
            raise ValueError(f"baseline invocation must not set {option}")
        if streaming_options.get(option) != expected_value:
            raise ValueError(f"streaming {option} must equal {expected_value}, got {streaming_options.get(option)!r}")

    if common_option_map(baseline_argv) != common_option_map(streaming_argv):
        raise ValueError("baseline and streaming common arguments differ")

    for option, expected_value in _REQUIRED_FAIR_OPTIONS.items():
        for name, options in (("baseline", baseline_options), ("streaming", streaming_options)):
            if options.get(option) != expected_value:
                raise ValueError(f"{name} {option} must equal {expected_value}, got {options.get(option)!r}")


def _segment_times(segment: dict[str, object]) -> tuple[int, int]:
    """Parse the trailing ``_<start_ms>_<end_ms>`` fields from a segment ID."""
    segment_id = segment.get("segment_id")
    if not isinstance(segment_id, str):
        raise ValueError("segment_id must be a string ending in _<start_ms>_<end_ms>")
    parts = segment_id.rsplit("_", 2)
    if len(parts) != 3 or not parts[0] or not parts[1].isdigit() or not parts[2].isdigit():
        raise ValueError(f"malformed segment_id: {segment_id!r}")
    start_ms, end_ms = int(parts[1]), int(parts[2])
    if end_ms <= start_ms:
        raise ValueError(f"invalid segment interval: {start_ms}-{end_ms}")
    if "duration_ms" in segment:
        try:
            duration_ms = int(segment["duration_ms"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid duration_ms for {segment_id!r}") from error
        if duration_ms != end_ms - start_ms:
            raise ValueError(f"inconsistent duration_ms for {segment_id!r}: expected {end_ms - start_ms}, got {duration_ms}")
    return start_ms, end_ms


def write_metrics_asr_file(output_dir: Path, file_id: str, result: dict[str, object]) -> Path:
    """Convert result.json segments to the three-column diarization metric format."""
    segments = result.get("segments")
    if not isinstance(segments, list):
        raise ValueError("result.json must contain a segments list")
    rows: list[str] = []
    for item in segments:
        if not isinstance(item, dict):
            raise ValueError("result segment must be an object")
        start_ms, end_ms = _segment_times(item)
        speaker = str(item.get("speaker_id", "-")).strip().strip("()") or "-"
        text = str(item.get("asr_text", "")).replace("\r", " ").replace("\n", " ").strip()
        rows.append(f"{file_id}_{start_ms}_{end_ms - start_ms} {speaker} {text}".rstrip())
    output_path = output_dir / f"{file_id}_asr.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return output_path


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 using bounded 1 MiB reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash one model file or a deterministic tree of model files."""
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise ValueError(f"model path does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"model directory contains no files: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def build_model_inventory(paths: ModelPaths) -> dict[str, dict[str, str]]:
    """Inventory model directories, retaining raw strings while converting only for local hashing."""
    inventory: dict[str, dict[str, str]] = {}
    for role in MODEL_ROLES:
        directory = getattr(paths, f"{role}_dir")
        inventory[role] = {
            "directory": str(directory),
            "sha256": sha256_path(Path(directory)),
        }
    return inventory


def _copy_file_atomically(source: Path, destination: Path) -> str:
    source_hash = sha256_file(source)
    if destination.is_file() and sha256_file(destination) == source_hash:
        return source_hash
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with source.open("rb") as src, temporary.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        if sha256_file(temporary) != source_hash:
            raise ValueError(f"copied file hash mismatch: {source}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return source_hash


def prepare_inputs(output_root: Path, *, cases: Sequence[TestCase] = TEST_CASES) -> list[dict[str, Any]]:
    """Freeze only the declared PCM/label inputs under ``01_input``."""
    records: list[dict[str, Any]] = []
    for case in cases:
        if not case.pcm.is_file():
            raise FileNotFoundError(f"PCM input does not exist: {case.pcm}")
        if not case.label.is_file():
            raise FileNotFoundError(f"label input does not exist: {case.label}")
        frozen_pcm = output_root / "01_input" / "pcm" / case.pcm.name
        frozen_label = output_root / "01_input" / "labels" / case.label.name
        records.append({
            "file_id": case.file_id,
            "pcm": {"source_path": str(case.pcm), "path": str(frozen_pcm), "sha256": _copy_file_atomically(case.pcm, frozen_pcm)},
            "label": {"source_path": str(case.label), "path": str(frozen_label), "sha256": _copy_file_atomically(case.label, frozen_label)},
        })
    return records


def parse_three_column_file(path: Path) -> list[TimedText]:
    """Read strict ``id speaker [text]`` records, preserving empty text."""
    records: list[TimedText] = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=2)
        if len(fields) < 2 or not fields[0].strip() or not fields[1].strip():
            raise ValueError(f"{path}:{number}: expected id speaker [text]")
        records.append(TimedText(fields[0], fields[1].strip().strip("()"), fields[2] if len(fields) == 3 else ""))
    return records


def _record_start_ms(record: TimedText) -> int:
    parts = record.segment_id.rsplit("_", 2)
    if len(parts) != 3 or not parts[0] or not parts[1].isdigit() or not parts[2].isdigit():
        raise ValueError(f"malformed timed segment ID: {record.segment_id!r}")
    return int(parts[1])


def _text_normalization(text: str, language: str) -> str:
    evaluation_path = SCRIPT_DIR / "evaluation.py"
    spec = importlib.util.spec_from_file_location("segmentation_evaluation_normalization", evaluation_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load text normalization from {evaluation_path}")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    return evaluation.text_normalization(text, language)


def _edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int, int]:
    """Return total errors and I/D/S counts using deterministic Levenshtein DP."""
    cells: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(len(hypothesis) + 1)] for _ in range(len(reference) + 1)
    ]
    for row in range(1, len(reference) + 1):
        cells[row][0] = (row, 0, row, 0)
    for column in range(1, len(hypothesis) + 1):
        cells[0][column] = (column, column, 0, 0)
    for row in range(1, len(reference) + 1):
        for column in range(1, len(hypothesis) + 1):
            if reference[row - 1] == hypothesis[column - 1]:
                cells[row][column] = cells[row - 1][column - 1]
                continue
            previous = cells[row - 1][column - 1]
            substitute = (previous[0] + 1, previous[1], previous[2], previous[3] + 1)
            previous = cells[row - 1][column]
            delete = (previous[0] + 1, previous[1], previous[2] + 1, previous[3])
            previous = cells[row][column - 1]
            insert = (previous[0] + 1, previous[1] + 1, previous[2], previous[3])
            cells[row][column] = min(substitute, delete, insert)
    return cells[-1][-1]


def _write_wer_artifacts(summary: Mapping[str, Any], output_root: Path, scheme: str, file_id: str) -> Path:
    metrics_dir = output_root / scheme / file_id / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "wer.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    detail = "\n".join([
        "# 中文 WER",
        "",
        f"- reference_text: {summary['reference_text']}",
        f"- hypothesis_text: {summary['hypothesis_text']}",
        f"- reference_tokens: {summary['reference_tokens']}",
        f"- insertions: {summary['insertions']}",
        f"- deletions: {summary['deletions']}",
        f"- substitutions: {summary['substitutions']}",
        f"- errors: {summary['errors']}",
        f"- wer: {summary['wer']}",
        "",
    ])
    (metrics_dir / "wer_detail.md").write_text(detail, encoding="utf-8")
    return metrics_dir


def compute_wer(
    label: Path,
    predicted: Path,
    *,
    language: str = "zh",
    output_root: Path | None = None,
    scheme: str | None = None,
    file_id: str | None = None,
) -> dict[str, Any]:
    """Compute corpus WER after repository-standard text normalization."""
    reference_records = sorted(parse_three_column_file(label), key=_record_start_ms)
    hypothesis_records = sorted(parse_three_column_file(predicted), key=_record_start_ms)
    reference_text = _text_normalization(" ".join(record.text for record in reference_records), language)
    hypothesis_text = _text_normalization(" ".join(record.text for record in hypothesis_records), language)
    reference_tokens = reference_text.split()
    hypothesis_tokens = hypothesis_text.split()
    errors, insertions, deletions, substitutions = _edit_counts(reference_tokens, hypothesis_tokens)
    summary: dict[str, Any] = {
        "language": language,
        "reference_text": reference_text,
        "hypothesis_text": hypothesis_text,
        "reference_tokens": len(reference_tokens),
        "hypothesis_tokens": len(hypothesis_tokens),
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
        "errors": errors,
        "wer": errors / max(len(reference_tokens), 1),
    }
    if output_root is not None or scheme is not None or file_id is not None:
        if output_root is None or scheme is None or file_id is None:
            raise ValueError("output_root, scheme, and file_id must be supplied together")
        _write_wer_artifacts(summary, output_root, scheme, file_id)
    return summary


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _is_absolute_directory(value: str) -> bool:
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


def _frozen_input_path(output_root: str, directory: str, filename: str) -> str:
    if PureWindowsPath(output_root).is_absolute():
        root = PureWindowsPath(output_root)
    elif PurePosixPath(output_root).is_absolute():
        root = PurePosixPath(output_root)
    else:
        raise ValueError(f"command --output-root must be absolute: {output_root!r}")
    return str(root / "01_input" / directory / filename)


def _require_mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def validate_manifest(manifest: Mapping[str, Any], *, verify_frozen_files: bool = False) -> None:
    """Validate local canonical commands; Task 4 renders and records remote argv separately."""
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must equal 1")
    status = manifest.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError("manifest status must be nonempty")
    expected_cases = {case.file_id: case for case in TEST_CASES}
    test_cases = manifest.get("test_cases")
    if not isinstance(test_cases, list):
        raise ValueError("manifest test_cases must be a list")
    found_ids = [item.get("file_id") if isinstance(item, Mapping) else None for item in test_cases]
    if found_ids != list(expected_cases):
        raise ValueError(f"manifest file IDs must exactly equal {list(expected_cases)!r}")
    frozen_audio: dict[str, str] = {}
    frozen_labels: dict[str, str] = {}
    frozen_records: dict[str, Mapping[str, Mapping[str, Any]]] = {}
    for item in test_cases:
        case = expected_cases[item["file_id"]]
        for name, source in (("pcm", case.pcm), ("label", case.label)):
            record = _require_mapping(item.get(name), f"{case.file_id} {name}")
            if record.get("source_path") != str(source):
                raise ValueError(f"{case.file_id} {name} source_path is not the fixed user-confirmed path")
            if not isinstance(record.get("path"), str) or not record["path"]:
                raise ValueError(f"{case.file_id} {name} path must be nonempty")
            if not _is_sha256(record.get("sha256")):
                raise ValueError(f"{case.file_id} {name} SHA-256 must be a 64-character hexadecimal value")
            if name == "pcm":
                frozen_audio[case.file_id] = record["path"]
            else:
                frozen_labels[case.file_id] = record["path"]
        frozen_records[case.file_id] = {"pcm": item["pcm"], "label": item["label"]}

    commands = _require_mapping(manifest.get("commands"), "commands")
    if set(commands) != {"baseline", "streaming"}:
        raise ValueError("commands must contain baseline and streaming only")
    for kind in ("baseline", "streaming"):
        if not isinstance(commands[kind], Mapping) or list(commands[kind]) != list(expected_cases):
            raise ValueError(f"{kind} commands must exactly match manifest file IDs")
    manifest_output_root: str | None = None
    for file_id in expected_cases:
        baseline = commands["baseline"][file_id]
        streaming = commands["streaming"][file_id]
        if not isinstance(baseline, list) or not isinstance(streaming, list) or not all(isinstance(value, str) for value in baseline + streaming):
            raise ValueError(f"{file_id} commands must be string argv lists")
        validate_variant_fairness(baseline, streaming)
        if option_value(baseline, "--audio") != frozen_audio[file_id]:
            raise ValueError(f"{file_id} command audio must equal the frozen PCM path")
        options = _option_map(baseline)
        output_root = options.get("--output-root")
        if output_root is None:
            raise ValueError(f"{file_id} command must set --output-root")
        if manifest_output_root is None:
            manifest_output_root = output_root
        elif output_root != manifest_output_root:
            raise ValueError("all commands must use one single consistent absolute --output-root")
        case = expected_cases[file_id]
        expected_pcm = _frozen_input_path(output_root, "pcm", case.pcm.name)
        expected_label = _frozen_input_path(output_root, "labels", case.label.name)
        if frozen_audio[file_id] != expected_pcm:
            raise ValueError(f"{file_id} PCM frozen path must equal {expected_pcm!r}")
        if frozen_labels[file_id] != expected_label:
            raise ValueError(f"{file_id} label frozen path must equal {expected_label!r}")
        for option, expected_value in _FIXED_COMMAND_OPTIONS.items():
            if options.get(option) != expected_value:
                raise ValueError(f"{file_id} {option} must equal {expected_value}")

    models = _require_mapping(manifest.get("models"), "models")
    if set(models) != {"local", "remote"}:
        raise ValueError("models must contain local and remote inventories")
    for role in MODEL_ROLES:
        local = _require_mapping(_require_mapping(models["local"], "local models").get(role), f"local {role} model")
        remote = _require_mapping(_require_mapping(models["remote"], "remote models").get(role), f"remote {role} model")
        for location, model in (("local", local), ("remote", remote)):
            if not isinstance(model.get("directory"), str) or not model["directory"]:
                raise ValueError(f"{location} {role} model directory must be nonempty")
            if not _is_absolute_directory(model["directory"]):
                raise ValueError(f"{location} {role} model directory must be absolute")
        if not _is_sha256(local.get("sha256")):
            raise ValueError(f"local {role} model SHA-256 must be a 64-character hexadecimal value")
        remote_sha256 = remote.get("sha256")
        if remote_sha256 == "pending":
            if status != "preflight-pending":
                raise ValueError(
                    f"remote {role} model SHA-256 may be 'pending' only when status is 'preflight-pending'"
                )
        else:
            if not _is_sha256(remote_sha256):
                raise ValueError(f"remote {role} model SHA-256 must be a 64-character hexadecimal value")
            if local["sha256"] != remote_sha256:
                raise ValueError(f"{role.upper()} model SHA-256 differs between local and remote")
        option = f"--{role}-dir"
        expected_local_directory = local["directory"]
        for file_id in expected_cases:
            for kind in ("baseline", "streaming"):
                command_directory = option_value(commands[kind][file_id], option)
                if command_directory != expected_local_directory:
                    raise ValueError(
                        f"local {role} model directory must equal audited inventory: "
                        f"{command_directory!r} != {expected_local_directory!r}"
                    )

    if verify_frozen_files:
        for file_id, records in frozen_records.items():
            for name, display_name in (("pcm", "PCM"), ("label", "label")):
                record = records[name]
                frozen_path = Path(record["path"])
                if not frozen_path.is_file():
                    raise FileNotFoundError(f"{file_id} frozen {display_name} file is missing: {frozen_path}")
                actual_sha256 = sha256_file(frozen_path)
                if actual_sha256 != record["sha256"]:
                    raise ValueError(
                        f"{file_id} frozen {display_name} SHA-256 differs: "
                        f"{actual_sha256} != {record['sha256']}"
                    )

    if manifest.get("metrics_parameters") != METRIC_PARAMETERS:
        metrics = manifest.get("metrics_parameters")
        if isinstance(metrics, Mapping):
            for key, value in METRIC_PARAMETERS.items():
                if metrics.get(key) != value:
                    raise ValueError(f"metrics parameter {key} must equal {value}")
        raise ValueError("metrics_parameters must equal the fixed evaluation settings")
    for location in ("windows", "remote"):
        repository = _require_mapping(_require_mapping(manifest.get("repositories"), "repositories").get(location), f"{location} repository")
        if not isinstance(repository.get("commit"), str) or not repository["commit"]:
            raise ValueError(f"{location} repository commit must be nonempty")
        if not isinstance(repository.get("status_porcelain"), str):
            raise ValueError(f"{location} repository status_porcelain must be a string")
    if not isinstance(manifest.get("created_at"), str) or not manifest["created_at"]:
        raise ValueError("manifest created_at must be nonempty")


def create_manifest(
    path: Path,
    *,
    inputs: Sequence[Mapping[str, Any]],
    commands: Mapping[str, Mapping[str, Sequence[str]]],
    local_models: Mapping[str, Mapping[str, str]],
    remote_models: Mapping[str, Mapping[str, str]],
    windows_repo: Mapping[str, str],
    remote_repo: Mapping[str, str],
    created_at: str,
    status: str,
) -> dict[str, Any]:
    """Create and validate a UTF-8 reproducibility manifest atomically."""
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": created_at,
        "repositories": {"windows": dict(windows_repo), "remote": dict(remote_repo)},
        "test_cases": [dict(item) for item in inputs],
        "commands": {kind: {file_id: list(argv) for file_id, argv in per_case.items()} for kind, per_case in commands.items()},
        "models": {"local": {role: dict(value) for role, value in local_models.items()}, "remote": {role: dict(value) for role, value in remote_models.items()}},
        "metrics_parameters": dict(METRIC_PARAMETERS),
        "status": status,
    }
    validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return manifest



METRICS_TOOL_FILENAME = "speaker_diarization_metrics.py"
DEFAULT_METRICS_SOURCE = Path(
    r"D:\code\proMax\tz-llm-sdk\agent-sdk-test\python\speaker_diarization_metrics.py"
)
SCHEMES = ("baseline", "streaming")
BOUNDARY_DIFFERENCE_FIELDS = (
    "file_id",
    "reference_start_ms",
    "reference_end_ms",
    "reference_boundary_ms",
    "baseline_match_ms",
    "streaming_match_ms",
    "tolerance_ms",
    "classification",
    "before_text",
    "after_text",
)


def _write_json(path: Path, payload: object) -> None:
    """Atomically write a UTF-8 JSON artifact with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def install_metrics_tool(source: Path, tools_dir: Path) -> Path:
    """Record the source SHA-256, then atomically copy the approved metrics script."""
    source = Path(source)
    tools_dir = Path(tools_dir)
    if source.name != METRICS_TOOL_FILENAME:
        raise ValueError(f"metrics source must be named {METRICS_TOOL_FILENAME!r}: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"metrics source is missing: {source}")
    digest = sha256_file(source)
    destination = tools_dir / METRICS_TOOL_FILENAME
    receipt = tools_dir / "speaker_diarization_metrics.sha256.json"
    tools_dir.mkdir(parents=True, exist_ok=True)
    _write_json(receipt, {
        "source_path": str(source),
        "sha256": digest,
        "destination_path": str(destination),
    })
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != digest:
        temporary.unlink(missing_ok=True)
        raise ValueError("metrics tool SHA-256 changed while copying")
    temporary.replace(destination)
    return destination


def _read_json_mapping(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot parse {description}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return payload


def _extract_der_components_from_tool(
    tool_path: Path,
    normalized_result_root: Path,
    input_labels_dir: Path,
) -> tuple[dict[str, float] | None, dict[str, dict[str, float]], str | None]:
    """Expose aggregate and per-file DER components from the exact copied metric source.

    Its CLI publishes aggregate DER but not miss/false-alarm/confusion components.
    The copied source's public ``evaluate_paths`` API retains those values, so this
    reads them without reimplementing diarization scoring.
    """
    try:
        module_name = f"_copied_speaker_metrics_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, tool_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load copied metrics tool: {tool_path}")
        metrics_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = metrics_module
        try:
            spec.loader.exec_module(metrics_module)
        finally:
            sys.modules.pop(module_name, None)
        report = metrics_module.evaluate_paths(
            results_dir=normalized_result_root,
            labels_dir=input_labels_dir,
            boundary_tolerance_ms=METRIC_PARAMETERS["boundary_tolerance_ms"],
            collar_ms=METRIC_PARAMETERS["collar_ms"],
        )
        evaluated = [item for item in report.files if item.get("error") is None]
        if not evaluated:
            return None, {}, "copied metrics tool evaluated no files while extracting DER components"
        aggregate = {key: 0.0 for key in ("miss", "false_alarm", "confusion", "total")}
        per_file: dict[str, dict[str, float]] = {}
        for item in evaluated:
            file_id = item.get("file_id")
            values = item.get("der_components")
            if not isinstance(file_id, str) or not isinstance(values, Mapping):
                return None, {}, "copied metrics tool did not expose per-file DER components"
            parsed: dict[str, float] = {}
            for key in aggregate:
                value = values.get(key)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    return None, {}, f"invalid copied metrics {key} component: {value!r}"
                parsed[key] = float(value)
                aggregate[key] += parsed[key]
            per_file[file_id] = parsed
        return aggregate, per_file, None
    except Exception as error:  # CLI output remains authoritative; retain the extraction limitation.
        return None, {}, f"unable to extract DER components from copied metrics tool: {error}"


def run_speaker_metrics(
    scheme: str,
    tools_dir: Path,
    normalized_result_root: Path,
    input_labels_dir: Path,
    metrics_dir: Path,
) -> dict[str, Any]:
    """Run the copied metrics CLI and retain its command, output and failure logs."""
    if scheme not in SCHEMES:
        raise ValueError(f"unsupported scheme: {scheme}")
    tools_dir = Path(tools_dir)
    normalized_result_root = Path(normalized_result_root)
    input_labels_dir = Path(input_labels_dir)
    metrics_dir = Path(metrics_dir)
    tool_path = tools_dir / METRICS_TOOL_FILENAME
    if not tool_path.is_file():
        raise FileNotFoundError(f"copied metrics tool is missing: {tool_path}")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(tool_path),
        "--results-dir", str(normalized_result_root),
        "--labels-dir", str(input_labels_dir),
        "--output-dir", str(metrics_dir),
        "--boundary-tolerance-ms", str(METRIC_PARAMETERS["boundary_tolerance_ms"]),
        "--collar-ms", str(METRIC_PARAMETERS["collar_ms"]),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    stdout_path = metrics_dir / "speaker_metrics.stdout.log"
    stderr_path = metrics_dir / "speaker_metrics.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    summary_path = metrics_dir / "speaker_diarization_summary.json"
    boundary_path = metrics_dir / "speaker_diarization_boundary_details.csv"
    per_file_path = metrics_dir / "speaker_diarization_per_file.json"
    result: dict[str, Any] = {
        "scheme": scheme,
        "status": "metrics_failed" if completed.returncode else "metrics_missing_summary",
        "returncode": completed.returncode,
        "command": command,
        "metrics_dir": str(metrics_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "summary_path": str(summary_path),
        "boundary_details_path": str(boundary_path),
        "per_file_path": str(per_file_path),
        "summary": None,
        "per_file": None,
        "der_components": None,
        "der_components_by_file": {},
        "der_components_error": None,
    }
    if completed.returncode:
        return result
    if not summary_path.is_file():
        return result
    try:
        summary = _read_json_mapping(summary_path, "speaker diarization summary")
        counts = summary.get("counts")
        metrics = summary.get("metrics")
        if not isinstance(counts, Mapping) or not isinstance(metrics, Mapping):
            raise ValueError("speaker diarization summary lacks counts or metrics object")
        result["summary"] = summary
        if per_file_path.is_file():
            result["per_file"] = _read_json_mapping(per_file_path, "speaker diarization per-file result")
        components, components_by_file, component_error = _extract_der_components_from_tool(
            tool_path, normalized_result_root, input_labels_dir,
        )
        result["der_components"] = components
        result["der_components_by_file"] = components_by_file
        result["der_components_error"] = component_error
        result["status"] = "metrics_completed"
    except ValueError as error:
        result["status"] = "metrics_invalid_summary"
        result["summary_error"] = str(error)
    return result


def _as_metric_number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def select_winner(metrics: Mapping[str, Mapping[str, object]]) -> str | None:
    """Apply the fixed hit-rate-first, DER-second comparison policy."""
    baseline = metrics.get("baseline")
    streaming = metrics.get("streaming")
    if not isinstance(baseline, Mapping) or not isinstance(streaming, Mapping):
        return None
    baseline_hit = _as_metric_number(baseline.get("hit_rate"))
    streaming_hit = _as_metric_number(streaming.get("hit_rate"))
    baseline_der = _as_metric_number(baseline.get("der"))
    streaming_der = _as_metric_number(streaming.get("der"))
    if None in (baseline_hit, streaming_hit, baseline_der, streaming_der):
        return None
    assert baseline_hit is not None and streaming_hit is not None
    assert baseline_der is not None and streaming_der is not None
    if abs(baseline_hit - streaming_hit) >= 0.01:
        return "baseline" if baseline_hit > streaming_hit else "streaming"
    if baseline_der < streaming_der:
        return "baseline"
    if streaming_der < baseline_der:
        return "streaming"
    return None


def make_boundary_difference(
    *,
    reference_boundary_ms: int,
    baseline_match_ms: int | None,
    streaming_match_ms: int | None,
    tolerance_ms: int,
    before_text: str,
    after_text: str,
    reference_start_ms: int | None = None,
    reference_end_ms: int | None = None,
    file_id: str = "",
) -> dict[str, object]:
    """Create one auditable row explaining whether either scheme hit a reference turn."""
    if baseline_match_ms is not None and streaming_match_ms is not None:
        classification = "both_hit"
    elif baseline_match_ms is not None:
        classification = "baseline_only_hit"
    elif streaming_match_ms is not None:
        classification = "streaming_only_hit"
    else:
        classification = "both_miss"
    return {
        "file_id": file_id,
        "reference_start_ms": reference_boundary_ms if reference_start_ms is None else reference_start_ms,
        "reference_end_ms": reference_boundary_ms if reference_end_ms is None else reference_end_ms,
        "reference_boundary_ms": reference_boundary_ms,
        "baseline_match_ms": baseline_match_ms,
        "streaming_match_ms": streaming_match_ms,
        "tolerance_ms": tolerance_ms,
        "classification": classification,
        "before_text": before_text,
        "after_text": after_text,
    }


def _csv_string_value(row: Mapping[str, str | None], name: str, path: Path) -> str:
    value = row.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{path} has a missing value for {name}")
    return value.strip()


def _int_csv_value(row: Mapping[str, str | None], name: str, path: Path) -> int:
    value = _csv_string_value(row, name, path)
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{path} has invalid {name}: {value!r}") from error


def _optional_int_csv_value(row: Mapping[str, str | None], name: str, path: Path) -> int | None:
    value = _csv_string_value(row, name, path)
    return None if not value else _int_csv_value(row, name, path)


def _read_boundary_details(path: Path) -> dict[tuple[str, int, int], dict[str, object]]:
    if not path.is_file():
        return {}
    required = {
        "文件ID", "区间起始毫秒", "区间结束毫秒", "扩展后起始毫秒", "扩展后结束毫秒",
        "匹配预测边界毫秒", "状态",
    }
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"boundary details CSV has unexpected columns: {path}")
        details: dict[tuple[str, int, int], dict[str, object]] = {}
        for row in reader:
            if None in row:
                raise ValueError(f"boundary details CSV has unexpected extra fields: {path}")
            file_id = _csv_string_value(row, "文件ID", path)
            if not file_id:
                raise ValueError(f"boundary details CSV has an empty file ID: {path}")
            start_ms = _int_csv_value(row, "区间起始毫秒", path)
            end_ms = _int_csv_value(row, "区间结束毫秒", path)
            key = (file_id, start_ms, end_ms)
            if key in details:
                raise ValueError(f"duplicate reference boundary in {path}: {key}")
            matched_ms = _optional_int_csv_value(row, "匹配预测边界毫秒", path)
            status = _csv_string_value(row, "状态", path)
            if status == "命中" and matched_ms is None:
                raise ValueError(f"boundary details CSV has a hit without a matched prediction boundary: {path}")
            if status == "未命中" and matched_ms is not None:
                raise ValueError(f"boundary details CSV has a miss with a matched prediction boundary: {path}")
            if status not in {"命中", "未命中"}:
                raise ValueError(f"boundary details CSV has invalid status {status!r}: {path}")
            details[key] = {
                "file_id": file_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "matched_ms": matched_ms,
                "tolerance_ms": max(
                    abs(_int_csv_value(row, "扩展后起始毫秒", path) - min(start_ms, end_ms)),
                    abs(_int_csv_value(row, "扩展后结束毫秒", path) - max(start_ms, end_ms)),
                ),
            }
    return details


def _reference_boundary_texts(labels_dir: Path) -> dict[tuple[str, int, int], tuple[str, str]]:
    """Mirror the copied tool's same-speaker turn merge while retaining adjacent text."""
    texts: dict[tuple[str, int, int], tuple[str, str]] = {}
    for case in TEST_CASES:
        label_path = Path(labels_dir) / case.label.name
        if not label_path.is_file():
            continue
        records = sorted(parse_three_column_file(label_path), key=lambda record: (_record_start_ms(record), record.segment_id))
        turns: list[dict[str, object]] = []
        for record in records:
            if record.speaker_id.lower() == "multi":
                continue
            start_ms = _record_start_ms(record)
            duration_ms = int(record.segment_id.rsplit("_", 1)[1])
            end_ms = start_ms + duration_ms
            if turns and turns[-1]["speaker_id"] == record.speaker_id:
                turns[-1]["end_ms"] = max(int(turns[-1]["end_ms"]), end_ms)
                turns[-1]["last_text"] = record.text
            else:
                turns.append({
                    "speaker_id": record.speaker_id,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "first_text": record.text,
                    "last_text": record.text,
                })
        for previous, following in zip(turns, turns[1:]):
            key = (case.file_id, int(previous["end_ms"]), int(following["start_ms"]))
            texts[key] = (str(previous["last_text"]), str(following["first_text"]))
    return texts


def _merge_boundary_detail_maps(
    baseline: Mapping[tuple[str, int, int], Mapping[str, object]],
    streaming: Mapping[tuple[str, int, int], Mapping[str, object]],
    label_texts: Mapping[tuple[str, int, int], tuple[str, str]],
) -> list[dict[str, object]]:
    differences: list[dict[str, object]] = []
    for key in sorted(set(baseline) | set(streaming)):
        file_id, start_ms, end_ms = key
        left = baseline.get(key, {})
        right = streaming.get(key, {})
        tolerance = int(left.get("tolerance_ms", right.get("tolerance_ms", METRIC_PARAMETERS["boundary_tolerance_ms"])))
        before_text, after_text = label_texts.get(key, ("", ""))
        differences.append(make_boundary_difference(
            file_id=file_id,
            reference_start_ms=start_ms,
            reference_end_ms=end_ms,
            reference_boundary_ms=start_ms if start_ms == end_ms else (start_ms + end_ms) // 2,
            baseline_match_ms=left.get("matched_ms") if isinstance(left.get("matched_ms"), int) else None,
            streaming_match_ms=right.get("matched_ms") if isinstance(right.get("matched_ms"), int) else None,
            tolerance_ms=tolerance,
            before_text=before_text,
            after_text=after_text,
        ))
    return differences


def merge_boundary_differences(
    baseline_boundary_path: Path,
    streaming_boundary_path: Path,
    labels_dir: Path,
) -> list[dict[str, object]]:
    """Merge two copied-tool boundary CSVs by file and reference interval."""
    return _merge_boundary_detail_maps(
        _read_boundary_details(Path(baseline_boundary_path)),
        _read_boundary_details(Path(streaming_boundary_path)),
        _reference_boundary_texts(Path(labels_dir)),
    )


def _load_wer_by_file(output_root: Path, scheme: str) -> dict[str, dict[str, object]]:
    values: dict[str, dict[str, object]] = {}
    for case in TEST_CASES:
        path = output_root / scheme / case.file_id / "metrics" / "wer.json"
        if not path.is_file():
            values[case.file_id] = {"status": "missing", "path": str(path), "wer": None}
            continue
        try:
            payload = _read_json_mapping(path, "WER result")
            wer = _as_metric_number(payload.get("wer"))
            if wer is None:
                raise ValueError("WER result must contain a finite numeric wer value")
            values[case.file_id] = {
                "status": "completed",
                "path": str(path),
                "wer": wer,
                "metrics": payload,
            }
        except ValueError as error:
            values[case.file_id] = {"status": "invalid", "path": str(path), "wer": None, "error": str(error)}
    return values


def _find_runtime_metadata(output_root: Path) -> dict[str, dict[str, dict[str, object]]]:
    """Best-effort discovery of Task-4 run metadata without executing remote code."""
    found: dict[str, dict[str, dict[str, object]]] = {scheme: {} for scheme in SCHEMES}
    remote_root = Path(output_root) / "remote-artifacts"
    if not remote_root.is_dir():
        return found
    for path in remote_root.rglob("run_metadata.json"):
        try:
            payload = _read_json_mapping(path, "run metadata")
        except ValueError:
            continue
        path_text = str(path).replace("\\", "/")
        scheme = next((name for name in SCHEMES if f"/{name}/" in path_text), None)
        if scheme is None:
            continue
        file_id = next((case.file_id for case in TEST_CASES if case.file_id in path_text), None)
        if file_id is None:
            file_id = next((case.file_id for case in TEST_CASES if case.file_id == payload.get("file_id")), None)
        if file_id is not None:
            found[scheme][file_id] = {"path": str(path), "metadata": payload}
    return found


def _boundary_details_issue(result: Mapping[str, object]) -> str | None:
    """Return a concrete failure reason when a scheme lacks its boundary artifact."""
    value = result.get("boundary_details_path")
    if not isinstance(value, str) or not value.strip():
        return "boundary detail CSV is missing (path was not recorded)"
    if not Path(value).is_file():
        return f"boundary detail CSV is missing: {value}"
    return None


def _nonnegative_integer_metric(value: object) -> int | None:
    numeric = _as_metric_number(value)
    if numeric is None or numeric < 0 or not numeric.is_integer():
        return None
    return int(numeric)


def _reference_boundary_texts_for_validation(labels_dir: Path) -> dict[tuple[str, int, int], tuple[str, str]]:
    missing = [case.label.name for case in TEST_CASES if not (Path(labels_dir) / case.label.name).is_file()]
    if missing:
        raise ValueError(f"reference label files are missing: {', '.join(missing)}")
    return _reference_boundary_texts(Path(labels_dir))


def _validate_boundary_details(
    result: Mapping[str, object],
    expected_keys: set[tuple[str, int, int]],
) -> tuple[dict[tuple[str, int, int], dict[str, object]] | None, str | None]:
    path_issue = _boundary_details_issue(result)
    if path_issue is not None:
        return None, f"boundary detail evidence is incomplete: {path_issue}"
    path = Path(str(result["boundary_details_path"]))
    try:
        details = _read_boundary_details(path)
    except (OSError, csv.Error, ValueError) as error:
        return None, f"boundary detail evidence is incomplete: boundary detail CSV is invalid: {error}"
    actual_keys = set(details)
    if actual_keys != expected_keys:
        missing = len(expected_keys - actual_keys)
        unexpected = len(actual_keys - expected_keys)
        return None, (
            "boundary detail evidence is incomplete: reference boundary keys do not match labels "
            f"(missing={missing}, unexpected={unexpected})"
        )
    summary = result.get("summary")
    metrics = summary.get("metrics") if isinstance(summary, Mapping) else None
    total_count = _nonnegative_integer_metric(metrics.get("speaker_change_count")) if isinstance(metrics, Mapping) else None
    if total_count is None or total_count != len(details):
        return None, (
            "boundary detail evidence is incomplete: summary speaker_change_count does not match "
            f"reference boundary count ({total_count!r} != {len(details)})"
        )
    per_file = result.get("per_file")
    files = per_file.get("files") if isinstance(per_file, Mapping) else None
    if not isinstance(files, list):
        return None, "boundary detail evidence is incomplete: per-file speaker metrics are missing"
    metrics_by_file = {
        item.get("file_id"): item.get("metrics")
        for item in files
        if isinstance(item, Mapping) and isinstance(item.get("file_id"), str) and isinstance(item.get("metrics"), Mapping)
    }
    for case in TEST_CASES:
        count = _nonnegative_integer_metric(
            metrics_by_file.get(case.file_id, {}).get("speaker_change_count")
            if isinstance(metrics_by_file.get(case.file_id), Mapping) else None
        )
        actual_count = sum(1 for key in details if key[0] == case.file_id)
        if count is None or count != actual_count:
            return None, (
                "boundary detail evidence is incomplete: per-file speaker_change_count does not match "
                f"reference boundary count for {case.file_id} ({count!r} != {actual_count})"
            )
    return details, None


def _scheme_is_rankable(result: Mapping[str, object], wer_by_file: Mapping[str, Mapping[str, object]]) -> tuple[bool, str]:
    if result.get("status") != "metrics_completed":
        return False, f"{result.get('scheme', 'scheme')} speaker metrics status is {result.get('status')}"
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        return False, "speaker diarization summary is missing"
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        return False, "speaker diarization summary metrics are missing"
    if _as_metric_number(metrics.get("der")) is None:
        return False, "speaker diarization summary DER must be a finite numeric value"
    if _as_metric_number(metrics.get("speaker_change_hit_rate")) is None:
        return False, "speaker diarization summary speaker_change_hit_rate must be a finite numeric value"
    counts = summary.get("counts")
    if not isinstance(counts, Mapping):
        return False, "speaker diarization summary counts are missing"
    if counts.get("evaluated") != len(TEST_CASES) or counts.get("errors") != 0:
        return False, "not all fixed audio cases were successfully evaluated"
    per_file = result.get("per_file")
    files = per_file.get("files") if isinstance(per_file, Mapping) else None
    if not isinstance(files, list):
        return False, "speaker diarization per-file results are missing"
    evaluated_file_ids: list[str] = []
    for item in files:
        if not isinstance(item, Mapping) or not isinstance(item.get("file_id"), str) or item.get("error") is not None:
            return False, "speaker diarization per-file results include an invalid or failed file"
        evaluated_file_ids.append(item["file_id"])
    expected_file_ids = [case.file_id for case in TEST_CASES]
    if len(evaluated_file_ids) != len(expected_file_ids) or set(evaluated_file_ids) != set(expected_file_ids):
        return False, "speaker diarization did not evaluate exactly the fixed audio file IDs"
    missing_wer = [file_id for file_id, payload in wer_by_file.items() if payload.get("status") != "completed"]
    if missing_wer:
        return False, f"WER is unavailable for {', '.join(missing_wer)}"
    validation = result.get("boundary_details_validation")
    if not isinstance(validation, Mapping) or validation.get("available") is not True:
        reason = validation.get("reason") if isinstance(validation, Mapping) else None
        return False, str(reason or "boundary detail evidence is incomplete")
    return True, "all fixed audio cases have speaker metrics, valid WER, and complete boundary evidence"


def write_comparison(
    output_root: Path,
    scheme_results: Mapping[str, Mapping[str, object]],
    labels_dir: Path,
) -> dict[str, Any]:
    """Write machine-readable aggregate comparison and auditable boundary differences."""
    output_root = Path(output_root)
    comparison_dir = output_root / "05_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    normalized: dict[str, dict[str, Any]] = {}
    rank_inputs: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for scheme in SCHEMES:
        result = dict(scheme_results.get(scheme, {"scheme": scheme, "status": "metrics_missing"}))
        result.setdefault("scheme", scheme)
        result["wer_by_file"] = _load_wer_by_file(output_root, scheme)
        normalized[scheme] = result

    try:
        label_texts = _reference_boundary_texts_for_validation(Path(labels_dir))
        expected_keys = set(label_texts)
        label_issue: str | None = None
    except (OSError, ValueError) as error:
        label_texts = {}
        expected_keys = set()
        label_issue = f"boundary detail evidence is incomplete: reference labels are invalid: {error}"

    details_by_scheme: dict[str, dict[tuple[str, int, int], dict[str, object]]] = {}
    boundary_issues: dict[str, str | None] = {}
    for scheme in SCHEMES:
        path_issue = _boundary_details_issue(normalized[scheme])
        if path_issue is not None:
            details, issue = None, f"boundary detail evidence is incomplete: {path_issue}"
        elif label_issue is not None:
            details, issue = None, label_issue
        else:
            details, issue = _validate_boundary_details(normalized[scheme], expected_keys)
        if details is not None:
            details_by_scheme[scheme] = details
        boundary_issues[scheme] = issue

    if not any(boundary_issues.values()):
        baseline_keys = set(details_by_scheme["baseline"])
        streaming_keys = set(details_by_scheme["streaming"])
        if baseline_keys != streaming_keys:
            issue = "boundary detail evidence is incomplete: baseline and streaming reference boundary keys differ"
            boundary_issues = {scheme: issue for scheme in SCHEMES}

    for scheme in SCHEMES:
        result = normalized[scheme]
        issue = boundary_issues[scheme]
        result["boundary_details_validation"] = {"available": issue is None, "reason": issue}
        rankable, reason = _scheme_is_rankable(result, result["wer_by_file"])
        result["ranking_eligible"] = rankable
        result["ranking_reason"] = reason
        if result.get("status") != "metrics_completed":
            failures.append(f"{scheme}: {result.get('status')}")
        component_error = result.get("der_components_error")
        if isinstance(component_error, str) and component_error:
            failures.append(f"{scheme}: {component_error}")
        if issue is not None:
            failures.append(f"{scheme}: {issue}")
        summary = result.get("summary")
        if isinstance(summary, Mapping) and isinstance(summary.get("metrics"), Mapping):
            metric_values = summary["metrics"]
            rank_inputs[scheme] = {
                "hit_rate": metric_values.get("speaker_change_hit_rate"),
                "der": metric_values.get("der"),
            }

    boundary_differences_available = not any(boundary_issues.values())
    if boundary_differences_available:
        boundary_rows = _merge_boundary_detail_maps(
            details_by_scheme["baseline"], details_by_scheme["streaming"], label_texts,
        )
    else:
        boundary_rows = []
    boundary_path = comparison_dir / "boundary_differences.csv"
    with boundary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=BOUNDARY_DIFFERENCE_FIELDS)
        writer.writeheader()
        writer.writerows(boundary_rows)

    eligibility = [normalized[scheme]["ranking_eligible"] for scheme in SCHEMES]
    ranking_eligible = all(eligibility)
    winner = select_winner(rank_inputs) if ranking_eligible else None
    reasons = [f"{scheme}: {normalized[scheme]['ranking_reason']}" for scheme in SCHEMES]
    if not ranking_eligible:
        ranking_reason = "；".join(reasons)
    elif winner is None:
        ranking_reason = "命中率差值小于 0.01，且 DER 无差异"
    else:
        ranking_reason = "按断句命中率优先、DER 次级规则得出"
    comparison: dict[str, Any] = {
        "schema_version": 1,
        "metric_parameters": dict(METRIC_PARAMETERS),
        "schemes": normalized,
        "boundary_differences": boundary_rows,
        "boundary_differences_path": str(boundary_path),
        "boundary_differences_available": boundary_differences_available,
        "boundary_differences_error": "；".join(
            f"{scheme}: {issue}" for scheme, issue in boundary_issues.items() if issue is not None
        ) or None,
        "runtime_metadata": _find_runtime_metadata(output_root),
        "ranking_eligible": ranking_eligible,
        "winner": winner,
        "ranking_reason": ranking_reason,
        "failures": failures,
    }
    _write_json(comparison_dir / "comparison.json", comparison)
    return comparison


def _display_metric(value: object) -> str:
    numeric = _as_metric_number(value)
    return "未提供" if numeric is None else f"{numeric:.4f}"


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_report(
    output_root: Path,
    manifest: Mapping[str, object],
    comparison: Mapping[str, object],
    *,
    diart_summary: Mapping[str, object] | None = None,
) -> Path:
    """Write the Chinese Task-3 report without claiming a ranking on partial results."""
    output_root = Path(output_root)
    report_path = output_root / "报告.md"
    schemes = comparison.get("schemes") if isinstance(comparison.get("schemes"), Mapping) else {}
    lines = [
        "# sherpa-onnx pyannote segmentation 断句优化测试报告",
        "",
        "## 评测范围与公平性",
        "- 方式一：`offline-long-audio-pipeline-asr-speaker.py`。",
        "- 方式二：`offline-long-audio-pipeline-asr-speaker-segmentation.py`（10 秒窗口、1 秒滑移融合；`--segmentation-chunk-ms 32`）。",
        "- 固定聚类参数：`--num-clusters -1`、`--cluster-threshold 0.6`；DER collar 与断句容差均为 500 ms。",
        "",
        "## 实际命令",
    ]
    commands = manifest.get("commands") if isinstance(manifest.get("commands"), Mapping) else {}
    lines.append("```json")
    lines.append(json.dumps(commands, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.extend(["", "## 模型路径与 SHA-256"])
    models = manifest.get("models") if isinstance(manifest.get("models"), Mapping) else {}
    for location in ("local", "remote"):
        inventory = models.get(location) if isinstance(models, Mapping) else None
        lines.append(f"### {location}")
        if not isinstance(inventory, Mapping):
            lines.append("未记录。")
            continue
        lines.extend(["| 角色 | 路径 | SHA-256 |", "| --- | --- | --- |"])
        for role, item in inventory.items():
            if isinstance(item, Mapping):
                lines.append(f"| {_markdown_cell(role)} | {_markdown_cell(item.get('directory', ''))} | {_markdown_cell(item.get('sha256', ''))} |")
    runtime_metadata = comparison.get("runtime_metadata") if isinstance(comparison.get("runtime_metadata"), Mapping) else {}
    lines.extend(["", "## 实际远程命令记录"])
    actual_commands: dict[str, object] = {}
    for scheme in SCHEMES:
        scheme_runtime = runtime_metadata.get(scheme) if isinstance(runtime_metadata, Mapping) else {}
        if not isinstance(scheme_runtime, Mapping):
            continue
        for file_id, run in scheme_runtime.items():
            metadata = run.get("metadata") if isinstance(run, Mapping) and isinstance(run.get("metadata"), Mapping) else {}
            actual_argv = metadata.get("argv", metadata.get("command")) if isinstance(metadata, Mapping) else None
            if actual_argv is not None:
                actual_commands.setdefault(scheme, {})[file_id] = actual_argv
    if actual_commands:
        lines.append("以下 argv/command 来自远端回收的 `run_metadata.json`。")
        lines.append("```json")
        lines.append(json.dumps(actual_commands, ensure_ascii=False, indent=2))
        lines.append("```")
    else:
        lines.append("尚未回收实际远程命令；上方仅为受控命令模板，不能视为已执行记录。")
    lines.extend(["", "## 方案汇总指标"])
    lines.append("| 方案 | 汇总 DER | 汇总 speaker_change_hit_rate |")
    lines.append("| --- | --- | --- |")
    for scheme in SCHEMES:
        result = schemes.get(scheme) if isinstance(schemes, Mapping) else None
        summary = result.get("summary") if isinstance(result, Mapping) and isinstance(result.get("summary"), Mapping) else {}
        aggregate_metrics = summary.get("metrics") if isinstance(summary, Mapping) and isinstance(summary.get("metrics"), Mapping) else {}
        lines.append(
            f"| {scheme} | {_display_metric(aggregate_metrics.get('der'))} | "
            f"{_display_metric(aggregate_metrics.get('speaker_change_hit_rate'))} |"
        )
    lines.extend(["", "## 每条音频指标"])
    lines.append("| 方案 | 音频 | WER | DER | miss | false_alarm | confusion | 命中率 | 音频时长 | 运行耗时 | RTF | segment 数 | speaker 分布 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for scheme in SCHEMES:
        result = schemes.get(scheme) if isinstance(schemes, Mapping) else None
        result = result if isinstance(result, Mapping) else {}
        summary = result.get("summary") if isinstance(result.get("summary"), Mapping) else {}
        aggregate_metrics = summary.get("metrics") if isinstance(summary, Mapping) and isinstance(summary.get("metrics"), Mapping) else {}
        per_file_payload = result.get("per_file") if isinstance(result.get("per_file"), Mapping) else {}
        public_files = per_file_payload.get("files") if isinstance(per_file_payload, Mapping) else []
        if not isinstance(public_files, list):
            public_files = []
        metrics_by_file = {
            item.get("file_id"): item.get("metrics")
            for item in public_files
            if isinstance(item, Mapping) and isinstance(item.get("file_id"), str) and isinstance(item.get("metrics"), Mapping)
        }
        components_by_file = result.get("der_components_by_file") if isinstance(result.get("der_components_by_file"), Mapping) else {}
        wer_by_file = result.get("wer_by_file") if isinstance(result.get("wer_by_file"), Mapping) else {}
        runtime_by_file = runtime_metadata.get(scheme) if isinstance(runtime_metadata, Mapping) else {}
        for case in TEST_CASES:
            metrics = metrics_by_file.get(case.file_id, {})
            components = components_by_file.get(case.file_id, {}) if isinstance(components_by_file, Mapping) else {}
            wer = wer_by_file.get(case.file_id) if isinstance(wer_by_file, Mapping) else {}
            run = runtime_by_file.get(case.file_id) if isinstance(runtime_by_file, Mapping) else {}
            metadata = run.get("metadata") if isinstance(run, Mapping) and isinstance(run.get("metadata"), Mapping) else {}
            duration = metadata.get("audio_duration_seconds", metadata.get("audio_duration_s", "未记录"))
            elapsed = metadata.get("elapsed_seconds", metadata.get("elapsed_s", metadata.get("run_seconds", "未记录")))
            rtf = metadata.get("rtf", "未记录")
            segment_count = metadata.get("segment_count", "未记录")
            speaker_distribution = metadata.get("speaker_distribution", "未记录")
            lines.append(
                f"| {scheme} | {case.file_id} | {_display_metric(wer.get('wer') if isinstance(wer, Mapping) else None)} | "
                f"{_display_metric(metrics.get('der'))} | {_display_metric(components.get('miss'))} | "
                f"{_display_metric(components.get('false_alarm'))} | {_display_metric(components.get('confusion'))} | "
                f"{_display_metric(metrics.get('speaker_change_hit_rate'))} | {_markdown_cell(duration)} | "
                f"{_markdown_cell(elapsed)} | {_markdown_cell(rtf)} | {_markdown_cell(segment_count)} | {_markdown_cell(speaker_distribution)} |"
            )
    boundary_available = comparison.get("boundary_differences_available", True)
    if boundary_available is False:
        lines.extend(["", "## 边界差异（不可用）"])
        lines.append(f"- 未生成边界差异表：{comparison.get('boundary_differences_error', 'boundary detail CSV 缺失')}。")
    else:
        lines.extend(["", "## 边界差异（最多前 50 条）"])
        boundary_rows = comparison.get("boundary_differences") if isinstance(comparison.get("boundary_differences"), list) else []
        lines.extend(["| 文件 | 参考边界(ms) | baseline | streaming | 分类 | 前文本 | 后文本 |", "| --- | --- | --- | --- | --- | --- | --- |"])
        for row in boundary_rows[:50]:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                f"| {_markdown_cell(row.get('file_id', ''))} | {_markdown_cell(row.get('reference_boundary_ms', ''))} | "
                f"{_markdown_cell(row.get('baseline_match_ms', ''))} | {_markdown_cell(row.get('streaming_match_ms', ''))} | "
                f"{_markdown_cell(row.get('classification', ''))} | {_markdown_cell(row.get('before_text', ''))} | {_markdown_cell(row.get('after_text', ''))} |"
            )
    lines.extend(["", "## 结论"])
    if comparison.get("ranking_eligible"):
        winner = comparison.get("winner")
        if winner in SCHEMES:
            lines.append(f"本次数据集排名：{winner}。{comparison.get('ranking_reason', '')}")
        else:
            lines.append("无明显差异。命中率差值小于 0.01，且 DER 无差异。")
    else:
        lines.append(f"未形成有效总体排名：{comparison.get('ranking_reason', '两条音频未全部成功评价')}。")
    lines.extend(["", "## Diart 源码分析摘要"])
    if diart_summary is None:
        default_diart = output_root / "04_diart_source_analysis" / "diart_analysis.json"
        if default_diart.is_file():
            try:
                diart_summary = _read_json_mapping(default_diart, "Diart analysis")
            except ValueError as error:
                lines.append(f"Diart 分析文件不可读取：{error}")
        else:
            lines.append("尚未生成（Task 5 负责源码分析；本 Task 不运行 Diart 推理）。")
    if isinstance(diart_summary, Mapping):
        lines.append("```json")
        lines.append(json.dumps(diart_summary, ensure_ascii=False, indent=2))
        lines.append("```")
    lines.extend(["", "## 失败与限制"])
    failures = comparison.get("failures") if isinstance(comparison.get("failures"), list) else []
    if failures:
        lines.extend(f"- {item}" for item in failures)
    else:
        lines.append("- 未记录 Task 3 汇总阶段失败；远程构建/推理及 Diart 分析由后续任务执行。")
    for scheme in SCHEMES:
        result = schemes.get(scheme) if isinstance(schemes, Mapping) else None
        if isinstance(result, Mapping):
            for key in ("stdout_log", "stderr_log", "summary_path", "boundary_details_path"):
                if result.get(key):
                    lines.append(f"- {scheme} {key}: {result[key]}")
            if result.get("der_components_error"):
                lines.append(f"- {scheme} DER 分量限制：{result['der_components_error']}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def summarize_evaluation(
    output_root: Path,
    *,
    manifest_path: Path | None = None,
    metrics_source: Path = DEFAULT_METRICS_SOURCE,
    normalized_results_root: Path | None = None,
    labels_dir: Path | None = None,
) -> dict[str, Any]:
    """Run Task-3 local metrics/report aggregation only; it never starts remote work."""
    output_root = Path(output_root)
    manifest_path = output_root / "manifest.json" if manifest_path is None else Path(manifest_path)
    manifest = _read_json_mapping(manifest_path, "manifest")
    validate_manifest(manifest, verify_frozen_files=True)
    labels_dir = output_root / "01_input" / "labels" if labels_dir is None else Path(labels_dir)
    normalized_results_root = output_root / "03_normalized_results" if normalized_results_root is None else Path(normalized_results_root)
    tools_dir = output_root / "tools"
    scheme_results: dict[str, dict[str, Any]] = {}
    try:
        install_metrics_tool(Path(metrics_source), tools_dir)
    except (OSError, ValueError) as error:
        for scheme in SCHEMES:
            scheme_results[scheme] = {"scheme": scheme, "status": "metrics_failed", "error": str(error)}
        comparison = write_comparison(output_root, scheme_results, labels_dir)
        write_report(output_root, manifest, comparison)
        return comparison
    for scheme in SCHEMES:
        scheme_results[scheme] = run_speaker_metrics(
            scheme,
            tools_dir,
            normalized_results_root / scheme,
            labels_dir,
            output_root / "03_metrics" / scheme,
        )
    comparison = write_comparison(output_root, scheme_results, labels_dir)
    write_report(output_root, manifest, comparison)
    return comparison

def _legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("baseline", "streaming"), required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asr-dir", type=Path, required=True)
    parser.add_argument("--vad-dir", type=Path, required=True)
    parser.add_argument("--speaker-dir", type=Path, required=True)
    parser.add_argument("--segmentation-dir", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse legacy preview options or Task-2 subcommands."""
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0].startswith("--"):
        arguments = _legacy_parser().parse_args(values)
        arguments.command = "preview"
        return arguments
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    normalize = subparsers.add_parser("normalize-results")
    normalize.add_argument("--result-json", type=Path, required=True)
    normalize.add_argument("--file-id", required=True)
    normalize.add_argument("--output-dir", type=Path, required=True)
    wer = subparsers.add_parser("compute-wer")
    wer.add_argument("--label", type=Path, required=True)
    wer.add_argument("--predicted", type=Path, required=True)
    wer.add_argument("--scheme", required=True)
    wer.add_argument("--file-id", required=True)
    wer.add_argument("--output-root", type=Path, required=True)
    wer.add_argument("--language", default="zh")
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    summarize.add_argument("--manifest", type=Path)
    summarize.add_argument("--metrics-source", type=Path, default=DEFAULT_METRICS_SOURCE)
    summarize.add_argument("--normalized-results-root", type=Path)
    summarize.add_argument("--labels-dir", type=Path)
    return parser.parse_args(values)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a Task-1 preview or one of the Task-2 local data operations."""
    args = parse_args(argv)
    if args.command == "preview":
        paths = ModelPaths(args.asr_dir, args.vad_dir, args.speaker_dir, args.segmentation_dir)
        command = build_invocation(args.kind, args.audio, build_common_arguments(paths, args.output_root))
        print(json.dumps({"argv": command}, ensure_ascii=False))
        return 0
    if args.command == "prepare":
        print(json.dumps({"test_cases": prepare_inputs(args.output_root)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "validate-manifest":
        validate_manifest(
            json.loads(args.manifest.read_text(encoding="utf-8")),
            verify_frozen_files=True,
        )
        print(json.dumps({"manifest": str(args.manifest), "valid": True}, ensure_ascii=False))
        return 0
    if args.command == "normalize-results":
        result = json.loads(args.result_json.read_text(encoding="utf-8"))
        print(json.dumps({"output": str(write_metrics_asr_file(args.output_dir, args.file_id, result))}, ensure_ascii=False))
        return 0
    if args.command == "compute-wer":
        print(json.dumps(compute_wer(args.label, args.predicted, language=args.language, output_root=args.output_root, scheme=args.scheme, file_id=args.file_id), ensure_ascii=False, indent=2))
        return 0
    if args.command == "summarize":
        comparison = summarize_evaluation(
            args.output_root,
            manifest_path=args.manifest,
            metrics_source=args.metrics_source,
            normalized_results_root=args.normalized_results_root,
            labels_dir=args.labels_dir,
        )
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
