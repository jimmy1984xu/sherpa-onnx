"""Reproducible pyannote segmentation-punctuation evaluation helpers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
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


def _cross_platform_basename(path: str) -> str:
    """Return a path basename while recognizing both POSIX and Windows separators."""
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def validate_variant_fairness(baseline_argv: Sequence[str], streaming_argv: Sequence[str]) -> None:
    """Verify only approved variant-specific differences, including official entrypoint basenames."""
    for variant, argv in (("baseline", baseline_argv), ("streaming", streaming_argv)):
        expected_filename = ENTRY_SCRIPT_FILENAMES[variant]
        actual_script = argv[1] if len(argv) >= 2 else None
        if not isinstance(actual_script, str) or _cross_platform_basename(actual_script) != expected_filename:
            raise ValueError(
                f"{variant} entry script must have basename {expected_filename!r}, got {actual_script!r}"
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
    """Inventory explicit model directories with their content SHA-256 values."""
    return {
        role: {"directory": str(getattr(paths, f"{role}_dir")), "sha256": sha256_path(getattr(paths, f"{role}_dir"))}
        for role in MODEL_ROLES
    }


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
    subparsers.add_parser("summarize")
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
        print(json.dumps({"status": "deferred", "reason": "Task 3 report, DER, and boundary analysis are not implemented in Task 2."}, ensure_ascii=False))
        return 0
    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
