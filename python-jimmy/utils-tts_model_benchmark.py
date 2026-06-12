#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark streaming-capable sherpa-onnx TTS models.

This script downloads TTS models from the sherpa-onnx tts-models release,
generates one test sentence per model with callback streaming, saves WAV files,
and writes a TSV performance summary.

Examples:
python ./python-jimmy/utils-tts_model_benchmark.py ^
  --languages zh,en,es ^
  --model-dir D:/models/tts ^
  --audio-dir D:/tmp/tts-audio

python ./python-jimmy/utils-tts_model_benchmark.py ^
  --models vits-piper-en_US-amy-low,sherpa-onnx-pocket-tts-int8-2026-01-26 ^
  --model-dir D:/models/tts ^
  --audio-dir D:/tmp/tts-audio

python ./python-jimmy/utils-tts_model_benchmark.py ^
  --languages zh,en ^
  --model-dir D:/models/tts

python ./python-jimmy/utils-tts_model_benchmark.py ^
  --languages zh ^
  --model-dir D:/models/tts ^
  --audio-dir D:/tmp/tts-audio ^
  --reference-text "Reference text for ZipVoice fallback"

Notes:
- Streaming means sherpa_onnx.OfflineTts.generate() is called with a callback.
- The callback returns 1 to continue generation.
- Many models call back after each sentence, so default texts use short sentences.
- PocketTTS uses reference audio from the downloaded model directory.
- ZipVoice also needs reference text; the script auto-detects it when possible.
- fp32 and fp16 models are skipped; int8 models are preferred.
- Matcha models can use a specified or auto-downloaded Vocos vocoder.
- If --audio-dir is omitted, the script only downloads and extracts models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import numpy as np


RELEASE_API_URL = "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/tts-models"
RELEASE_DOWNLOAD_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
VOCODER_DOWNLOAD_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models"
ZIPVOICE_DEFAULT_VOCODER = "vocos_24khz.onnx"
VOCODER_NAMES = (
    "vocos_24khz.onnx",
    "vocos-22khz-univ.onnx",
    "vocos-16khz-univ.onnx",
)
MMS_LANG_MAP = {
    "eng": "en",
    "spa": "es",
}

DEFAULT_TEXT_BY_LANG = {
    "zh": "开始测试。今天我们一起测试一下TTS模型的性能。确认是否满足速度要求。",
    "en": "Start test. Today we test the first playable audio chunk latency. Finally we measure total generation speed.",
    "es": "Inicio de prueba. Hoy medimos la latencia del primer fragmento reproducible. Finalmente medimos la velocidad total.",
}

FALLBACK_MODELS = {
    "zh": [
        "vits-piper-zh_CN-huayan-medium",
        "vits-icefall-zh-aishell3",
        "vits-zh-aishell3",
        "vits-melo-tts-zh_en",
        "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia",
    ],
    "en": [
        "vits-piper-en_US-amy-low",
        "vits-piper-en_US-amy-medium",
        "vits-piper-en_US-lessac-medium",
        "vits-piper-en_US-ryan-medium",
        "vits-coqui-en-ljspeech",
        "vits-coqui-en-vctk",
        "vits-mms-eng",
        "vits-vctk",
        "kitten-nano-en-v0_8-fp32",
        "kitten-nano-en-v0_8-int8",
        "sherpa-onnx-pocket-tts-int8-2026-01-26",
        "sherpa-onnx-supertonic-3-tts-int8-2026-05-11",
        "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia",
    ],
    "es": [
        "vits-coqui-es-css10",
        "vits-mms-spa",
        "vits-piper-es-glados-medium",
        "vits-piper-es_ES-carlfm-x_low",
        "vits-piper-es_ES-davefx-medium",
        "vits-piper-es_ES-sharvard-medium",
        "vits-piper-es_MX-ald-medium",
        "vits-piper-es_MX-claude-high",
    ],
}

REFERENCE_AUDIO_NAMES = (
    "reference.wav",
    "prompt.wav",
    "test.wav",
    "sample.wav",
    "bria.wav",
    "leijun-1.wav",
)
REFERENCE_TEXT_NAMES = (
    "reference.txt",
    "prompt.txt",
    "reference_text.txt",
    "prompt_text.txt",
    "test.txt",
)
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
SUMMARY_FIELDNAMES = [
    "model",
    "language",
    "family",
    "text_id",
    "text",
    "first_chunk_latency_ms",
    "total_generate_ms",
    "audio_duration_ms",
    "rtf",
    "callback_count",
    "first_chunk_samples",
    "first_chunk_audio_ms",
    "stream",
    "speed",
    "voice",
    "voice_type",
    "output_wav",
    "status",
    "reason",
]
DOWNLOAD_FIELDNAMES = [
    "model",
    "language",
    "asset_name",
    "model_dir",
    "status",
    "reason",
]
@dataclass(frozen=True)
class CandidateModel:
    name: str
    lang: str
    asset_name: str


@dataclass(frozen=True)
class ModelConfigResult:
    family: str
    config: object
    gen_config: object
    text: str


@dataclass(frozen=True)
class ExcelColumn:
    header: str
    key: str
    width: int


EXCEL_COLUMNS = [
    ExcelColumn("model", "model", 42),
    ExcelColumn("lang", "language", 8),
    ExcelColumn("first_ms", "first_chunk_latency_ms", 12),
    ExcelColumn("rtf", "rtf", 8),
    ExcelColumn("status", "status", 10),
    ExcelColumn("reason", "reason", 36),
    ExcelColumn("wav", "output_wav", 54),
    ExcelColumn("text", "text", 42),
]


def parse_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def sanitize_filename_part(value: str) -> str:
    sanitized = SAFE_NAME_RE.sub("_", value.strip()).strip("._")
    return sanitized or "unknown"


def infer_capabilities(model_name: str, family: str = "") -> dict[str, str]:
    lowered = model_name.lower()
    family = family.lower()

    if "zipvoice" in lowered:
        return {"speed": "yes", "voice": "yes", "voice_type": "reference_audio"}
    if "pocket-tts" in lowered:
        return {"speed": "yes", "voice": "yes", "voice_type": "reference_audio"}
    if "supertonic" in lowered:
        return {"speed": "yes", "voice": "yes", "voice_type": "voice_style"}
    if "kokoro" in lowered or family == "kokoro":
        return {"speed": "yes", "voice": "yes", "voice_type": "voice_embedding"}
    if "kitten" in lowered or family == "kitten":
        return {"speed": "yes", "voice": "yes", "voice_type": "voice_embedding"}
    if "vctk" in lowered:
        return {"speed": "yes", "voice": "yes", "voice_type": "sid"}
    if family in {"vits", "matcha"} or lowered.startswith(("vits-", "matcha-")):
        return {"speed": "yes", "voice": "no", "voice_type": "single"}

    return {"speed": "unknown", "voice": "unknown", "voice_type": "unknown"}


def resolve_effective_lang(candidate: CandidateModel, args: argparse.Namespace) -> str:
    if candidate.lang != "multi":
        return candidate.lang

    user_languages = [item.lower() for item in parse_csv_arg(getattr(args, "languages", None))]
    if len(user_languages) == 1:
        return user_languages[0]

    return candidate.lang


def infer_lang_from_name(name: str) -> str:
    lowered = name.lower()
    if "supertonic" in lowered or "multi-lang" in lowered or "multilang" in lowered:
        return "multi"
    if "pocket-tts" in lowered:
        return "en"
    mms_match = re.search(r"(?:^|[-_])vits[-_]mms[-_]([a-z]{3})(?:$|[-_.])", lowered)
    if mms_match is not None:
        return MMS_LANG_MAP.get(mms_match.group(1), "unknown")
    if "-zh" in lowered or "_zh" in lowered or "zh_" in lowered or "zh-" in lowered:
        return "zh"
    if "-en" in lowered or "_en" in lowered or "en_" in lowered or "en-" in lowered:
        return "en"
    if "-es" in lowered or "_es" in lowered or "es_" in lowered or "es-" in lowered:
        return "es"
    return "unknown"


def model_matches_lang(name: str, lang: str) -> bool:
    lowered = name.lower()
    lang = lang.lower()
    if "supertonic" in lowered or "multi-lang" in lowered or "multilang" in lowered:
        return True
    if "pocket-tts" in lowered:
        return lang == "en"
    mms_lang = infer_lang_from_name(name)
    if lowered.startswith(("vits-mms-", "vits_mms_")):
        return mms_lang == lang
    patterns = [
        f"-{lang}-",
        f"-{lang}_",
        f"_{lang}_",
        f"_{lang}-",
        f"-{lang}.",
    ]
    return any(pattern in lowered for pattern in patterns)


def is_allowed_precision_model(name: str) -> bool:
    parts = re.split(r"[-_.]+", name.lower())
    if "int8" in parts:
        return True
    return not any(part.startswith("int") or part.startswith("fp") for part in parts)


def is_allowed_model_family(name: str) -> bool:
    return True


def is_allowed_candidate_model(name: str) -> bool:
    return is_allowed_precision_model(name) and is_allowed_model_family(name)


def precision_rank(name: str) -> int:
    parts = re.split(r"[-_.]+", name.lower())
    if "int8" in parts:
        return 0
    if not any(part.startswith("int") or part.startswith("fp") for part in parts):
        return 1
    return 2


def version_tuple(name: str) -> tuple[int, ...]:
    date_match = re.search(r"(20\d{2})-(\d{2})-(\d{2})(?:$|[-_])", name.lower())
    if date_match is not None:
        return tuple(int(part) for part in date_match.groups())
    match = re.search(r"(?:^|[-_])v(\d+(?:[._]\d+)*)", name.lower())
    if match is None:
        return (0,)
    return tuple(int(part) for part in re.split(r"[._]", match.group(1)))


def strip_precision_and_version(parts: list[str]) -> list[str]:
    return [
        part
        for part in parts
        if part.lower() not in {"int8", "fp16", "fp32"}
        and not re.fullmatch(r"v\d+(?:[._]\d+)*", part.lower())
    ]


def strip_precision_suffix(parts: list[str]) -> list[str]:
    if parts and parts[-1].lower() in {"int8", "fp16", "fp32"}:
        return parts[:-1]
    return parts


def read_reference_text_file(path: Path | None) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8").strip()


def resolve_test_text(args: argparse.Namespace, lang: str) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    if args.text:
        return args.text
    return DEFAULT_TEXT_BY_LANG.get(lang, DEFAULT_TEXT_BY_LANG["en"])


def fetch_release_assets(timeout: int) -> list[str]:
    request = urllib.request.Request(
        RELEASE_API_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "sherpa-onnx-tts-benchmark",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    assets_url = str(payload.get("assets_url", "")).split("{", 1)[0]
    if not assets_url:
        assets = payload.get("assets", [])
        return [asset["name"] for asset in assets if isinstance(asset, dict) and asset.get("name")]

    names: list[str] = []
    page = 1
    while True:
        page_url = f"{assets_url}?per_page=100&page={page}"
        page_request = urllib.request.Request(
            page_url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "sherpa-onnx-tts-benchmark",
            },
        )
        with urllib.request.urlopen(page_request, timeout=timeout) as response:
            page_assets = json.loads(response.read().decode("utf-8"))
        if not page_assets:
            break
        names.extend(
            asset["name"]
            for asset in page_assets
            if isinstance(asset, dict) and asset.get("name")
        )
        if len(page_assets) < 100:
            break
        page += 1
    return names


def asset_to_model_name(asset_name: str) -> str:
    for suffix in (".tar.bz2", ".tar.gz", ".tgz", ".zip", ".onnx"):
        if asset_name.endswith(suffix):
            return asset_name[: -len(suffix)]
    return asset_name


def build_candidates_from_assets(assets: Iterable[str], languages: list[str]) -> list[CandidateModel]:
    candidates: list[CandidateModel] = []
    for asset_name in assets:
        if not asset_name.endswith((".tar.bz2", ".tar.gz", ".tgz")):
            continue
        model_name = asset_to_model_name(asset_name)
        if not is_allowed_candidate_model(model_name):
            continue
        for lang in languages:
            if model_matches_lang(model_name, lang):
                candidates.append(CandidateModel(model_name, lang, asset_name))
                break
    return candidates


def build_fallback_candidates(languages: list[str]) -> list[CandidateModel]:
    candidates: list[CandidateModel] = []
    seen: set[str] = set()
    for lang in languages:
        for model_name in FALLBACK_MODELS.get(lang, []):
            if model_name in seen:
                continue
            if not is_allowed_candidate_model(model_name):
                continue
            seen.add(model_name)
            candidates.append(CandidateModel(model_name, lang, f"{model_name}.tar.bz2"))
    return candidates


def voice_variant_group_key(candidate: CandidateModel) -> str:
    name = candidate.name
    lowered = name.lower()
    lang = candidate.lang

    if lowered.startswith("vits-zh-hf-"):
        return f"{lang}:vits-zh-hf"

    if lowered.startswith("vits-cantonese-hf-"):
        return f"{lang}:vits-cantonese-hf"

    if lowered.startswith("vits-piper-"):
        parts = strip_precision_suffix(name.split("-"))
        locale = parts[2] if len(parts) > 2 else lang
        quality = parts[-1] if len(parts) > 3 else "default"
        return f"{lang}:vits-piper:{locale}:{quality}"

    if lowered.startswith("vits-coqui-"):
        parts = name.split("-")
        model_lang = parts[2] if len(parts) > 2 else lang
        return f"{lang}:vits-coqui:{model_lang}"

    if lowered.startswith("vits-mms-"):
        return f"{lang}:vits-mms:{infer_lang_from_name(name)}"

    if lowered.startswith("vits-mimic3-"):
        parts = strip_precision_suffix(name.split("-"))
        model_lang = parts[2] if len(parts) > 2 else lang
        quality = parts[-1] if len(parts) > 3 else "default"
        return f"{lang}:vits-mimic3:{model_lang}:{quality}"

    if lowered.startswith("kitten-"):
        return f"{lang}:kitten"

    if lowered.startswith("kokoro-"):
        return f"{lang}:{'-'.join(strip_precision_and_version(name.split('-')))}"

    if "supertonic" in lowered:
        return f"{lang}:supertonic"

    if "pocket-tts" in lowered:
        return f"{lang}:pocket"

    if "zipvoice" in lowered:
        return f"{lang}:zipvoice"

    return f"{lang}:{lowered}"


def dedupe_voice_variants(candidates: list[CandidateModel]) -> list[CandidateModel]:
    selected: dict[str, CandidateModel] = {}
    for candidate in sorted(
        candidates,
        key=lambda item: (precision_rank(item.name), tuple(-part for part in version_tuple(item.name)), item.name.lower()),
    ):
        key = voice_variant_group_key(candidate)
        selected.setdefault(key, candidate)
    return list(selected.values())


def resolve_candidates(args: argparse.Namespace) -> list[CandidateModel]:
    model_names = parse_csv_arg(args.models)
    languages = [item.lower() for item in parse_csv_arg(args.languages)]

    if model_names:
        candidates = [
            CandidateModel(name, infer_lang_from_name(name), f"{name}.tar.bz2")
            for name in model_names
            if is_allowed_candidate_model(name)
        ]
        if not candidates:
            raise ValueError("No usable models after filtering unsupported models")
        return candidates

    if not languages:
        raise ValueError("Please specify --languages or --models")

    try:
        assets = fetch_release_assets(args.http_timeout)
        candidates = build_candidates_from_assets(assets, languages)
        if candidates:
            return candidates if args.keep_voice_variants else dedupe_voice_variants(candidates)
        print("No matching assets found from GitHub API, using fallback model list.", file=sys.stderr)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
        print(f"Failed to query GitHub API, using fallback model list: {exc}", file=sys.stderr)

    candidates = build_fallback_candidates(languages)
    if not candidates:
        raise ValueError(f"No fallback models for languages: {','.join(languages)}")
    return candidates if args.keep_voice_variants else dedupe_voice_variants(candidates)


def download_file(url: str, dst: Path, timeout: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "sherpa-onnx-tts-benchmark"})
    with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(dst)


def safe_extract_tar(archive: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    base = dst.resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (dst / member.name).resolve()
            if base != target and base not in target.parents:
                raise ValueError(f"Unsafe path in archive: {member.name}")
        tar.extractall(dst)


def build_model_dir_index(model_dir: Path) -> dict[str, Path]:
    if not model_dir.is_dir():
        return {}
    return {path.name: path for path in model_dir.iterdir() if path.is_dir()}


def ensure_model_downloaded(
    candidate: CandidateModel,
    model_dir: Path,
    timeout: int,
    keep_archive: bool,
    model_dir_index: dict[str, Path] | None = None,
) -> Path:
    if model_dir_index is not None and candidate.name in model_dir_index:
        return model_dir_index[candidate.name]

    target_dir = model_dir / candidate.name
    if target_dir.is_dir():
        if model_dir_index is not None:
            model_dir_index[candidate.name] = target_dir
        return target_dir

    archive = model_dir / candidate.asset_name
    if not archive.is_file():
        url = f"{RELEASE_DOWNLOAD_BASE}/{candidate.asset_name}"
        print(f"Downloading {url}", file=sys.stderr)
        download_file(url, archive, timeout)

    print(f"Extracting {archive}", file=sys.stderr)
    safe_extract_tar(archive, model_dir)

    if not keep_archive:
        archive.unlink(missing_ok=True)

    if target_dir.is_dir():
        if model_dir_index is not None:
            model_dir_index[candidate.name] = target_dir
        return target_dir

    dirs = [
        path
        for name, path in (model_dir_index or build_model_dir_index(model_dir)).items()
        if name.startswith(candidate.name)
    ]
    if len(dirs) == 1:
        if model_dir_index is not None:
            model_dir_index[candidate.name] = dirs[0]
        return dirs[0]

    raise FileNotFoundError(f"Cannot find extracted model directory for {candidate.name}")


def find_file(root: Path, names: Iterable[str]) -> Path | None:
    wanted = {name.lower() for name in names}
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() in wanted:
            return path
    return None


def find_one_by_suffix(root: Path, suffix: str, contains: str = "") -> Path | None:
    contains = contains.lower()
    for path in root.rglob(f"*{suffix}"):
        if path.is_file() and (not contains or contains in path.name.lower()):
            return path
    return None


def find_all_by_suffix(root: Path, suffix: str) -> list[Path]:
    return [path for path in root.rglob(f"*{suffix}") if path.is_file()]


def get_required_file(root: Path, relative_or_name: str) -> Path:
    direct = root / relative_or_name
    if direct.is_file() or direct.is_dir():
        return direct
    by_name = find_file(root, [Path(relative_or_name).name])
    if by_name is None:
        raise FileNotFoundError(f"Cannot find {relative_or_name} under {root}")
    return by_name


def read_wav_float(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sample_width == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    return samples.astype(np.float32), sample_rate


def write_wav_float(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def load_sherpa_onnx():
    try:
        import sherpa_onnx  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Cannot import sherpa_onnx. Please run with the Python environment that has it installed.") from exc
    return sherpa_onnx


def make_base_model_config(sherpa_onnx, num_threads: int, provider: str):
    return sherpa_onnx.OfflineTtsModelConfig(
        num_threads=num_threads,
        provider=provider,
    )


def build_vits_config(sherpa_onnx, root: Path, args: argparse.Namespace):
    onnx_files = find_all_by_suffix(root, ".onnx")
    if not onnx_files:
        raise FileNotFoundError("No ONNX model file found")

    model = (
        find_file(root, ["model.onnx"])
        or find_one_by_suffix(root, ".onnx", "vits")
        or next((p for p in onnx_files if "vocoder" not in p.name.lower()), onnx_files[0])
    )
    tokens = get_required_file(root, "tokens.txt")
    lexicon = find_file(root, ["lexicon.txt"])
    data_dir = root / "espeak-ng-data"
    dict_dir = root / "dict"

    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(model),
        tokens=str(tokens),
        lexicon=str(lexicon) if lexicon else "",
        data_dir=str(data_dir) if data_dir.is_dir() else "",
        dict_dir=str(dict_dir) if dict_dir.is_dir() else "",
    )
    model_config = make_base_model_config(sherpa_onnx, args.num_threads, args.provider)
    model_config.vits = vits
    config = sherpa_onnx.OfflineTtsConfig(
        model=model_config,
        rule_fsts=",".join(str(p) for p in root.rglob("*.fst")),
        rule_fars=",".join(str(p) for p in root.rglob("*.far")),
        max_num_sentences=1,
    )
    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.speed = args.speed
    return config, gen_config


def build_matcha_config(sherpa_onnx, root: Path, args: argparse.Namespace):
    acoustic = (
        find_one_by_suffix(root, ".onnx", "acoustic")
        or find_one_by_suffix(root, ".onnx", "model")
        or get_required_file(root, "model.onnx")
    )
    vocoder = find_matcha_vocoder(root, args)
    tokens = get_required_file(root, "tokens.txt")
    lexicon = find_file(root, ["lexicon.txt"])
    data_dir = root / "espeak-ng-data"
    dict_dir = root / "dict"

    matcha = sherpa_onnx.OfflineTtsMatchaModelConfig(
        acoustic_model=str(acoustic),
        vocoder=str(vocoder),
        tokens=str(tokens),
        lexicon=str(lexicon) if lexicon else "",
        data_dir=str(data_dir) if data_dir.is_dir() else "",
        dict_dir=str(dict_dir) if dict_dir.is_dir() else "",
    )
    model_config = make_base_model_config(sherpa_onnx, args.num_threads, args.provider)
    model_config.matcha = matcha
    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.speed = args.speed
    return sherpa_onnx.OfflineTtsConfig(model=model_config, max_num_sentences=1), gen_config


def build_kokoro_config(sherpa_onnx, root: Path, args: argparse.Namespace, lang: str):
    model = find_file(root, ["model.onnx"]) or find_one_by_suffix(root, ".onnx", "model")
    if model is None:
        raise FileNotFoundError("Cannot find Kokoro model ONNX")
    voices = find_file(root, ["voices.bin"])
    if voices is None:
        raise FileNotFoundError("Cannot find voices.bin")
    tokens = get_required_file(root, "tokens.txt")
    data_dir = root / "espeak-ng-data"
    dict_dir = root / "dict"
    multilang_zh = (
        lang == "zh"
        and (root / "lexicon-us-en.txt").is_file()
        and (root / "lexicon-zh.txt").is_file()
        and (root / "phone-zh.fst").is_file()
        and (root / "date-zh.fst").is_file()
        and (root / "number-zh.fst").is_file()
    )

    if multilang_zh:
        lexicon = ",".join(
            str(get_required_file(root, name))
            for name in ("lexicon-us-en.txt", "lexicon-zh.txt")
        )
        rule_fsts = ",".join(
            str(get_required_file(root, name))
            for name in ("phone-zh.fst", "date-zh.fst", "number-zh.fst")
        )
        kokoro_lang = ""
    else:
        lexicon_path = find_file(root, ["lexicon.txt"])
        lexicon = str(lexicon_path) if lexicon_path else ""
        rule_fsts = ""
        kokoro_lang = "" if lang == "unknown" else lang

    kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
        model=str(model),
        voices=str(voices),
        tokens=str(tokens),
        lexicon=lexicon,
        data_dir=str(data_dir) if data_dir.is_dir() else "",
        dict_dir=str(dict_dir) if dict_dir.is_dir() else "",
        lang=kokoro_lang,
    )
    model_config = make_base_model_config(sherpa_onnx, args.num_threads, args.provider)
    model_config.kokoro = kokoro
    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.sid = args.sid
    gen_config.speed = args.speed
    return sherpa_onnx.OfflineTtsConfig(model=model_config, rule_fsts=rule_fsts, max_num_sentences=1), gen_config


def build_kitten_config(sherpa_onnx, root: Path, args: argparse.Namespace):
    model = find_file(root, ["model.onnx"]) or find_one_by_suffix(root, ".onnx", "model")
    if model is None:
        raise FileNotFoundError("Cannot find Kitten model ONNX")
    voices = find_file(root, ["voices.bin"])
    if voices is None:
        raise FileNotFoundError("Cannot find voices.bin")
    tokens = get_required_file(root, "tokens.txt")
    data_dir = root / "espeak-ng-data"

    kitten = sherpa_onnx.OfflineTtsKittenModelConfig(
        model=str(model),
        voices=str(voices),
        tokens=str(tokens),
        data_dir=str(data_dir) if data_dir.is_dir() else "",
    )
    model_config = make_base_model_config(sherpa_onnx, args.num_threads, args.provider)
    model_config.kitten = kitten
    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.sid = args.sid
    gen_config.speed = args.speed
    return sherpa_onnx.OfflineTtsConfig(model=model_config, max_num_sentences=1), gen_config


def build_pocket_config(sherpa_onnx, root: Path, args: argparse.Namespace):
    pocket = sherpa_onnx.OfflineTtsPocketModelConfig(
        lm_flow=str(get_required_file(root, "lm_flow.int8.onnx")),
        lm_main=str(get_required_file(root, "lm_main.int8.onnx")),
        encoder=str(get_required_file(root, "encoder.onnx")),
        decoder=str(get_required_file(root, "decoder.int8.onnx")),
        text_conditioner=str(get_required_file(root, "text_conditioner.onnx")),
        vocab_json=str(get_required_file(root, "vocab.json")),
        token_scores_json=str(get_required_file(root, "token_scores.json")),
    )
    reference_audio = find_file(root, REFERENCE_AUDIO_NAMES) or find_one_by_suffix(root, ".wav")
    if reference_audio is None:
        raise FileNotFoundError("Cannot find reference WAV for PocketTTS")
    samples, sample_rate = read_wav_float(reference_audio)

    model_config = make_base_model_config(sherpa_onnx, args.num_threads, args.provider)
    model_config.pocket = pocket
    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.speed = args.speed
    gen_config.num_steps = args.num_steps
    gen_config.reference_audio = samples.tolist()
    gen_config.reference_sample_rate = sample_rate
    return sherpa_onnx.OfflineTtsConfig(model=model_config, max_num_sentences=1), gen_config


def build_supertonic_config(sherpa_onnx, root: Path, args: argparse.Namespace, lang: str):
    supertonic = sherpa_onnx.OfflineTtsSupertonicModelConfig(
        duration_predictor=str(get_required_file(root, "duration_predictor.int8.onnx")),
        text_encoder=str(get_required_file(root, "text_encoder.int8.onnx")),
        vector_estimator=str(get_required_file(root, "vector_estimator.int8.onnx")),
        vocoder=str(get_required_file(root, "vocoder.int8.onnx")),
        tts_json=str(get_required_file(root, "tts.json")),
        unicode_indexer=str(get_required_file(root, "unicode_indexer.bin")),
        voice_style=str(get_required_file(root, "voice.bin")),
    )
    model_config = make_base_model_config(sherpa_onnx, args.num_threads, args.provider)
    model_config.supertonic = supertonic
    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.speed = args.speed
    gen_config.num_steps = args.num_steps
    if lang != "unknown":
        gen_config.extra["lang"] = lang
    return sherpa_onnx.OfflineTtsConfig(model=model_config, max_num_sentences=1), gen_config


def explicit_vocoder_path(args: argparse.Namespace) -> Path | None:
    value = args.vocoder or args.zipvoice_vocoder
    if value:
        path = Path(value)
        if path.is_file():
            return path
        raise FileNotFoundError(f"Vocoder does not exist: {path}")
    return None


def find_existing_vocoder(model_root: Path, model_dir: Path, names: Iterable[str]) -> Path | None:
    path = find_file(model_root, names)
    if path is not None:
        return path
    for name in names:
        path = model_dir / name
        if path.is_file():
            return path
    return None


def download_vocoder(args: argparse.Namespace, filename: str) -> Path:
    path = Path(args.model_dir) / filename
    if path.is_file():
        return path
    url = f"{VOCODER_DOWNLOAD_BASE}/{filename}"
    print(f"Downloading {url}", file=sys.stderr)
    download_file(url, path, args.http_timeout)
    return path


def default_matcha_vocoder_name(model_name: str) -> str:
    lowered = model_name.lower()
    if lowered == "matcha-icefall-zh-en":
        return "vocos-16khz-univ.onnx"
    return "vocos-22khz-univ.onnx"


def find_matcha_vocoder(model_root: Path, args: argparse.Namespace) -> Path:
    explicit = explicit_vocoder_path(args)
    if explicit is not None:
        return explicit
    default_name = default_matcha_vocoder_name(model_root.name)
    existing = find_existing_vocoder(model_root, Path(args.model_dir), [default_name])
    if existing is not None:
        return existing
    return download_vocoder(args, default_name)


def ensure_zipvoice_vocoder(args: argparse.Namespace) -> Path:
    explicit = explicit_vocoder_path(args)
    if explicit is not None:
        return explicit

    existing = find_existing_vocoder(Path(args.model_dir), Path(args.model_dir), [ZIPVOICE_DEFAULT_VOCODER])
    if existing is not None:
        return existing

    return download_vocoder(args, ZIPVOICE_DEFAULT_VOCODER)


def detect_reference_text(root: Path, fallback_text: str) -> str:
    text_file = find_file(root, REFERENCE_TEXT_NAMES)
    if text_file is not None:
        text = text_file.read_text(encoding="utf-8").strip()
        if text:
            return text

    for path in root.rglob("*.txt"):
        if path.name.lower() in {"tokens.txt", "lexicon.txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text and len(text) < 500:
            return text

    return fallback_text


def build_zipvoice_config(sherpa_onnx, root: Path, args: argparse.Namespace, fallback_reference_text: str):
    reference_audio = find_file(root, REFERENCE_AUDIO_NAMES) or find_one_by_suffix(root, ".wav")
    if reference_audio is None:
        raise FileNotFoundError("Cannot find reference WAV for ZipVoice")

    reference_text = detect_reference_text(root, fallback_reference_text)
    if not reference_text:
        raise FileNotFoundError("Cannot find reference text for ZipVoice")

    samples, sample_rate = read_wav_float(reference_audio)
    zipvoice = sherpa_onnx.OfflineTtsZipvoiceModelConfig(
        tokens=str(get_required_file(root, "tokens.txt")),
        encoder=str(get_required_file(root, "encoder.int8.onnx")),
        decoder=str(get_required_file(root, "decoder.int8.onnx")),
        vocoder=str(ensure_zipvoice_vocoder(args)),
        data_dir=str(root / "espeak-ng-data") if (root / "espeak-ng-data").is_dir() else "",
        lexicon=str(get_required_file(root, "lexicon.txt")),
    )
    model_config = make_base_model_config(sherpa_onnx, args.num_threads, args.provider)
    model_config.zipvoice = zipvoice
    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.speed = args.speed
    gen_config.num_steps = args.num_steps
    gen_config.reference_audio = samples.tolist()
    gen_config.reference_sample_rate = sample_rate
    gen_config.reference_text = reference_text
    return sherpa_onnx.OfflineTtsConfig(model=model_config, max_num_sentences=1), gen_config


def build_tts_config(
    sherpa_onnx,
    candidate: CandidateModel,
    root: Path,
    args: argparse.Namespace,
    fallback_reference_text: str,
    text: str,
) -> ModelConfigResult:
    name = candidate.name.lower()
    lang = resolve_effective_lang(candidate, args)

    if "pocket-tts" in name:
        config, gen_config = build_pocket_config(sherpa_onnx, root, args)
        return ModelConfigResult("pocket", config, gen_config, text)
    if "supertonic" in name:
        config, gen_config = build_supertonic_config(sherpa_onnx, root, args, lang)
        return ModelConfigResult("supertonic", config, gen_config, text)
    if "zipvoice" in name:
        config, gen_config = build_zipvoice_config(sherpa_onnx, root, args, fallback_reference_text)
        return ModelConfigResult("zipvoice", config, gen_config, text)
    if "matcha" in name:
        config, gen_config = build_matcha_config(sherpa_onnx, root, args)
        return ModelConfigResult("matcha", config, gen_config, text)
    if "kokoro" in name:
        config, gen_config = build_kokoro_config(sherpa_onnx, root, args, lang)
        return ModelConfigResult("kokoro", config, gen_config, text)
    if "kitten" in name:
        config, gen_config = build_kitten_config(sherpa_onnx, root, args)
        return ModelConfigResult("kitten", config, gen_config, text)

    config, gen_config = build_vits_config(sherpa_onnx, root, args)
    return ModelConfigResult("vits", config, gen_config, text)


def benchmark_one(
    sherpa_onnx,
    candidate: CandidateModel,
    model_root: Path,
    audio_dir: Path,
    args: argparse.Namespace,
    fallback_reference_text: str,
    text: str,
    text_id: int,
) -> dict[str, str]:
    capabilities = infer_capabilities(candidate.name)
    effective_lang = resolve_effective_lang(candidate, args)
    row = {
        "model": candidate.name,
        "language": effective_lang,
        "family": "",
        "text_id": str(text_id),
        "text": "",
        "first_chunk_latency_ms": "",
        "total_generate_ms": "",
        "audio_duration_ms": "",
        "rtf": "",
        "callback_count": "",
        "first_chunk_samples": "",
        "first_chunk_audio_ms": "",
        "stream": "",
        "speed": capabilities["speed"],
        "voice": capabilities["voice"],
        "voice_type": capabilities["voice_type"],
        "output_wav": "",
        "status": "ok",
        "reason": "",
    }

    try:
        load_start = time.perf_counter()
        config_result = build_tts_config(
            sherpa_onnx,
            candidate,
            model_root,
            args,
            fallback_reference_text,
            text,
        )
        row["family"] = config_result.family
        row["text"] = config_result.text
        row.update(infer_capabilities(candidate.name, config_result.family))

        if not config_result.config.validate():
            raise ValueError("sherpa_onnx OfflineTtsConfig validation failed")

        tts = sherpa_onnx.OfflineTts(config_result.config)
        load_end = time.perf_counter()
        load_ms = (load_end - load_start) * 1000.0
        print(
            f"  Model ready: family={config_result.family} load_ms={load_ms:.2f} model_dir={model_root}",
            file=sys.stderr,
        )

        def run_generate(use_callback: bool) -> tuple[object, float, float, int, int, float]:
            first_chunk_time: float | None = None
            callback_count = 0
            first_chunk_samples = 0
            start = time.perf_counter()

            def callback(samples, progress):
                nonlocal callback_count, first_chunk_samples, first_chunk_time
                callback_count += 1
                if first_chunk_time is None and len(samples) > 0:
                    first_chunk_time = time.perf_counter()
                    first_chunk_samples = len(samples)
                return 1

            if use_callback:
                audio = tts.generate(config_result.text, config_result.gen_config, callback)
            else:
                audio = tts.generate(config_result.text, config_result.gen_config)
            end = time.perf_counter()
            latency_ms = (first_chunk_time - start) * 1000.0 if first_chunk_time else math.nan
            total_s = end - start
            duration_s = len(audio.samples) / float(audio.sample_rate) if audio.sample_rate else 0.0
            return audio, total_s, latency_ms, callback_count, first_chunk_samples, duration_s

        warmup_audio, warmup_total_s, warmup_latency_ms, warmup_callback_count, warmup_first_chunk_samples, warmup_duration_s = run_generate(
            args.stream
        )
        if int(warmup_audio.sample_rate) <= 0:
            raise ValueError("Warm-up audio has invalid sample_rate")
        if len(warmup_audio.samples) == 0:
            raise ValueError("Warm-up generated audio is empty. Try a different sid or check model config.")
        warmup_rtf = warmup_total_s / warmup_duration_s if warmup_duration_s > 0 else math.inf
        print(
            "  Warm-up(discarded): "
            f"first_ms={(f'{warmup_latency_ms:.2f}' if not math.isnan(warmup_latency_ms) else 'NA')} "
            f"total_ms={warmup_total_s * 1000.0:.2f} "
            f"audio_ms={warmup_duration_s * 1000.0:.2f} "
            f"rtf={(f'{warmup_rtf:.4f}' if math.isfinite(warmup_rtf) else 'NA')} "
            f"cb={warmup_callback_count} "
            f"chunk_ms={(f'{warmup_first_chunk_samples * 1000.0 / float(warmup_audio.sample_rate):.2f}' if warmup_audio.sample_rate and warmup_first_chunk_samples else 'NA')}",
            file=sys.stderr,
        )

        audio, total_s, latency_ms, callback_count, first_chunk_samples, duration_s = run_generate(args.stream)

        if int(audio.sample_rate) <= 0:
            raise ValueError("Generated audio has invalid sample_rate")
        if len(audio.samples) == 0:
            raise ValueError("Generated audio is empty. Try a different sid or check model config.")

        rtf = total_s / duration_s if duration_s > 0 else math.inf
        out_wav = audio_dir / f"{sanitize_filename_part(candidate.name)}_text{text_id}.wav"
        write_wav_float(out_wav, np.asarray(audio.samples, dtype=np.float32), int(audio.sample_rate))
        print(
            "  Measured: "
            f"first_ms={(f'{latency_ms:.2f}' if not math.isnan(latency_ms) else 'NA')} "
            f"total_ms={total_s * 1000.0:.2f} "
            f"audio_ms={duration_s * 1000.0:.2f} "
            f"rtf={(f'{rtf:.4f}' if math.isfinite(rtf) else 'NA')} "
            f"cb={callback_count} "
            f"chunk_ms={(f'{first_chunk_samples * 1000.0 / float(audio.sample_rate):.2f}' if audio.sample_rate and first_chunk_samples else 'NA')}",
            file=sys.stderr,
        )

        row["first_chunk_latency_ms"] = f"{latency_ms:.2f}" if not math.isnan(latency_ms) else "NA"
        row["total_generate_ms"] = f"{total_s * 1000.0:.2f}"
        row["audio_duration_ms"] = f"{duration_s * 1000.0:.2f}"
        row["rtf"] = f"{rtf:.4f}" if math.isfinite(rtf) else "NA"
        row["callback_count"] = str(callback_count)
        row["first_chunk_samples"] = str(first_chunk_samples)
        row["first_chunk_audio_ms"] = (
            f"{first_chunk_samples * 1000.0 / float(audio.sample_rate):.2f}"
            if audio.sample_rate and first_chunk_samples
            else "NA"
        )
        row["stream"] = "yes" if args.stream else "no"
        row["output_wav"] = str(out_wav)
        return row
    except Exception as exc:
        row["status"] = "skipped"
        row["reason"] = str(exc)
        return row


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_download_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DOWNLOAD_FIELDNAMES, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def excel_cell_ref(row_idx: int, col_idx: int) -> str:
    letters = ""
    col = col_idx
    while col:
        col, remainder = divmod(col - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return f"{letters}{row_idx}"


def excel_cell_xml(row_idx: int, col_idx: int, value: str) -> str:
    ref = excel_cell_ref(row_idx, col_idx)
    value = "" if value is None else str(value)
    if value and re.fullmatch(r"-?\d+(\.\d+)?", value):
        return f'<c r="{ref}"><v>{value}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'


def write_excel_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_rows: list[str] = []
    header_cells = [
        excel_cell_xml(1, idx, column.header)
        for idx, column in enumerate(EXCEL_COLUMNS, start=1)
    ]
    sheet_rows.append(f'<row r="1">{"".join(header_cells)}</row>')

    for row_idx, row in enumerate(rows, start=2):
        cells = [
            excel_cell_xml(row_idx, col_idx, row.get(column.key, ""))
            for col_idx, column in enumerate(EXCEL_COLUMNS, start=1)
        ]
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    cols_xml = "".join(
        f'<col min="{idx}" max="{idx}" width="{column.width}" customWidth="1"/>'
        for idx, column in enumerate(EXCEL_COLUMNS, start=1)
    )
    dimension = f"A1:{excel_cell_ref(max(1, len(rows) + 1), len(EXCEL_COLUMNS))}"
    sheet_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="{dimension}"/>
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <cols>{cols_xml}</cols>
  <sheetData>{"".join(sheet_rows)}</sheetData>
</worksheet>'''
    workbook_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="tts_benchmark" sheetId="1" r:id="rId1"/></sheets>
</workbook>'''
    workbook_rels_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>'''
    root_rels_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''
    content_types_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>'''

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def write_candidate_list(path: Path, candidates: list[CandidateModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for candidate in candidates:
            f.write(f"{candidate.name}\t{candidate.lang}\t{candidate.asset_name}\n")


def read_candidate_list(path: Path) -> list[CandidateModel]:
    candidates: list[CandidateModel] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            candidates.append(CandidateModel(parts[0], parts[1], parts[2]))
    return candidates


def download_models_only(
    candidates: list[CandidateModel],
    model_dir: Path,
    timeout: int,
    keep_archive: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    model_dir_index = build_model_dir_index(model_dir)
    for candidate in candidates:
        row = {
            "model": candidate.name,
            "language": candidate.lang,
            "asset_name": candidate.asset_name,
            "model_dir": "",
            "status": "ok",
            "reason": "",
        }
        try:
            already_downloaded = candidate.name in model_dir_index
            if not already_downloaded:
                print(f"Downloading/checking {candidate.name}", file=sys.stderr)
            model_root = ensure_model_downloaded(candidate, model_dir, timeout, keep_archive, model_dir_index)
            row["model_dir"] = str(model_root)
            if already_downloaded:
                row["reason"] = "already exists"
        except Exception as exc:
            row["status"] = "failed"
            row["reason"] = str(exc)
        rows.append(row)
    return rows


def parse_float_for_sort(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def sort_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return rows


def sort_excel_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return rows


def order_rows_by_candidates(rows: list[dict[str, str]], candidates: list[CandidateModel]) -> list[dict[str, str]]:
    order_map = {candidate.name: index for index, candidate in enumerate(candidates)}
    return sorted(
        rows,
        key=lambda row: (
            order_map.get(row.get("model", ""), len(order_map)),
            int(row.get("text_id", "1") or "1"),
        ),
    )


def print_terminal_summary(rows: list[dict[str, str]]) -> None:
    columns = [
        ("model", "model"),
        ("lang", "language"),
        ("first_ms", "first_chunk_latency_ms"),
        ("rtf", "rtf"),
        ("status", "status"),
        ("reason", "reason"),
    ]
    widths: list[int] = []
    for header, key in columns:
        widths.append(
            max(
                len(header),
                *(len(str(row.get(key, ""))) for row in rows),
            )
        )

    def format_line(values: list[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    print()
    print("Terminal summary:")
    print(format_line([header for header, _ in columns]))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_line([str(row.get(key, "")) for _, key in columns]))


def output_scope_suffix(args: argparse.Namespace) -> str:
    languages = [sanitize_filename_part(item.lower()) for item in parse_csv_arg(args.languages)]
    if languages:
        return "_".join(languages)
    models = parse_csv_arg(args.models)
    if models:
        return "models"
    return "all"


def output_mode_suffix(args: argparse.Namespace) -> str:
    return "stream" if args.stream else "nonstream"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and benchmark streaming-capable sherpa-onnx TTS models."
    )
    parser.add_argument("--languages", help="Comma-separated language codes, e.g. zh,es,en")
    parser.add_argument("--models", help="Comma-separated model names. Overrides --languages auto selection.")
    parser.add_argument("--model-dir", required=True, help="Directory used to store downloaded models")
    parser.add_argument("--audio-dir", help="Directory used to store generated WAV files. Omit it for download-only mode.")
    parser.add_argument("--summary-file", help="TSV summary path. Default: <audio-dir>/tts_benchmark_<scope>.tsv")
    parser.add_argument("--excel-file", help="Excel summary path. Default: <audio-dir>/tts_benchmark_<scope>.xlsx")
    parser.add_argument("--candidate-file", help="Filtered candidate list path. Default: <model-dir>/tts_candidates.txt")
    parser.add_argument("--use-candidate-file", action="store_true", help="Read candidates from --candidate-file and skip GitHub API query")
    parser.add_argument("--text-file", help="Read one test text from a UTF-8 txt file for every model")
    parser.add_argument("--text", help="Override test text for every model")
    parser.add_argument("--reference-text", help="Fallback reference text for ZipVoice")
    parser.add_argument("--reference-text-file", help="Fallback reference text file for ZipVoice")
    parser.add_argument("--vocoder", help="Path to a Vocos vocoder for Matcha or ZipVoice")
    parser.add_argument("--zipvoice-vocoder", help="Compatibility alias for --vocoder")
    parser.add_argument("--provider", default="cpu", help="ONNX Runtime provider, default: cpu")
    parser.add_argument("--num-threads", type=int, default=1, help="Number of inference threads")
    parser.add_argument("--sid", type=int, default=0, help="Speaker ID for models that support sid, such as Kokoro and Kitten")
    parser.add_argument("--stream", action="store_true", help="Enable streaming callback mode. Default is non-streaming")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--num-steps", type=int, default=4, help="Flow matching steps for supported models")
    parser.add_argument("--http-timeout", type=int, default=60, help="Download/API timeout in seconds")
    parser.add_argument("--keep-archive", action="store_true", help="Keep downloaded tar archives")
    parser.add_argument(
        "--keep-voice-variants",
        action="store_true",
        help="Keep all voice variants in automatic language mode",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    model_dir = Path(args.model_dir)
    scope = output_scope_suffix(args)
    candidate_file = Path(args.candidate_file) if args.candidate_file else model_dir / "tts_candidates.txt"
    fallback_reference_text = args.reference_text or read_reference_text_file(
        Path(args.reference_text_file) if args.reference_text_file else None
    )

    model_dir.mkdir(parents=True, exist_ok=True)

    if args.use_candidate_file:
        candidates = read_candidate_list(candidate_file)
        if not candidates:
            raise SystemExit(f"error: no candidates in {candidate_file}")
        print(f"Read candidate list: {candidate_file}")
    else:
        try:
            candidates = resolve_candidates(args)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from exc
        write_candidate_list(candidate_file, candidates)
        print(f"Wrote candidate list: {candidate_file}")

    if not args.audio_dir:
        download_summary_file = model_dir / f"tts_download_{scope}.tsv"
        download_rows = download_models_only(
            candidates,
            model_dir,
            args.http_timeout,
            args.keep_archive,
        )
        write_download_summary(download_summary_file, download_rows)
        print(f"Wrote download summary: {download_summary_file}")
        return

    audio_dir = Path(args.audio_dir)
    mode_suffix = output_mode_suffix(args)
    summary_file = Path(args.summary_file) if args.summary_file else audio_dir / f"tts_benchmark_{scope}_{mode_suffix}.tsv"
    excel_file = Path(args.excel_file) if args.excel_file else audio_dir / f"tts_benchmark_{scope}_{mode_suffix}.xlsx"
    audio_dir.mkdir(parents=True, exist_ok=True)

    sherpa_onnx = load_sherpa_onnx()
    rows: list[dict[str, str]] = []
    model_dir_index = build_model_dir_index(model_dir)

    for candidate in candidates:
        print(f"Processing {candidate.name}", file=sys.stderr)
        first_text = resolve_test_text(args, resolve_effective_lang(candidate, args))
        try:
            model_root = ensure_model_downloaded(
                candidate,
                model_dir,
                args.http_timeout,
                args.keep_archive,
                model_dir_index,
            )
            row = benchmark_one(
                sherpa_onnx,
                candidate,
                model_root,
                audio_dir,
                args,
                fallback_reference_text,
                first_text,
                1,
            )
            rows.append(row)
        except Exception as exc:
            capabilities = infer_capabilities(candidate.name)
            row = {
                "model": candidate.name,
                "language": candidate.lang,
                "family": "",
                "text_id": "1",
                "text": "",
                "first_chunk_latency_ms": "",
                "total_generate_ms": "",
                "audio_duration_ms": "",
                "rtf": "",
                "callback_count": "",
                "first_chunk_samples": "",
                "first_chunk_audio_ms": "",
                "stream": "",
                "speed": capabilities["speed"],
                "voice": capabilities["voice"],
                "voice_type": capabilities["voice_type"],
                "output_wav": "",
                "status": "skipped",
                "reason": str(exc),
            }
            rows.append(row)

    rows = sort_summary_rows(rows)
    write_summary(summary_file, rows)
    print(f"Wrote summary: {summary_file}")
    excel_rows = order_rows_by_candidates(
        sort_excel_rows(rows),
        candidates,
    )
    write_excel_summary(excel_file, excel_rows)
    print(f"Wrote Excel summary: {excel_file}")
    print_terminal_summary(excel_rows)


if __name__ == "__main__":
    main()
