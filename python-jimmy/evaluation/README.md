# Portable long-audio ASR and speaker evaluation

This directory is intentionally self-contained so it can be copied into another
Python project. It evaluates a pipeline's detailed `result.json`; it does **not**
change, invoke, or add evaluation responsibilities to either pipeline script.

## Install

```bash
python -m pip install kaldialign pyannote.core pyannote.metrics
```

`evaluation.py` has a pure-Python edit-distance fallback when `kaldialign` is
not available. The Excel workbook uses only Python's standard library, and
`pyannote.core` plus `pyannote.metrics` are required only for DER and
speaker-boundary metrics. The unified runner records a failed status and logs
when an optional metric dependency is absent; it does not silently omit it.

## Run one pipeline result

```bash
python python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py \
  --results-dir /path/to/pipeline-run \
  --labels-dir /path/to/labels \
  --output-dir /path/to/pipeline-run/evaluation \
  --language ZH \
  --boundary-tolerance-ms 500 \
  --collar-ms 500
```

For a single result audio, an explicit label can replace automatic lookup:

```bash
python python-jimmy/evaluation/evaluate_long_audio_asr_speaker.py \
  --results-dir /path/to/pipeline-run \
  --label /path/to/audio_label.txt \
  --output-dir /path/to/pipeline-run/evaluation
```

`--results-dir` may be a run directory or a parent directory. The runner
recursively discovers `result.json`. Each discovered audio must have a unique
file ID; evaluate different runs of the same audio separately.

## Input contracts

Pipeline `result.json` is the only pipeline input. It must contain:

- top-level `audio_name`;
- top-level `segments` array;
- for every segment: `segment_id`, `duration_ms`, `speaker_id`, and `asr_text`;
- an ID ending in `<file_id>_<start_ms>_<duration_ms>`.

Labels are named `<file_id>_label.txt` and contain rows such as:

```text
23_asr_1782715267098_1118_2335 (leslie) text
23_asr_1782715267098_3453_6823 (hui) text
```

The result and label durations do not need to be identical. Per-audio ASR detail
workbooks use the Agent SDK's fixed 500 ms time-alignment tolerance and
connected time groups, so split, merge, and unmatched cases remain visible.

## Output layout

The runner writes only to `--output-dir`; it never overwrites pipeline files.
Every discovered audio produces one meeting directory, including a single-audio
run. The directory name is the unique `file_id` derived from its segment IDs,
and the artifact names inside it are fixed for portable automation.

```text
<output-dir>/
├── evaluation_report.md
├── <file_id-A>/
│   ├── asr.txt
│   ├── wer_detail.txt
│   ├── wer_summary.json
│   ├── segment_asr_detail.xlsx
│   ├── speaker_summary.json
│   └── speaker_diarization_boundary_details.csv
└── <file_id-B>/
    └── ...the same fixed names...
```

`asr.txt` is always generated and contains:

```text
<segment_id> <speaker_id> <asr_text>
```

For a meeting that has a matching label, the runner independently produces its
WER detail/summary, Agent SDK time-aligned segment workbook, speaker summary,
and speaker boundary CSV. `speaker_summary.json` has the same metrics schema
as the former `speaker_diarization_summary.json` output.

For a meeting with no matching label, its directory contains only `asr.txt`;
`evaluation_report.md` records the `skipped_no_label` state. The root report is
the only cross-meeting artifact. The runner does not retain `result.txt`,
`evaluation_manifest.json`, `evaluation_status.json`, `inputs/`, `asr/`,
`speaker/`, WER ref/hyp files, subprocess logs, or
`speaker_diarization_per_file.json`.

## Metric conventions

- **Whole-audio WER:** concatenate label and recognition text in time order per
  audio and use one utterance ID per audio. This avoids penalizing a valid
  change in segmentation directly as an ASR error.
- **DER:** `speaker_diarization_metrics.py` uses `skip_overlap=True`.
- **`multi` labels:** their text remains in whole-audio WER and they are
  skipped from DER overlap scoring.
- **Per-audio Excel workbook:** one `<file_id>_segment_asr_detail.xlsx` is
  generated for every matched label. It uses the Agent SDK's six columns:
  `标注片段ID`, `标注ASR文本`, `识别片段ID`, `识别ASR文本`, `对齐状态`, and `差异文本`.
  Time spans that overlap, or whose gap is at most 500 ms, are joined into a
  connected alignment group. This supports one-to-many, many-to-one,
  many-to-many, missing-recognition, and extra-recognition diagnostics without
  creating a wide custom table.
