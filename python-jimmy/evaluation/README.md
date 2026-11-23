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

Labels are named `<file_id>_label.txt`. The current typed schema is:

```text
<segment_id> (<speaker_id>)(<segment_type>) <text>
```

`<segment_type>` must be one of `单人`, `短插话`, `重叠`, or `听不清`.
`重叠` must use speaker ID `MULTI`; `听不清` must use `UNCLEAR`; `单人` and
`短插话` cannot use either reserved ID. Invalid typed rows fail fast with the
label path and line number. Historical untyped rows remain readable: ordinary
speaker IDs default to `单人`, while `(MULTI)`/`(multi)` and `(UNCLEAR)` infer
`重叠` and `听不清` respectively.

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
│   ├── wer_exclude_unclear_detail.txt
│   ├── wer_clean_detail.txt
│   ├── wer_summary.json
│   ├── wer_metrics.json
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
three WER detail files, `wer_metrics.json`, Agent SDK time-aligned segment
workbook, speaker summary, and speaker boundary CSV. `wer_detail.txt` and
`wer_summary.json` remain the compatibility aliases for `wer_all`.
`wer_metrics.json` contains `wer_all`, `wer_exclude_unclear`, and `wer_clean`.
`speaker_summary.json` has the same metrics schema as the former
`speaker_diarization_summary.json` output.

For a meeting with no matching label, its directory contains only `asr.txt`;
`evaluation_report.md` records the `skipped_no_label` state. The root report is
the only cross-meeting artifact. The runner does not retain `result.txt`,
`evaluation_manifest.json`, `evaluation_status.json`, `inputs/`, `asr/`,
`speaker/`, WER ref/hyp files, subprocess logs, or
`speaker_diarization_per_file.json`.

## Metric conventions

- **Whole-audio WER:** concatenate label and recognition text in time order per
  audio and use one utterance ID per audio. This avoids penalizing a valid
  change in segmentation directly as an ASR error. The runner reports:
  - `wer_all`: `单人` + `短插话` + `重叠` + `听不清`;
  - `wer_exclude_unclear`: `单人` + `短插话` + `重叠`;
  - `wer_clean`: `单人` + `短插话`.
  For a given variant, recognition text remains the complete meeting
  hypothesis; only the reference types are filtered. This preserves the
  existing whole-audio WER convention and makes excluded-region output visible
  as insertions instead of silently discarding it.
- **DER and speaker inventory:** only `单人` and `短插话` enter DER, reference/
  predicted speaker counts, speaker-change boundaries, and segment-alignment
  metrics. DER uses `skip_overlap=True`, a 500 ms collar by default, and the
  eligible-reference timeline as UEM, so predictions in `重叠`/`听不清` regions
  do not become DER false alarms.
- **Boundary and segment diagnostics:** boundary matching uses the configured
  tolerance (500 ms by default). Segment alignment is one-to-one greedy IoU
  matching with IoU >= 0.50. The speaker summary reports precision, recall, and
  F1/counts for both diagnostics, plus DER miss, false alarm, and confusion.
- **Unknown-speaker diagnostic:** `重叠` and `听不清` are reference UNKNOWN
  regions. The summary reports duration-weighted UNKNOWN precision/recall/F1,
  with overlap and unclear recall breakdowns. A prediction counts only when its
  final speaker ID is exactly `UNKNOWN`; the evaluator never fabricates it.
- **Per-audio Excel workbook:** one `<file_id>_segment_asr_detail.xlsx` is
  generated for every matched label. It uses the Agent SDK's six columns:
  `标注片段ID`, `标注ASR文本`, `识别片段ID`, `识别ASR文本`, `对齐状态`, and `差异文本`.
  Time spans that overlap, or whose gap is at most 500 ms, are joined into a
  connected alignment group. This supports one-to-many, many-to-one,
  many-to-many, missing-recognition, and extra-recognition diagnostics without
  creating a wide custom table.
