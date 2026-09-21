# Portable long-audio ASR and speaker evaluation

This directory is intentionally self-contained so it can be copied into another
Python project. It evaluates a pipeline's detailed `result.json`; it does **not**
change, invoke, or add evaluation responsibilities to either pipeline script.

## Install

```bash
python -m pip install kaldialign openpyxl pyannote.core pyannote.metrics
```

`evaluation.py` has a pure-Python edit-distance fallback when `kaldialign` is
not available. `openpyxl` is required only for the Excel workbook, and
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

The result and label durations do not need to be identical. Segment diagnostics
use positive time overlap rather than exact segment-ID equality, so split,
merge, and unmatched cases remain visible.

## Output layout

The runner writes only to `--output-dir`; it never overwrites pipeline files.
The pipeline's existing human-readable `result.txt` remains unchanged. The
independent evaluation output has a machine-readable `result.txt` whose rows
are:

```text
<segment_id> <speaker_id> <asr_text>
```

```text
<output-dir>/
├── result.txt
├── evaluation_manifest.json
├── evaluation_status.json
├── evaluation_report.md
├── inputs/
│   ├── <file_id>_asr.txt
│   ├── labels/<file_id>_label.txt
│   ├── wer_label.txt
│   └── wer_hyp.txt
├── asr/
│   ├── wer_detail.txt
│   ├── wer_summary.json
│   ├── evaluation.stdout.log
│   └── evaluation.stderr.log
├── speaker/
│   ├── speaker_diarization_summary.json
│   ├── speaker_diarization_per_file.json
│   ├── speaker_diarization_boundary_details.csv
│   ├── speaker_metrics.stdout.log
│   └── speaker_metrics.stderr.log
└── asr_segment_diff.xlsx
```

`evaluation_status.json` records every task as `success`, `failed`, or
`skipped`, with command, log, error, and artifact paths as applicable.

## Metric conventions

- **Whole-audio WER:** concatenate label and recognition text in time order per
  audio and use one utterance ID per audio. This avoids penalizing a valid
  change in segmentation directly as an ASR error.
- **DER:** `speaker_diarization_metrics.py` uses `skip_overlap=True`.
- **`multi` labels:** their text remains in whole-audio WER; they are skipped
  from DER overlap scoring and marked `overlap_not_scored` in the
  single-speaker segment diagnostic workbook.
- **Excel workbook:** `summary` contains aggregate WER/DER/boundary and
  segment-diagnostic counts. `segment_details` contains label/prediction time
  spans, overlap coverage, split/merge mapping type, ASR error counts, and
  unmatched rows.
