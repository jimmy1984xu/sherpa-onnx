"""Local pyannote 3.0 ONNX speaker-count activity inference."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np


TARGET_SAMPLE_RATE = 16000
WINDOW_SIZE = 10 * TARGET_SAMPLE_RATE
WINDOW_SHIFT = WINDOW_SIZE // 10
POWERSET_MASKS = np.array(
    [
        [0, 0, 0],  # class 0: no active speaker
        [1, 0, 0],  # class 1: local track A
        [0, 1, 0],  # class 2: local track B
        [0, 0, 1],  # class 3: local track C
        [1, 1, 0],  # class 4: local tracks A+B
        [1, 0, 1],  # class 5: local tracks A+C
        [0, 1, 1],  # class 6: local tracks B+C
    ],
    dtype=np.int8,
)
POWERSET_COUNTS = POWERSET_MASKS.sum(axis=1).astype(np.int8)
POWERSET_CLASS_BY_MASK = {
    tuple(int(value) for value in mask): class_index
    for class_index, mask in enumerate(POWERSET_MASKS)
}
_REQUIRED_METADATA = (
    "sample_rate",
    "window_size",
    "receptive_field_shift",
    "num_speakers",
    "powerset_max_classes",
    "num_classes",
)


@dataclass(frozen=True)
class SpeakerCountSpan:
    """A half-open time span with activity and local speaker-track identity."""

    start_ms: int
    end_ms: int
    active_speaker_count: int
    speaker_mask: tuple[int, int, int] | None = None
    class_index: int | None = None


@dataclass(frozen=True)
class SegmentationDiagnostics:
    """Raw window outputs plus a fused frame-level pyannote timeline."""

    window_starts_samples: np.ndarray
    window_valid_frame_counts: np.ndarray
    window_logits: np.ndarray
    window_probabilities: np.ndarray
    window_class_indices: np.ndarray
    window_masks: np.ndarray
    timeline_probabilities: np.ndarray
    timeline_class_indices: np.ndarray
    timeline_masks: np.ndarray
    timeline_coverage: np.ndarray
    activity_spans: list[SpeakerCountSpan]
    timeline_spans: list[SpeakerCountSpan]


class SegmentationRuntime:
    """Runs the local pyannote powerset model and aggregates its activity count."""

    def __init__(self, session: Any) -> None:
        self.session = session
        metadata = dict(session.get_modelmeta().custom_metadata_map)
        missing = [name for name in _REQUIRED_METADATA if name not in metadata]
        if missing:
            raise ValueError(f"Missing segmentation model metadata: {missing[0]}")

        self.sample_rate = int(metadata["sample_rate"])
        self.window_size = int(metadata["window_size"])
        self.receptive_field_shift = int(metadata["receptive_field_shift"])
        self.num_speakers = int(metadata["num_speakers"])
        self.powerset_max_classes = int(metadata["powerset_max_classes"])
        self.num_classes = int(metadata["num_classes"])
        if self.sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(f"Expected 16000 Hz segmentation model, got {self.sample_rate}")
        if self.window_size != WINDOW_SIZE:
            raise ValueError(f"Expected 10-second segmentation window, got {self.window_size}")
        if self.receptive_field_shift <= 0:
            raise ValueError("Segmentation receptive_field_shift must be positive")
        if self.num_speakers != 3 or self.powerset_max_classes != 2 or self.num_classes != 7:
            raise ValueError("Unsupported pyannote segmentation powerset metadata")

        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    def _validate_logits(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=np.float32)
        if values.ndim != 3 or values.shape[-1] != self.num_classes:
            raise ValueError(
                "Expected segmentation logits shaped (batch, frames, "
                f"{self.num_classes})"
            )
        if not np.isfinite(values).all():
            raise ValueError("Segmentation logits contain non-finite values")
        return values

    def decode_speaker_probabilities(self, logits: np.ndarray) -> np.ndarray:
        """Convert model scores to per-frame probabilities without discarding margins."""
        values = self._validate_logits(logits)
        shifted = values - np.max(values, axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        return np.ascontiguousarray(probabilities, dtype=np.float32)

    def decode_speaker_classes(self, logits: np.ndarray) -> np.ndarray:
        """Return the original Powerset class selected for every output frame."""
        probabilities = self.decode_speaker_probabilities(logits)
        return np.argmax(probabilities, axis=-1).astype(np.int64)

    def decode_speaker_masks(self, logits: np.ndarray) -> np.ndarray:
        """Decode Powerset classes while preserving all three local tracks."""
        return POWERSET_MASKS[self.decode_speaker_classes(logits)]

    def decode_speaker_counts(self, logits: np.ndarray) -> np.ndarray:
        """Convert pyannote Powerset logits to 0/1/2 active-speaker counts."""
        return self.decode_speaker_masks(logits).sum(axis=-1).astype(np.int8)

    def _window_starts(self, sample_count: int) -> list[int]:
        if sample_count <= 0:
            return []
        if sample_count <= self.window_size:
            return [0]
        starts = list(range(0, sample_count - self.window_size + 1, WINDOW_SHIFT))
        tail_start = len(starts) * WINDOW_SHIFT
        if tail_start < sample_count:
            starts.append(tail_start)
        return starts

    def _run_batch(
        self, waveform: np.ndarray, starts: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch = np.zeros((len(starts), 1, self.window_size), dtype=np.float32)
        for index, start in enumerate(starts):
            end = min(start + self.window_size, waveform.size)
            if end > start:
                batch[index, 0, : end - start] = waveform[start:end]
        outputs = self.session.run([self.output_name], {self.input_name: batch})
        if len(outputs) != 1:
            raise RuntimeError("Segmentation model returned an unexpected output count")
        values = np.asarray(outputs[0], dtype=np.float32)
        probabilities = self.decode_speaker_probabilities(values)
        classes = np.argmax(probabilities, axis=-1)
        return values, probabilities, POWERSET_MASKS[classes]

    @staticmethod
    def _alignment_score(local_activity: np.ndarray, reference_activity: np.ndarray) -> float:
        """Score one local/global track pairing over the window overlap."""
        local = np.asarray(local_activity, dtype=np.int8)
        reference = np.asarray(reference_activity, dtype=np.int8)
        # Reward simultaneous activity and penalize disagreements. This avoids
        # mapping every quiet local track to the same quiet global track.
        intersection = float(np.logical_and(local, reference).sum())
        disagreement = float(np.logical_xor(local, reference).sum())
        return 2.0 * intersection - disagreement

    def _best_track_permutation(
        self,
        local_masks: np.ndarray,
        start_frame: int,
        summed_masks: np.ndarray,
        coverage: np.ndarray,
    ) -> tuple[int, ...]:
        """Find the local-to-global track mapping using overlapping frame activity."""
        local = np.asarray(local_masks, dtype=np.int8)
        if local.ndim != 2 or local.shape[1] != self.num_speakers:
            raise ValueError("Expected local segmentation masks shaped (frames, 3)")

        overlap_start = max(0, start_frame)
        overlap_end = min(start_frame + local.shape[0], coverage.size)
        valid = coverage[overlap_start:overlap_end] > 0
        if not np.any(valid):
            return tuple(range(self.num_speakers))

        reference = (
            summed_masks[overlap_start:overlap_end]
            / np.maximum(coverage[overlap_start:overlap_end, None], 1)
            >= 0.5
        ).astype(np.int8)
        local_overlap = local[: overlap_end - overlap_start][valid]
        reference = reference[valid]

        import itertools

        best_permutation = tuple(range(self.num_speakers))
        best_score = float("-inf")
        for permutation in itertools.permutations(range(self.num_speakers)):
            score = sum(
                self._alignment_score(local_overlap[:, local_index], reference[:, global_index])
                for local_index, global_index in enumerate(permutation)
            )
            if score > best_score:
                best_score = score
                best_permutation = permutation
        return best_permutation

    @staticmethod
    def _apply_track_permutation(
        local_masks: np.ndarray, permutation: tuple[int, ...]
    ) -> np.ndarray:
        local = np.asarray(local_masks, dtype=np.int8)
        aligned = np.zeros_like(local)
        for local_index, global_index in enumerate(permutation):
            aligned[:, global_index] = local[:, local_index]
        return aligned

    def _remap_class_probabilities(
        self, probabilities: np.ndarray, permutation: tuple[int, ...]
    ) -> np.ndarray:
        """Remap all seven Powerset classes after local-track alignment."""
        values = np.asarray(probabilities, dtype=np.float32)
        if values.shape[-1] != self.num_classes:
            raise ValueError(f"Expected probabilities with {self.num_classes} classes")
        flat = values.reshape(-1, self.num_classes)
        remapped = np.zeros_like(flat)
        for local_class, local_mask in enumerate(POWERSET_MASKS):
            global_mask = [0] * self.num_speakers
            for local_index, global_index in enumerate(permutation):
                global_mask[global_index] = int(local_mask[local_index])
            global_class = POWERSET_CLASS_BY_MASK[tuple(global_mask)]
            remapped[:, global_class] += flat[:, local_class]
        return remapped.reshape(values.shape)

    def _align_local_tracks(
        self,
        local_masks: np.ndarray,
        start_frame: int,
        summed_masks: np.ndarray,
        coverage: np.ndarray,
    ) -> np.ndarray:
        """Align a window's arbitrary local track labels to prior global tracks."""
        permutation = self._best_track_permutation(
            local_masks, start_frame, summed_masks, coverage
        )
        return self._apply_track_permutation(local_masks, permutation)

    def _build_spans(
        self,
        masks: np.ndarray,
        class_indices: np.ndarray,
        duration_ms: int,
        *,
        split_on_class: bool,
    ) -> list[SpeakerCountSpan]:
        spans: list[SpeakerCountSpan] = []
        if masks.shape[0] == 0:
            return spans
        start_frame = 0
        for frame_index in range(1, masks.shape[0] + 1):
            same_state = (
                frame_index != masks.shape[0]
                and np.array_equal(masks[frame_index], masks[start_frame])
                and (
                    not split_on_class
                    or int(class_indices[frame_index]) == int(class_indices[start_frame])
                )
            )
            if same_state:
                continue
            start_ms = int(
                start_frame * self.receptive_field_shift * 1000 / self.sample_rate
            )
            end_ms = min(
                duration_ms,
                int(
                    math.ceil(
                        frame_index * self.receptive_field_shift * 1000 / self.sample_rate
                    )
                ),
            )
            if end_ms > start_ms:
                mask = tuple(int(value) for value in masks[start_frame])
                spans.append(
                    SpeakerCountSpan(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        active_speaker_count=int(sum(mask)),
                        speaker_mask=mask,
                        class_index=int(class_indices[start_frame]),
                    )
                )
            start_frame = frame_index
        return spans

    def infer_diagnostics(self, waveform: np.ndarray) -> SegmentationDiagnostics:
        """Run segmentation and retain raw windows plus reviewable fused timelines."""
        samples = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32))
        if samples.ndim != 1:
            raise ValueError("Segmentation waveform must be one-dimensional")
        if samples.size == 0:
            empty_windows = np.empty((0,), dtype=np.int64)
            empty_frames = np.empty((0, 0), dtype=np.int64)
            empty_masks = np.empty((0, self.num_speakers), dtype=np.int8)
            return SegmentationDiagnostics(
                window_starts_samples=empty_windows,
                window_valid_frame_counts=empty_windows.copy(),
                window_logits=np.empty((0, 0, self.num_classes), dtype=np.float32),
                window_probabilities=np.empty((0, 0, self.num_classes), dtype=np.float32),
                window_class_indices=empty_frames,
                window_masks=np.empty((0, 0, self.num_speakers), dtype=np.int8),
                timeline_probabilities=np.empty((0, self.num_classes), dtype=np.float32),
                timeline_class_indices=np.empty((0,), dtype=np.int64),
                timeline_masks=empty_masks,
                timeline_coverage=np.empty((0,), dtype=np.int32),
                activity_spans=[],
                timeline_spans=[],
            )

        starts = self._window_starts(samples.size)
        frame_count = math.ceil(samples.size / self.receptive_field_shift)
        all_window_logits: np.ndarray | None = None
        all_window_probabilities: np.ndarray | None = None
        all_window_classes: np.ndarray | None = None
        all_window_masks: np.ndarray | None = None
        window_valid_frame_counts = np.zeros(len(starts), dtype=np.int64)
        summed_masks = np.zeros((frame_count, self.num_speakers), dtype=np.float64)
        summed_probabilities = np.zeros((frame_count, self.num_classes), dtype=np.float64)
        coverage = np.zeros(frame_count, dtype=np.int32)
        batch_size = 32
        for batch_offset in range(0, len(starts), batch_size):
            batch_starts = starts[batch_offset : batch_offset + batch_size]
            logits, probabilities, masks = self._run_batch(samples, batch_starts)
            if all_window_logits is None:
                all_window_logits = np.empty(
                    (len(starts), logits.shape[1], self.num_classes), dtype=np.float32
                )
                all_window_probabilities = np.empty_like(all_window_logits)
                all_window_classes = np.empty(
                    (len(starts), logits.shape[1]), dtype=np.int64
                )
                all_window_masks = np.empty(
                    (len(starts), logits.shape[1], self.num_speakers), dtype=np.int8
                )
            window_slice = slice(batch_offset, batch_offset + len(batch_starts))
            all_window_logits[window_slice] = logits
            all_window_probabilities[window_slice] = probabilities
            all_window_classes[window_slice] = np.argmax(probabilities, axis=-1)
            all_window_masks[window_slice] = masks
            for relative_index, local_masks in enumerate(masks):
                start_frame = int(
                    batch_starts[relative_index] / self.receptive_field_shift + 0.5
                )
                end_frame = min(start_frame + local_masks.shape[0], frame_count)
                valid_frame_count = max(0, end_frame - start_frame)
                window_valid_frame_counts[batch_offset + relative_index] = valid_frame_count
                if valid_frame_count <= 0:
                    continue
                usable_masks = local_masks[:valid_frame_count]
                permutation = self._best_track_permutation(
                    usable_masks, start_frame, summed_masks, coverage
                )
                aligned_masks = self._apply_track_permutation(
                    usable_masks, permutation
                )
                usable_probabilities = probabilities[relative_index][:valid_frame_count]
                aligned_probabilities = self._remap_class_probabilities(
                    usable_probabilities, permutation
                )
                summed_masks[start_frame:end_frame] += aligned_masks
                summed_probabilities[start_frame:end_frame] += aligned_probabilities
                coverage[start_frame:end_frame] += 1

        assert all_window_logits is not None
        assert all_window_probabilities is not None
        assert all_window_classes is not None
        assert all_window_masks is not None
        global_probabilities = summed_probabilities / np.maximum(coverage[:, None], 1)
        global_class_indices = np.argmax(global_probabilities, axis=-1).astype(np.int64)
        global_masks = POWERSET_MASKS[global_class_indices]
        duration_ms = int(samples.size * 1000 / self.sample_rate)
        activity_spans = self._build_spans(
            global_masks, global_class_indices, duration_ms, split_on_class=False
        )
        timeline_spans = self._build_spans(
            global_masks, global_class_indices, duration_ms, split_on_class=True
        )
        return SegmentationDiagnostics(
            window_starts_samples=np.asarray(starts, dtype=np.int64),
            window_valid_frame_counts=window_valid_frame_counts,
            window_logits=all_window_logits,
            window_probabilities=all_window_probabilities,
            window_class_indices=all_window_classes,
            window_masks=all_window_masks,
            timeline_probabilities=global_probabilities.astype(np.float32),
            timeline_class_indices=global_class_indices,
            timeline_masks=global_masks.astype(np.int8),
            timeline_coverage=coverage,
            activity_spans=activity_spans,
            timeline_spans=timeline_spans,
        )

    def infer_speaker_count_spans(self, waveform: np.ndarray) -> list[SpeakerCountSpan]:
        """Return activity spans, preserving local speaker-track mask changes."""
        return self.infer_diagnostics(waveform).activity_spans


def create_segmentation_runtime(model_path: Path, *, num_threads: int) -> SegmentationRuntime:
    """Load the configured local pyannote ONNX model on CPU."""
    if num_threads <= 0:
        raise ValueError("Segmentation thread count must be positive")
    if not model_path.is_file():
        raise FileNotFoundError(f"Segmentation model not found: {model_path}")
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime is required; run: python -m pip install -r "
            "python-jimmy/offline-long-audio-pipeline/requirements.txt"
        ) from error

    options = ort.SessionOptions()
    options.intra_op_num_threads = num_threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return SegmentationRuntime(session)
