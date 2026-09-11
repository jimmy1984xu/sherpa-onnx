"""Resolve VAD speech and pyannote speaker-count activity on one time axis."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import numpy as np

from segmentation import POWERSET_CLASS_BY_MASK, SpeakerCountSpan
from vad import SpeechSegment


SINGLE_SPEAKER = "single_speaker"
OVERLAPPED_SPEAKERS = "overlapped_speakers"
UNKNOWN_ACTIVITY = "unknown_activity"


def _seconds_to_ms(seconds: float) -> int:
    return int(round(seconds * 1000.0))


def _class_index_for_span(span: SpeakerCountSpan) -> int:
    if span.class_index is not None:
        return int(span.class_index)
    if span.speaker_mask is not None:
        return POWERSET_CLASS_BY_MASK.get(tuple(int(value) for value in span.speaker_mask), 0)
    return 0


def format_pyannote_mask_rle(
    activity: Sequence[SpeakerCountSpan], start_ms: int, end_ms: int
) -> str:
    """Run-length encode raw pyannote class indices inside [start_ms, end_ms)."""
    if end_ms <= start_ms:
        return ""
    pieces: list[tuple[int, int]] = []

    def append(duration_ms: int, class_index: int) -> None:
        if duration_ms <= 0:
            return
        if pieces and pieces[-1][1] == class_index:
            pieces[-1] = (pieces[-1][0] + duration_ms, class_index)
        else:
            pieces.append((duration_ms, class_index))

    cursor = start_ms
    for span in sorted(activity, key=lambda item: (item.start_ms, item.end_ms)):
        if span.end_ms <= start_ms or span.start_ms >= end_ms:
            continue
        left = max(span.start_ms, start_ms, cursor)
        right = min(span.end_ms, end_ms)
        if left > cursor:
            append(left - cursor, 0)
            cursor = left
        if right > cursor:
            append(right - cursor, _class_index_for_span(span))
            cursor = right
    if cursor < end_ms:
        append(end_ms - cursor, 0)
    return "".join(f"[{duration_ms},{class_index}]" for duration_ms, class_index in pieces)


def _track_key(span: SpeakerCountSpan) -> tuple[int, int, int] | None:
    mask = span.speaker_mask
    if mask is None or sum(mask) == 0:
        return None
    return mask


def smooth_activity_spans(
    activity: Sequence[SpeakerCountSpan],
    *,
    min_duration_on_ms: int,
    min_duration_off_ms: int,
) -> list[SpeakerCountSpan]:
    """Apply official per-track min_duration_off merge then min_duration_on drop."""
    if min_duration_on_ms < 0:
        raise ValueError("min_duration_on must be non-negative")
    if min_duration_off_ms < 0:
        raise ValueError("min_duration_off must be non-negative")

    tracks: dict[tuple[int, int, int] | None, list[SpeakerCountSpan]] = {}
    for span in activity:
        if span.end_ms <= span.start_ms:
            continue
        tracks.setdefault(_track_key(span), []).append(span)

    smoothed: list[SpeakerCountSpan] = []
    for key, spans in tracks.items():
        ordered = sorted(spans, key=lambda item: (item.start_ms, item.end_ms))
        merged: list[SpeakerCountSpan] = []
        if key is not None:
            for span in ordered:
                if merged and span.start_ms - merged[-1].end_ms <= min_duration_off_ms:
                    previous = merged[-1]
                    merged[-1] = SpeakerCountSpan(
                        previous.start_ms,
                        max(previous.end_ms, span.end_ms),
                        previous.active_speaker_count,
                        previous.speaker_mask,
                        previous.class_index,
                    )
                else:
                    merged.append(span)
        else:
            merged = ordered
        for span in merged:
            if span.end_ms - span.start_ms >= min_duration_on_ms:
                smoothed.append(span)
    return sorted(smoothed, key=lambda item: (item.start_ms, item.end_ms))


@dataclass(frozen=True)
class TimelineResolutionStats:
    true_overlap_count: int


@dataclass(frozen=True)
class _AtomicInterval:
    start_ms: int
    end_ms: int
    composition: str
    speaker_mask: tuple[int, int, int] | None = None
    overlap_regions: tuple[tuple[int, int], ...] = ()


def _composition_for_interval(
    start_ms: int, end_ms: int, activity: Sequence[SpeakerCountSpan]
) -> tuple[str, tuple[int, int, int] | None]:
    relevant = [
        span
        for span in activity
        if span.start_ms < end_ms and start_ms < span.end_ms
    ]
    masks = [span.speaker_mask for span in relevant if span.speaker_mask is not None]
    if masks:
        combined = tuple(
            int(any(mask[index] for mask in masks))
            for index in range(3)
        )
        count = sum(combined)
        if count >= 2:
            return OVERLAPPED_SPEAKERS, combined
        if count == 1:
            return SINGLE_SPEAKER, combined
        return UNKNOWN_ACTIVITY, combined

    count = max((span.active_speaker_count for span in relevant), default=0)
    if count >= 2:
        return OVERLAPPED_SPEAKERS, None
    if count == 1:
        return SINGLE_SPEAKER, None
    return UNKNOWN_ACTIVITY, None


def _apply_vad_speech_fallback(
    intervals: Sequence[_AtomicInterval],
) -> list[_AtomicInterval]:
    """Fill only activity holes that are consistent with one local track."""
    resolved = list(intervals)
    index = 0
    while index < len(resolved):
        if resolved[index].composition != UNKNOWN_ACTIVITY:
            index += 1
            continue
        end_index = index + 1
        while (
            end_index < len(resolved)
            and resolved[end_index].composition == UNKNOWN_ACTIVITY
            and resolved[end_index - 1].end_ms == resolved[end_index].start_ms
        ):
            end_index += 1

        left = resolved[index - 1] if index > 0 else None
        right = resolved[end_index] if end_index < len(resolved) else None
        left_single = left is not None and left.composition == SINGLE_SPEAKER
        right_single = right is not None and right.composition == SINGLE_SPEAKER
        masks_agree = (
            left_single
            and right_single
            and left.speaker_mask is not None
            and right.speaker_mask is not None
            and left.speaker_mask == right.speaker_mask
        )
        can_fill = (
            masks_agree
            or (left_single and right is None)
            or (right_single and left is None)
            or (left_single and right_single and left.speaker_mask is None and right.speaker_mask is None)
            or (left_single and not right_single)
            or (right_single and not left_single)
        )
        if can_fill:
            fill_mask = (
                left.speaker_mask if left_single and left.speaker_mask is not None else
                right.speaker_mask if right_single else None
            )
            for candidate in range(index, end_index):
                resolved[candidate] = _AtomicInterval(
                    resolved[candidate].start_ms,
                    resolved[candidate].end_ms,
                    SINGLE_SPEAKER,
                    fill_mask,
                )
        index = end_index
    return resolved


def _merge_adjacent(
    intervals: Sequence[_AtomicInterval],
    *,
    min_exclusive_ms: int | None = None,
) -> list[_AtomicInterval]:
    merged: list[_AtomicInterval] = []
    for interval in intervals:
        if (
            merged
            and merged[-1].composition == interval.composition
            and merged[-1].speaker_mask == interval.speaker_mask
            and merged[-1].end_ms == interval.start_ms
        ):
            if (
                min_exclusive_ms is not None
                and _exclusive_overlap_ms(merged[-1]) >= min_exclusive_ms
                and _exclusive_overlap_ms(interval) >= min_exclusive_ms
            ):
                merged.append(interval)
                continue
            merged[-1] = _AtomicInterval(
                merged[-1].start_ms,
                interval.end_ms,
                interval.composition,
                interval.speaker_mask,
                merged[-1].overlap_regions + interval.overlap_regions,
            )
        else:
            merged.append(interval)
    return merged


def _absorb_short_intervals(
    intervals: Sequence[_AtomicInterval], min_duration_ms: int
) -> list[_AtomicInterval]:
    """Fold sub-threshold atoms into a neighbor instead of emitting them."""
    if min_duration_ms <= 0:
        return list(intervals)
    resolved = list(intervals)
    index = 0
    while index < len(resolved):
        duration_ms = resolved[index].end_ms - resolved[index].start_ms
        if duration_ms >= min_duration_ms:
            index += 1
            continue
        if index > 0:
            previous = resolved[index - 1]
            resolved[index - 1] = _AtomicInterval(
                previous.start_ms,
                resolved[index].end_ms,
                previous.composition,
                previous.speaker_mask,
                previous.overlap_regions + resolved[index].overlap_regions,
            )
            del resolved[index]
            continue
        if index + 1 < len(resolved):
            nxt = resolved[index + 1]
            resolved[index + 1] = _AtomicInterval(
                resolved[index].start_ms,
                nxt.end_ms,
                nxt.composition,
                nxt.speaker_mask,
                resolved[index].overlap_regions + nxt.overlap_regions,
            )
            del resolved[index]
            continue
        index += 1
    return resolved


MERGE_GAP_MAX_MS = 2000
MIN_SINGLE_CUT_MS = 1000
MIN_SAME_MASK_CUT_MS = 2000


def _gap_ms(earlier: _AtomicInterval, later: _AtomicInterval) -> int:
    return later.start_ms - earlier.end_ms


def _exclusive_overlap_ms(interval: _AtomicInterval) -> int:
    overlap_ms = sum(end_ms - start_ms for start_ms, end_ms in interval.overlap_regions)
    return max(0, interval.end_ms - interval.start_ms - overlap_ms)


def _is_overlap_island(interval: _AtomicInterval) -> bool:
    return interval.composition == OVERLAPPED_SPEAKERS


def _fold_overlaps(intervals: Sequence[_AtomicInterval]) -> list[_AtomicInterval]:
    """Merge overlap islands into the following sentence, else the previous."""
    resolved = list(intervals)
    index = 0
    while index < len(resolved):
        current = resolved[index]
        if not _is_overlap_island(current):
            index += 1
            continue
        host_index = None
        for candidate in range(index + 1, len(resolved)):
            if not _is_overlap_island(resolved[candidate]):
                host_index = candidate
                break
        if host_index is None:
            for candidate in range(index - 1, -1, -1):
                if not _is_overlap_island(resolved[candidate]):
                    host_index = candidate
                    break
        if host_index is None:
            index += 1
            continue
        host = resolved[host_index]
        if host_index > index and _gap_ms(current, host) > MERGE_GAP_MAX_MS:
            index += 1
            continue
        if host_index < index and _gap_ms(host, current) > MERGE_GAP_MAX_MS:
            index += 1
            continue
        region = (current.start_ms, current.end_ms)
        regions = host.overlap_regions + current.overlap_regions + (region,)
        if host_index > index:
            resolved[host_index] = _AtomicInterval(
                current.start_ms,
                host.end_ms,
                host.composition,
                host.speaker_mask,
                regions,
            )
            del resolved[index]
            continue
        resolved[host_index] = _AtomicInterval(
            host.start_ms,
            current.end_ms,
            host.composition,
            host.speaker_mask,
            regions,
        )
        del resolved[index]
    return resolved


def _fold_unknown_activity(intervals: Sequence[_AtomicInterval]) -> list[_AtomicInterval]:
    """Merge unknown islands into the following sentence, else the previous."""
    resolved = list(intervals)
    index = 0
    while index < len(resolved):
        current = resolved[index]
        if current.composition != UNKNOWN_ACTIVITY:
            index += 1
            continue
        if current.end_ms - current.start_ms > MERGE_GAP_MAX_MS:
            index += 1
            continue
        host_index = None
        for candidate in range(index + 1, len(resolved)):
            if resolved[candidate].composition != UNKNOWN_ACTIVITY:
                host_index = candidate
                break
        if host_index is None:
            for candidate in range(index - 1, -1, -1):
                if resolved[candidate].composition != UNKNOWN_ACTIVITY:
                    host_index = candidate
                    break
        if host_index is None:
            index += 1
            continue
        host = resolved[host_index]
        if host_index > index and _gap_ms(current, host) > MERGE_GAP_MAX_MS:
            index += 1
            continue
        if host_index < index and _gap_ms(host, current) > MERGE_GAP_MAX_MS:
            index += 1
            continue
        regions = host.overlap_regions + current.overlap_regions
        if host_index > index:
            resolved[host_index] = _AtomicInterval(
                current.start_ms,
                host.end_ms,
                host.composition,
                host.speaker_mask,
                regions,
            )
            del resolved[index]
            continue
        resolved[host_index] = _AtomicInterval(
            host.start_ms,
            current.end_ms,
            host.composition,
            host.speaker_mask,
            regions,
        )
        del resolved[index]
    return resolved


def _merge_weak_single_cuts(intervals: Sequence[_AtomicInterval]) -> list[_AtomicInterval]:
    """Drop a single-speaker cut unless both exclusive islands are at least 1s.

    Exclusive duration subtracts absorbed overlap. Short islands prefer the
    following segment; with no follower they join the previous.
    """
    resolved = list(intervals)
    index = 0
    while index + 1 < len(resolved):
        left = resolved[index]
        right = resolved[index + 1]
        can_cut = (
            left.composition == SINGLE_SPEAKER
            and right.composition == SINGLE_SPEAKER
            and left.end_ms == right.start_ms
            and left.speaker_mask != right.speaker_mask
        )
        if not can_cut:
            index += 1
            continue
        left_dur = _exclusive_overlap_ms(left)
        right_dur = _exclusive_overlap_ms(right)
        if left_dur >= MIN_SINGLE_CUT_MS and right_dur >= MIN_SINGLE_CUT_MS:
            index += 1
            continue
        nxt = resolved[index + 2] if index + 2 < len(resolved) else None
        follow_ok = (
            nxt is not None
            and nxt.composition == SINGLE_SPEAKER
            and right.end_ms == nxt.start_ms
            and _gap_ms(right, nxt) <= MERGE_GAP_MAX_MS
        )
        if left_dur < MIN_SINGLE_CUT_MS:
            resolved[index + 1] = _AtomicInterval(
                left.start_ms,
                right.end_ms,
                right.composition,
                right.speaker_mask,
                left.overlap_regions + right.overlap_regions,
            )
            del resolved[index]
            continue
        if follow_ok:
            resolved[index + 2] = _AtomicInterval(
                right.start_ms,
                nxt.end_ms,
                nxt.composition,
                nxt.speaker_mask,
                right.overlap_regions + nxt.overlap_regions,
            )
            del resolved[index + 1]
            continue
        resolved[index] = _AtomicInterval(
            left.start_ms,
            right.end_ms,
            left.composition,
            left.speaker_mask,
            left.overlap_regions + right.overlap_regions,
        )
        del resolved[index + 1]
    return resolved


def finalize_vad_only_segments(
    vad_segments: Sequence[SpeechSegment],
) -> tuple[list[SpeechSegment], TimelineResolutionStats]:
    """Keep VAD speech regions as the final ASR segments without pyannote cuts."""
    output: list[SpeechSegment] = []
    for vad_segment in sorted(vad_segments, key=lambda item: (item.start_ms, item.end_ms)):
        if vad_segment.end_ms <= vad_segment.start_ms:
            continue
        output.append(
            SpeechSegment(
                segment_index=len(output) + 1,
                start_ms=vad_segment.start_ms,
                end_ms=vad_segment.end_ms,
                samples=vad_segment.samples,
                speaker_composition=SINGLE_SPEAKER,
                cut_left="vad",
                cut_right="vad",
            )
        )
    if any(
        previous.end_ms > current.start_ms
        for previous, current in zip(output, output[1:])
    ):
        raise AssertionError("Resolved output segments overlap")
    return output, TimelineResolutionStats(true_overlap_count=0)


def resolve_final_segments(
    vad_segments: Sequence[SpeechSegment],
    activity: Sequence[SpeakerCountSpan],
    waveform: np.ndarray,
    sample_rate: int,
    *,
    min_duration_on: float = 0.5,
    min_duration_off: float = 0.5,
) -> tuple[list[SpeechSegment], TimelineResolutionStats]:
    """Create chronological non-overlapping ASR segments from VAD/activity data."""
    if sample_rate <= 0:
        raise ValueError("Timeline sample rate must be positive")
    if min_duration_on < 0:
        raise ValueError("min_duration_on must be non-negative")
    if min_duration_off < 0:
        raise ValueError("min_duration_off must be non-negative")

    samples = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32))
    output: list[SpeechSegment] = []
    ordered_activity = smooth_activity_spans(
        activity,
        min_duration_on_ms=_seconds_to_ms(min_duration_on),
        min_duration_off_ms=_seconds_to_ms(min_duration_off),
    )

    for vad_segment in sorted(vad_segments, key=lambda item: (item.start_ms, item.end_ms)):
        if vad_segment.end_ms <= vad_segment.start_ms:
            continue
        relevant = [
            span
            for span in ordered_activity
            if span.start_ms < vad_segment.end_ms and vad_segment.start_ms < span.end_ms
        ]
        cut_points = {vad_segment.start_ms, vad_segment.end_ms}
        for span in relevant:
            cut_points.add(max(vad_segment.start_ms, span.start_ms))
            cut_points.add(min(vad_segment.end_ms, span.end_ms))
        ordered_points = sorted(cut_points)
        atomic = []
        for start_ms, end_ms in zip(ordered_points, ordered_points[1:]):
            if end_ms <= start_ms:
                continue
            composition, speaker_mask = _composition_for_interval(
                start_ms, end_ms, relevant
            )
            atomic.append(
                _AtomicInterval(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    composition=composition,
                    speaker_mask=speaker_mask,
                )
            )
        atomic = _apply_vad_speech_fallback(atomic)
        atomic = _absorb_short_intervals(atomic, _seconds_to_ms(min_duration_on))
        atomic = _fold_overlaps(_merge_adjacent(atomic))
        atomic = _merge_weak_single_cuts(
            _merge_adjacent(
                _fold_unknown_activity(atomic),
                min_exclusive_ms=MIN_SAME_MASK_CUT_MS,
            )
        )
        if atomic and (
            atomic[0].start_ms != vad_segment.start_ms
            or atomic[-1].end_ms != vad_segment.end_ms
        ):
            raise AssertionError("Resolved segments must keep VAD hard bounds")
        for interval in atomic:
            start_sample = max(0, math.floor(interval.start_ms * sample_rate / 1000))
            end_sample = min(samples.size, math.ceil(interval.end_ms * sample_rate / 1000))
            if end_sample <= start_sample:
                continue
            overlap_regions = list(interval.overlap_regions)
            if interval.composition == OVERLAPPED_SPEAKERS and not overlap_regions:
                overlap_regions = [(interval.start_ms, interval.end_ms)]
            output.append(
                SpeechSegment(
                    segment_index=len(output) + 1,
                    start_ms=interval.start_ms,
                    end_ms=interval.end_ms,
                    samples=np.ascontiguousarray(samples[start_sample:end_sample]),
                    speaker_composition=interval.composition,
                    overlap_regions=overlap_regions,
                    cut_left="vad" if interval.start_ms == vad_segment.start_ms else "pyannote",
                    cut_right="vad" if interval.end_ms == vad_segment.end_ms else "pyannote",
                    pyannote_mask=format_pyannote_mask_rle(
                        activity, interval.start_ms, interval.end_ms
                    ),
                )
            )

    if any(
        previous.end_ms > current.start_ms
        for previous, current in zip(output, output[1:])
    ):
        raise AssertionError("Resolved output segments overlap")
    true_overlap_count = sum(
        segment.speaker_composition == OVERLAPPED_SPEAKERS for segment in output
    )
    return output, TimelineResolutionStats(true_overlap_count=true_overlap_count)
