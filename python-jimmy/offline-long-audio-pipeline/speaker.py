"""Speaker embedding extraction, clustering, and stable speaker labels."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import sherpa_onnx

from vad import POST_OVERLAP_PAD_MS, SpeechSegment

SAMPLE_RATE = 16000
DEFAULT_MAX_EMBEDDING_SAMPLES = 10 * SAMPLE_RATE


def stable_speaker_ids(cluster_ids: Sequence[int]) -> list[str]:
    """Map arbitrary clustering labels to first-occurrence stable names."""
    labels: dict[int, str] = {}
    return [
        labels.setdefault(cluster_id, f"speaker_{len(labels):02d}")
        for cluster_id in cluster_ids
    ]


def _default_clusterer_factory(
    threshold: float, num_clusters: int
) -> Callable[[np.ndarray], Sequence[int]]:
    config = sherpa_onnx.FastClusteringConfig()
    if num_clusters > 0:
        config.num_clusters = num_clusters
    else:
        config.threshold = threshold
    if not config.validate():
        raise ValueError("Invalid FastClustering configuration")
    return sherpa_onnx.FastClustering(config)


def _extract_embedding(extractor: Any, samples: np.ndarray, max_samples: int) -> np.ndarray:
    if samples.size == 0:
        raise RuntimeError("no speech left for embedding")
    stream = extractor.create_stream()
    stream.accept_waveform(
        sample_rate=SAMPLE_RATE,
        waveform=np.ascontiguousarray(samples[:max_samples], dtype=np.float32),
    )
    stream.input_finished()
    if not extractor.is_ready(stream):
        raise RuntimeError("extractor is not ready")
    embedding = np.ascontiguousarray(np.asarray(extractor.compute(stream), dtype=np.float32))
    if embedding.size == 0:
        raise RuntimeError("empty embedding")
    if not np.isfinite(embedding).all():
        raise RuntimeError("embedding contains non-finite values")
    return embedding


def samples_for_embedding(
    segment: SpeechSegment, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """Return the waveform used for Titanet, skipping overlap and a short tail after it."""
    samples = np.ascontiguousarray(np.asarray(segment.samples, dtype=np.float32))
    if sample_rate <= 0:
        raise ValueError("Embedding sample rate must be positive")
    skip_regions = segment.embedding_skip_regions(POST_OVERLAP_PAD_MS)
    if not skip_regions:
        return samples
    keep = np.ones(samples.size, dtype=bool)
    for start_ms, end_ms in skip_regions:
        rel_start = int(round((start_ms - segment.start_ms) * sample_rate / 1000.0))
        rel_end = int(round((end_ms - segment.start_ms) * sample_rate / 1000.0))
        rel_start = max(0, min(samples.size, rel_start))
        rel_end = max(0, min(samples.size, rel_end))
        keep[rel_start:rel_end] = False
    cropped = samples[keep]
    if cropped.size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.ascontiguousarray(cropped, dtype=np.float32)


def _adjacent_similarity(
    previous: np.ndarray | None, current: np.ndarray | None
) -> float | None:
    """Return cosine similarity only when adjacent segments both have embeddings."""
    if previous is None or current is None:
        return None
    previous64 = previous.astype(np.float64, copy=False)
    current64 = current.astype(np.float64, copy=False)
    denominator = float(np.linalg.norm(previous64) * np.linalg.norm(current64))
    if denominator == 0.0:
        return None
    similarity = float(np.dot(previous64, current64) / denominator)
    return float(np.clip(similarity, -1.0, 1.0))


def populate_previous_segment_similarities(
    extractor: Any,
    segments: Sequence[SpeechSegment],
    *,
    max_embedding_samples: int = DEFAULT_MAX_EMBEDDING_SAMPLES,
) -> int:
    """Populate adjacent-turn similarity without changing diarization speaker IDs.

    Returns the number of Titanet extraction failures. This intentionally does
    not construct a clusterer or modify preassigned diarization labels.
    """
    if max_embedding_samples <= 0:
        raise ValueError("Maximum embedding samples must be positive")

    embeddings: dict[int, np.ndarray] = {}
    error_count = 0
    for index, segment in enumerate(segments):
        segment.previous_segment_similarity = None
        try:
            embeddings[index] = _extract_embedding(
                extractor, samples_for_embedding(segment), max_embedding_samples
            )
        except Exception:
            error_count += 1

    for index, segment in enumerate(segments):
        if index:
            segment.previous_segment_similarity = _adjacent_similarity(
                embeddings.get(index - 1), embeddings.get(index)
            )
    return error_count

def assign_speaker_ids(
    extractor: Any,
    segments: Sequence[SpeechSegment],
    *,
    cluster_threshold: float,
    num_clusters: int,
    max_embedding_samples: int = DEFAULT_MAX_EMBEDDING_SAMPLES,
    clusterer_factory: Callable[[float, int], Callable[[np.ndarray], Sequence[int]]] | None = None,
) -> None:
    """Extract every segment embedding, assign clusters, and compare adjacent segments."""
    if max_embedding_samples <= 0:
        raise ValueError("Maximum embedding samples must be positive")

    valid_indices: list[int] = []
    embeddings: list[np.ndarray] = []
    embeddings_by_index: dict[int, np.ndarray] = {}
    for index, segment in enumerate(segments):
        segment.speaker_id = "unknown"
        segment.embedding_error = None
        segment.previous_segment_similarity = None
        try:
            embedding = _extract_embedding(
                extractor, samples_for_embedding(segment), max_embedding_samples
            )
        except Exception as error:
            segment.embedding_error = str(error)
            continue
        valid_indices.append(index)
        embeddings.append(embedding)
        embeddings_by_index[index] = embedding

    for index, segment in enumerate(segments):
        if index:
            segment.previous_segment_similarity = _adjacent_similarity(
                embeddings_by_index.get(index - 1), embeddings_by_index.get(index)
            )

    if not embeddings:
        return
    if num_clusters > len(embeddings):
        raise ValueError(
            f"num_clusters={num_clusters} exceeds successful embeddings={len(embeddings)}"
        )

    embedding_matrix = np.ascontiguousarray(np.stack(embeddings), dtype=np.float32)
    factory = clusterer_factory or _default_clusterer_factory
    labels = list(factory(cluster_threshold, num_clusters)(embedding_matrix))
    if len(labels) != len(valid_indices):
        raise RuntimeError("FastClustering returned an unexpected number of labels")

    for index, speaker_id in zip(valid_indices, stable_speaker_ids(labels)):
        segments[index].speaker_id = speaker_id



def l2_normalize(embedding: np.ndarray) -> np.ndarray | None:
    """Return a contiguous float32 L2-normalized vector, or None for zero norm."""
    vector = np.ascontiguousarray(np.asarray(embedding, dtype=np.float32).reshape(-1))
    if vector.size == 0 or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector.astype(np.float64, copy=False)))
    if norm == 0.0 or not np.isfinite(norm):
        return None
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def cosine(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    """Return cosine similarity, or None when either vector cannot be compared."""
    if left is None or right is None:
        return None
    return _adjacent_similarity(left, right)


def normalized_mean_by_stable_id(
    stable_ids: Sequence[str], embeddings: Sequence[np.ndarray]
) -> dict[str, np.ndarray]:
    """Build normalized centroids from normalized member embeddings."""
    grouped: dict[str, list[np.ndarray]] = {}
    for stable_id, embedding in zip(stable_ids, embeddings):
        grouped.setdefault(stable_id, []).append(embedding)

    centroids: dict[str, np.ndarray] = {}
    for stable_id, members in grouped.items():
        centroid = l2_normalize(np.mean(np.stack(members), axis=0))
        if centroid is not None:
            centroids[stable_id] = centroid
    return centroids


def max_similarity(
    embedding: np.ndarray | None, centroids: dict[str, np.ndarray]
) -> tuple[str | None, float | None]:
    """Return the stable ID and score of the most similar centroid."""
    best_id: str | None = None
    best_score: float | None = None
    for stable_id, centroid in centroids.items():
        score = cosine(embedding, centroid)
        if score is not None and (best_score is None or score > best_score):
            best_id, best_score = stable_id, score
    return best_id, best_score


def assign_speaker_ids_with_centroids(
    extractor: Any,
    segments: Sequence[SpeechSegment],
    *,
    cluster_threshold: float,
    num_clusters: int,
    assignment_similarity_threshold: float,
    max_embedding_samples: int = DEFAULT_MAX_EMBEDDING_SAMPLES,
    clusterer_factory: Callable[[float, int], Callable[[np.ndarray], Sequence[int]]] | None = None,
) -> tuple[int, int, int]:
    """Cluster eligible embeddings and assign excluded segments from final centroids.

    All segments receive a Titanet embedding attempt and an adjacent-similarity
    value. Only long, single-speaker segments may create or update clusters.
    Excluded segments are read-only centroid comparisons and therefore cannot
    influence the resulting speaker inventory.

    Returns ``(embedding_errors, assigned_excluded, unknown_excluded)``.
    """
    if max_embedding_samples <= 0:
        raise ValueError("Maximum embedding samples must be positive")
    if not -1.0 <= assignment_similarity_threshold <= 1.0:
        raise ValueError("assignment_similarity_threshold must be in [-1.0, 1.0]")

    embedding_errors = 0
    embeddings_by_index: dict[int, np.ndarray] = {}
    for index, segment in enumerate(segments):
        segment.speaker_id = "unknown"
        segment.embedding_error = None
        segment.embedding = None
        segment.previous_segment_similarity = None
        segment.cluster_assignment_similarity = None
        try:
            embedding = l2_normalize(
                _extract_embedding(
                    extractor, samples_for_embedding(segment), max_embedding_samples
                )
            )
            if embedding is None:
                raise RuntimeError("zero-norm embedding")
        except Exception as error:
            segment.embedding_error = str(error)
            embedding_errors += 1
            continue
        segment.embedding = embedding
        embeddings_by_index[index] = embedding

    for index, segment in enumerate(segments):
        if index:
            segment.previous_segment_similarity = cosine(
                embeddings_by_index.get(index - 1), embeddings_by_index.get(index)
            )

    eligible_indices = [
        index
        for index, segment in enumerate(segments)
        if segment.is_cluster_eligible and segment.embedding is not None
    ]
    if num_clusters > len(eligible_indices):
        raise ValueError(
            "num_clusters="
            f"{num_clusters} exceeds successful eligible embeddings={len(eligible_indices)}"
        )

    centroids: dict[str, np.ndarray] = {}
    if eligible_indices:
        eligible_embeddings = [segments[index].embedding for index in eligible_indices]
        embedding_matrix = np.ascontiguousarray(
            np.stack(eligible_embeddings), dtype=np.float32
        )
        factory = clusterer_factory or _default_clusterer_factory
        labels = list(factory(cluster_threshold, num_clusters)(embedding_matrix))
        if len(labels) != len(eligible_indices):
            raise RuntimeError("FastClustering returned an unexpected number of labels")
        stable_ids = stable_speaker_ids(labels)
        centroids = normalized_mean_by_stable_id(stable_ids, eligible_embeddings)
        for index, stable_id in zip(eligible_indices, stable_ids):
            segment = segments[index]
            segment.speaker_id = stable_id
            segment.cluster_assignment_similarity = cosine(
                segment.embedding, centroids.get(stable_id)
            )

    centroid_assigned_excluded = 0
    unknown_excluded = 0
    for segment in segments:
        if segment.is_cluster_eligible or segment.embedding is None:
            continue
        stable_id, score = max_similarity(segment.embedding, centroids)
        segment.cluster_assignment_similarity = score
        if stable_id is not None and score is not None and score >= assignment_similarity_threshold:
            segment.speaker_id = stable_id
            centroid_assigned_excluded += 1
        else:
            unknown_excluded += 1

    return embedding_errors, centroid_assigned_excluded, unknown_excluded
