"""Robust second-stage merging for initial clean FastClustering labels."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty 2-D array")
    if not np.isfinite(values).all():
        raise ValueError("embeddings must contain only finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0.0) or not np.isfinite(norms).all():
        raise ValueError("embeddings must not contain zero-norm rows")
    return values / norms


def _stable_relabel(labels: Sequence[int]) -> list[int]:
    mapping: dict[int, int] = {}
    result: list[int] = []
    for label in labels:
        value = int(label)
        if value not in mapping:
            mapping[value] = len(mapping)
        result.append(mapping[value])
    return result


def robust_center(
    embeddings: np.ndarray,
    indices: Sequence[int],
    *,
    center_method: str = "trimmed_centroid",
    trim_ratio: float = 0.10,
) -> np.ndarray:
    """Return a normalized medoid or trimmed centroid for one cluster."""
    if center_method not in {"trimmed_centroid", "medoid"}:
        raise ValueError(f"unsupported center_method: {center_method}")
    if not 0.0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must be in [0, 0.5)")
    normalized = _normalize_embeddings(embeddings)
    members = list(indices)
    if not members:
        raise ValueError("cluster must contain at least one member")
    if any(index < 0 or index >= len(normalized) for index in members):
        raise ValueError("cluster member index is out of range")
    values = normalized[members]

    if center_method == "medoid":
        scores = values @ values.T
        return np.asarray(values[int(np.argmax(np.sum(scores, axis=1)))], dtype=np.float64)

    center = np.mean(values, axis=0)
    center_norm = float(np.linalg.norm(center))
    if center_norm == 0.0 or not np.isfinite(center_norm):
        raise ValueError("cluster centroid has zero norm")
    center = center / center_norm
    keep_count = max(1, len(values) - int(len(values) * trim_ratio))
    selected = sorted(
        range(len(values)),
        key=lambda index: (-float(values[index] @ center), index),
    )[:keep_count]
    trimmed = np.mean(values[selected], axis=0)
    trimmed_norm = float(np.linalg.norm(trimmed))
    if trimmed_norm == 0.0 or not np.isfinite(trimmed_norm):
        raise ValueError("trimmed centroid has zero norm")
    return np.asarray(trimmed / trimmed_norm, dtype=np.float64)


def merge_initial_clusters(
    embeddings: np.ndarray,
    initial_labels: Sequence[int],
    *,
    similarity_threshold: float = 0.75,
    center_method: str = "trimmed_centroid",
    trim_ratio: float = 0.10,
) -> tuple[list[int], list[dict[str, object]]]:
    """Greedily merge complete initial clusters by robust-center similarity.

    The returned labels contain one label for every input row. No embedding is
    removed when calculating a robust center or when merging clusters.
    """
    normalized = _normalize_embeddings(embeddings)
    if len(initial_labels) != len(normalized):
        raise ValueError("initial_labels length must match embeddings")
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [-1, 1]")

    labels = _stable_relabel(initial_labels)
    diagnostics: list[dict[str, object]] = []

    while True:
        groups = {
            label: [index for index, value in enumerate(labels) if value == label]
            for label in sorted(set(labels))
        }
        if len(groups) < 2:
            break

        centers = {
            label: robust_center(
                normalized,
                members,
                center_method=center_method,
                trim_ratio=trim_ratio,
            )
            for label, members in groups.items()
        }
        candidates: list[tuple[float, tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...], int, int]] = []
        cluster_ids = sorted(groups)
        for position, left in enumerate(cluster_ids):
            for right in cluster_ids[position + 1 :]:
                similarity = float(centers[left] @ centers[right])
                if similarity < similarity_threshold:
                    continue
                left_key = tuple(tuple(np.round(normalized[index], 8)) for index in groups[left])
                right_key = tuple(tuple(np.round(normalized[index], 8)) for index in groups[right])
                if right_key < left_key:
                    left_key, right_key = right_key, left_key
                candidates.append((-similarity, left_key, right_key, left, right))

        if not candidates:
            break

        _, left_key, right_key, left, right = min(candidates)
        similarity = float(centers[left] @ centers[right])
        left_members = list(groups[left])
        right_members = list(groups[right])
        merged_label = min(left, right)
        labels = [merged_label if value in {left, right} else value for value in labels]
        labels = _stable_relabel(labels)
        diagnostics.append(
            {
                "left_cluster": left,
                "right_cluster": right,
                "left_member_count": len(left_members),
                "right_member_count": len(right_members),
                "merged_member_count": len(left_members) + len(right_members),
                "similarity": similarity,
                "left_member_indices": left_members,
                "right_member_indices": right_members,
            }
        )

    return labels, diagnostics
