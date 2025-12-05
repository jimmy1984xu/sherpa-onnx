#!/usr/bin/env python3

"""
Multi-round progressive clustering algorithm for speaker identification.

This algorithm uses multiple rounds with decreasing thresholds to gradually
cluster speaker segments. Short segments (< 3 seconds) are excluded initially.

Algorithm flow:
1. Filter out short segments (< min_duration_seconds)
2. Round 1: Use high threshold (e.g., 0.9), cluster segments
3. Round 2: Use medium threshold (e.g., 0.8), cluster remaining segments
4. Round 3: Use lower threshold (e.g., 0.7), cluster remaining segments
5. Update speaker embeddings as average of all segments in the cluster
6. Assign short segments to nearest speaker cluster
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    """Normalize embedding vector to unit length."""
    norm = np.linalg.norm(emb)
    if norm == 0.0:
        return emb
    return emb / norm


class MultiRoundClustering:
    """Multi-round progressive clustering algorithm."""
    
    def __init__(
        self,
        num_rounds: int = 3,
        start_threshold: float = 0.9,
        threshold_step: float = 0.1,
        min_duration_seconds: float = 3.0,
    ):
        """
        Initialize multi-round clustering.
        
        Args:
            num_rounds: Number of clustering rounds
            start_threshold: Starting similarity threshold (first round)
            threshold_step: Threshold decrease per round
            min_duration_seconds: Minimum segment duration to participate in initial clustering
        """
        self.num_rounds = num_rounds
        self.start_threshold = start_threshold
        self.threshold_step = threshold_step
        self.min_duration_seconds = min_duration_seconds
    
    def cluster(
        self,
        embeddings: np.ndarray,
        durations_ms: np.ndarray,
    ) -> Tuple[List[int], List[float]]:
        """
        Perform multi-round clustering.
        
        Args:
            embeddings: Array of embeddings (num_segments x embedding_dim)
            durations_ms: Array of segment durations in milliseconds
        
        Returns:
            Tuple of (cluster_labels, similarities)
            - cluster_labels: List of cluster IDs for each segment
            - similarities: List of similarity scores (to previous segment or cluster centroid)
        """
        num_segments = len(embeddings)
        if num_segments == 0:
            return [], []
        
        # Normalize all embeddings
        normalized_embeddings = np.array([normalize_embedding(emb) for emb in embeddings])
        
        # Convert durations from milliseconds to seconds
        durations_seconds = durations_ms / 1000.0
        
        # Initialize: all segments unassigned
        cluster_labels = [-1] * num_segments
        similarities = [0.0] * num_segments
        speaker_embeddings: Dict[int, np.ndarray] = {}  # speaker_id -> average embedding
        speaker_counts: Dict[int, int] = {}  # speaker_id -> segment count
        next_speaker_id = 0
        
        # Separate long and short segments
        long_segment_indices = [
            i for i in range(num_segments)
            if durations_seconds[i] >= self.min_duration_seconds
        ]
        short_segment_indices = [
            i for i in range(num_segments)
            if durations_seconds[i] < self.min_duration_seconds
        ]
        
        # Multi-round clustering for long segments
        unassigned_indices = long_segment_indices.copy()
        
        for round_num in range(self.num_rounds):
            threshold = self.start_threshold - round_num * self.threshold_step
            
            if not unassigned_indices:
                break
            
            print(f"      Round {round_num + 1}: threshold={threshold:.2f}, "
                  f"unassigned segments={len(unassigned_indices)}")
            
            # Try to assign unassigned segments to existing speakers
            newly_assigned = []
            
            for idx in unassigned_indices:
                emb = normalized_embeddings[idx]
                best_speaker_id = -1
                best_similarity = -1.0
                
                # Compare with existing speaker centroids
                for speaker_id, speaker_emb in speaker_embeddings.items():
                    sim = cosine_similarity(emb, speaker_emb)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_speaker_id = speaker_id
                
                # If similarity meets threshold, assign to speaker
                if best_similarity >= threshold:
                    cluster_labels[idx] = best_speaker_id
                    similarities[idx] = best_similarity
                    newly_assigned.append(idx)
                    
                    # Update speaker embedding (running average)
                    count = speaker_counts[best_speaker_id]
                    speaker_embeddings[best_speaker_id] = (
                        (speaker_embeddings[best_speaker_id] * count + emb) / (count + 1)
                    )
                    speaker_embeddings[best_speaker_id] = normalize_embedding(
                        speaker_embeddings[best_speaker_id]
                    )
                    speaker_counts[best_speaker_id] += 1
                else:
                    # Create new speaker
                    cluster_labels[idx] = next_speaker_id
                    similarities[idx] = 1.0  # Perfect similarity to itself
                    speaker_embeddings[next_speaker_id] = emb.copy()
                    speaker_counts[next_speaker_id] = 1
                    newly_assigned.append(idx)
                    next_speaker_id += 1
            
            # Remove newly assigned segments from unassigned list
            unassigned_indices = [i for i in unassigned_indices if i not in newly_assigned]
            
            # If no new assignments in this round, break early
            if not newly_assigned:
                print(f"      No new assignments in round {round_num + 1}, stopping")
                break
        
        # Assign remaining unassigned long segments to nearest speaker
        for idx in unassigned_indices:
            emb = normalized_embeddings[idx]
            best_speaker_id = -1
            best_similarity = -1.0
            
            if speaker_embeddings:
                for speaker_id, speaker_emb in speaker_embeddings.items():
                    sim = cosine_similarity(emb, speaker_emb)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_speaker_id = speaker_id
                
                cluster_labels[idx] = best_speaker_id
                similarities[idx] = best_similarity
                
                # Update speaker embedding
                count = speaker_counts[best_speaker_id]
                speaker_embeddings[best_speaker_id] = (
                    (speaker_embeddings[best_speaker_id] * count + emb) / (count + 1)
                )
                speaker_embeddings[best_speaker_id] = normalize_embedding(
                    speaker_embeddings[best_speaker_id]
                )
                speaker_counts[best_speaker_id] += 1
            else:
                # No existing speakers, create new one
                cluster_labels[idx] = next_speaker_id
                similarities[idx] = 1.0
                speaker_embeddings[next_speaker_id] = emb.copy()
                speaker_counts[next_speaker_id] = 1
                next_speaker_id += 1
        
        # Assign short segments: compare with all speakers, if similarity > threshold assign to that speaker,
        # otherwise assign to the previous segment's speaker
        # Use the last round's threshold as the comparison threshold
        short_segment_threshold = self.start_threshold - (self.num_rounds - 1) * self.threshold_step
        
        # Sort short segment indices to process in order
        short_segment_indices_sorted = sorted(short_segment_indices)
        
        for idx in short_segment_indices_sorted:
            emb = normalized_embeddings[idx]
            best_speaker_id = -1
            best_similarity = -1.0
            
            if speaker_embeddings:
                # Compare with all existing speakers
                for speaker_id, speaker_emb in speaker_embeddings.items():
                    sim = cosine_similarity(emb, speaker_emb)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_speaker_id = speaker_id
                
                # If similarity > threshold, assign to that speaker
                if best_similarity >= short_segment_threshold:
                    cluster_labels[idx] = best_speaker_id
                    similarities[idx] = best_similarity
                else:
                    # Otherwise, assign to previous segment's speaker
                    # Find the previous segment (the last assigned segment before this one)
                    prev_speaker_id = -1
                    for prev_idx in range(idx - 1, -1, -1):
                        if cluster_labels[prev_idx] >= 0:
                            prev_speaker_id = cluster_labels[prev_idx]
                            break
                    
                    if prev_speaker_id >= 0:
                        cluster_labels[idx] = prev_speaker_id
                        # Calculate similarity to previous segment's speaker
                        if prev_speaker_id in speaker_embeddings:
                            similarities[idx] = cosine_similarity(emb, speaker_embeddings[prev_speaker_id])
                        else:
                            similarities[idx] = best_similarity  # Fallback to best similarity
                    else:
                        # No previous segment found, use best match
                        cluster_labels[idx] = best_speaker_id
                        similarities[idx] = best_similarity
            else:
                # No existing speakers, create new one
                cluster_labels[idx] = next_speaker_id
                similarities[idx] = 1.0
                speaker_embeddings[next_speaker_id] = emb.copy()
                speaker_counts[next_speaker_id] = 1
                next_speaker_id += 1
        
        # Remap speaker IDs to be sequential starting from 0
        unique_speakers = sorted(set(cluster_labels))
        speaker_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_speakers)}
        cluster_labels = [speaker_id_map[label] for label in cluster_labels]
        
        return cluster_labels, similarities

