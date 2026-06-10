"""
Gradient vector clustering for redundancy removal in Phase 1.

Clusters samples based on their influence pattern similarity.
Samples with near-identical influence vectors are likely redundant.
"""

import logging
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger("dataprism.utils.cluster")


def cluster_by_similarity(
    vectors: np.ndarray,
    threshold: float = 0.85,
    method: str = "kmeans",
    num_clusters: Optional[int] = None,
    random_state: int = 42,
) -> tuple[list[int], np.ndarray]:
    """Cluster vectors and return centroid indices.

    Args:
        vectors: (n_samples, n_features) array of influence vectors.
        threshold: Similarity threshold for redundancy detection.
        method: Clustering method ('kmeans', 'agglomerative').
        num_clusters: Number of clusters (auto-determined if None).
        random_state: Random seed.

    Returns:
        Tuple of (centroid_indices, cluster_assignments).
        centroid_indices: List of indices of samples chosen as representatives.
        cluster_assignments: (n_samples,) array of cluster IDs.
    """
    n_samples = vectors.shape[0]

    if n_samples <= 1:
        return list(range(n_samples)), np.zeros(n_samples, dtype=int)

    # Auto-determine number of clusters
    if num_clusters is None:
        # Heuristic: sqrt(n) clusters, capped at n/2
        num_clusters = max(1, min(int(np.sqrt(n_samples)), n_samples // 2))

    logger.info(
        "Clustering %d samples into %d clusters (method=%s, threshold=%.2f)",
        n_samples, num_clusters, method, threshold,
    )

    if method == "kmeans":
        return _kmeans_cluster(vectors, num_clusters, random_state)
    elif method == "agglomerative":
        return _agglomerative_cluster(vectors, threshold, num_clusters)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def _kmeans_cluster(
    vectors: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> tuple[list[int], np.ndarray]:
    """K-means clustering with centroid selection by proximity to cluster center."""
    from sklearn.cluster import KMeans

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    cluster_ids = kmeans.fit_predict(vectors)

    # For each cluster, select the sample closest to the cluster center
    centroid_indices = []
    for c in range(n_clusters):
        cluster_mask = cluster_ids == c
        cluster_vectors = vectors[cluster_mask]
        cluster_original_indices = np.where(cluster_mask)[0]

        # Find sample closest to cluster center
        center = cluster_vectors.mean(axis=0)
        distances = cdist([center], cluster_vectors, metric="cosine")[0]
        closest_idx = cluster_original_indices[np.argmin(distances)]
        centroid_indices.append(int(closest_idx))

    logger.info(
        "K-means: %d clusters, %d centroids selected",
        n_clusters, len(centroid_indices),
    )
    return centroid_indices, cluster_ids


def _agglomerative_cluster(
    vectors: np.ndarray,
    threshold: float,
    n_clusters: Optional[int],
) -> tuple[list[int], np.ndarray]:
    """Agglomerative clustering based on cosine distance threshold."""
    from sklearn.cluster import AgglomerativeClustering

    # Convert cosine similarity threshold to distance threshold
    # cosine_distance = 1 - cosine_similarity
    distance_threshold = 1.0 - threshold

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
        distance_threshold=None if n_clusters is not None else distance_threshold,
    )
    cluster_ids = clustering.fit_predict(vectors)

    actual_n = len(set(cluster_ids))
    centroid_indices = []

    for c in range(actual_n):
        cluster_mask = cluster_ids == c
        cluster_vectors = vectors[cluster_mask]
        cluster_original_indices = np.where(cluster_mask)[0]

        if len(cluster_vectors) == 1:
            centroid_indices.append(int(cluster_original_indices[0]))
        else:
            center = cluster_vectors.mean(axis=0)
            distances = cdist([center], cluster_vectors, metric="cosine")[0]
            closest_idx = cluster_original_indices[np.argmin(distances)]
            centroid_indices.append(int(closest_idx))

    logger.info(
        "Agglomerative: %d clusters, %d centroids selected",
        actual_n, len(centroid_indices),
    )
    return centroid_indices, cluster_ids


def find_redundant_samples(
    vectors: np.ndarray,
    threshold: float = 0.85,
) -> tuple[list[int], list[int]]:
    """Identify redundant samples based on pairwise cosine similarity.

    A sample is redundant if its cosine similarity with another sample
    exceeds the threshold. Keeps the first sample in each redundant pair.

    Args:
        vectors: (n_samples, n_features) array.
        threshold: Cosine similarity threshold for redundancy.

    Returns:
        Tuple of (keep_indices, redundant_indices).
    """
    n = vectors.shape[0]
    if n <= 1:
        return list(range(n)), []

    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms

    # Compute cosine similarity matrix
    sim_matrix = normalized @ normalized.T

    # Find redundant pairs (upper triangle only)
    redundant = set()
    for i in range(n):
        if i in redundant:
            continue
        for j in range(i + 1, n):
            if j in redundant:
                continue
            if sim_matrix[i, j] >= threshold:
                redundant.add(j)  # Mark j as redundant (keep i)

    keep = [i for i in range(n) if i not in redundant]
    logger.info(
        "Redundancy analysis: %d/%d samples redundant (threshold=%.2f)",
        len(redundant), n, threshold,
    )
    return keep, list(redundant)
