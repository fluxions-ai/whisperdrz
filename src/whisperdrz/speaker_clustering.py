"""
Speaker clustering utilities for cross-chunk speaker ID assignment.

Uses GPU-accelerated agglomerative hierarchical clustering on speaker embeddings
extracted from the model's speaker_head during transcription.

Supports two clustering methods:
- AHC (default): Simple agglomerative hierarchical clustering
- VBx: Variational Bayes refinement for better probabilistic speaker assignment
"""

import re

import torch



def cluster_speakers(
    embeddings: list[tuple[int, int, torch.Tensor]],
    threshold: float = 0.6,
    min_cluster_size: int = 1,
) -> dict[tuple[int, int], int]:
    """Cluster speaker embeddings using GPU-accelerated agglomerative clustering.

    Takes speaker embeddings from multiple chunks (where each chunk has local
    speaker IDs 0, 1, 2...) and clusters them to assign global speaker IDs.

    Uses GPU for distance computation, equivalent to scipy's centroid linkage.

    Args:
        embeddings: List of (segment_idx, local_speaker_id, embedding) tuples.
            Embeddings should be L2-normalized tensors.
        threshold: Distance threshold for cutting dendrogram. Lower = more clusters.
            Recommended range: 0.2-0.5 for speaker embeddings.
        min_cluster_size: Minimum cluster size. Smaller clusters get reassigned
            to the nearest large cluster centroid.

    Returns:
        Dictionary mapping (segment_idx, local_speaker_id) -> global_speaker_id
    """
    if len(embeddings) == 0:
        return {}

    if len(embeddings) == 1:
        seg_idx, local_id, _ = embeddings[0]
        return {(seg_idx, local_id): 0}

    # Stack embeddings (keep on CPU - clustering is fast enough and avoids VRAM pressure)
    emb_tensor = torch.stack([e[2] for e in embeddings]).cpu()
    device = emb_tensor.device

    # L2-normalize
    emb_tensor = torch.nn.functional.normalize(emb_tensor, p=2, dim=-1)
    n = emb_tensor.shape[0]

    # GPU-accelerated agglomerative clustering (centroid linkage)
    # Each point starts as its own cluster
    clusters = torch.arange(n, device=device)
    centroids = emb_tensor.clone()  # [n, dim]
    cluster_sizes = torch.ones(n, device=device)
    active = torch.ones(n, dtype=torch.bool, device=device)

    while active.sum() > 1:
        # Compute pairwise distances between active centroids
        active_idx = torch.where(active)[0]
        active_centroids = centroids[active_idx]

        # Euclidean distance matrix (for L2-normalized vectors, equiv to sqrt(2 - 2*cosine))
        # cdist doesn't support half precision, cast to float32
        dists = torch.cdist(active_centroids.float(), active_centroids.float())
        # Set diagonal to inf to avoid self-merging
        dists.fill_diagonal_(float("inf"))

        # Find minimum distance
        min_val = dists.min()
        if min_val > threshold:
            break

        # Find pair to merge
        min_idx = (dists == min_val).nonzero()[0]
        i, j = active_idx[min_idx[0]], active_idx[min_idx[1]]
        if i > j:
            i, j = j, i  # Ensure i < j

        # Merge j into i (centroid linkage: weighted average)
        size_i, size_j = cluster_sizes[i], cluster_sizes[j]
        centroids[i] = (centroids[i] * size_i + centroids[j] * size_j) / (
            size_i + size_j
        )
        centroids[i] = torch.nn.functional.normalize(centroids[i], p=2, dim=0)
        cluster_sizes[i] = size_i + size_j

        # Update cluster assignments
        clusters[clusters == j] = i
        active[j] = False

    # Renumber clusters to 0, 1, 2, ...
    unique_clusters = clusters.unique()
    cluster_map = {int(c): idx for idx, c in enumerate(unique_clusters.tolist())}
    final_clusters = torch.tensor(
        [cluster_map[int(c)] for c in clusters.tolist()], device=device
    )

    # Handle min_cluster_size (reassign small clusters to nearest large)
    min_cluster_size = min(min_cluster_size, max(1, round(0.1 * n)))
    unique, counts = final_clusters.unique(return_counts=True)
    large_mask = counts >= min_cluster_size
    large_clusters = unique[large_mask]
    small_clusters = unique[~large_mask]

    if len(large_clusters) == 0:
        final_clusters[:] = 0
    elif len(small_clusters) > 0:
        # Compute centroids for large and small clusters
        large_centroids = torch.stack(
            [emb_tensor[final_clusters == c].mean(0) for c in large_clusters]
        )
        large_centroids = torch.nn.functional.normalize(large_centroids, p=2, dim=-1)

        for sc in small_clusters:
            sc_centroid = emb_tensor[final_clusters == sc].mean(0, keepdim=True)
            sc_centroid = torch.nn.functional.normalize(sc_centroid, p=2, dim=-1)
            dists = torch.cdist(sc_centroid.float(), large_centroids.float())
            nearest = large_clusters[dists.argmin()]
            final_clusters[final_clusters == sc] = nearest

        # Renumber again
        unique_clusters = final_clusters.unique()
        cluster_map = {int(c): idx for idx, c in enumerate(unique_clusters.tolist())}
        final_clusters = torch.tensor(
            [cluster_map[int(c)] for c in final_clusters.tolist()]
        )

    # Build mapping
    mapping = {}
    for i, (seg_idx, local_id, _) in enumerate(embeddings):
        mapping[(seg_idx, local_id)] = int(final_clusters[i])

    return mapping


def normalize_speaker_ids(text: str) -> str:
    """Renumber speaker IDs to 0, 1, 2... by order of first appearance."""
    pattern = re.compile(r"\[(\d+)\]")
    seen = {}
    next_id = 0

    def replacer(match):
        nonlocal next_id
        old_id = match.group(1)
        if old_id not in seen:
            seen[old_id] = next_id
            next_id += 1
        return f"[{seen[old_id]}]"

    return pattern.sub(replacer, text)

