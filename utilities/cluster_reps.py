from typing import List
from clustering.cluster_analyzer import Cluster
import numpy as np


def assign_cluster_representatives(clusters: List[Cluster]) -> None:
    """Find representative sequences using centroid-based selection."""
    if not clusters:
        return

    for cluster in clusters:
        if cluster.label == -1:
            continue  # skip noise cluster

        delta_embeddings = cluster.get_cluster_attr("delta_embeddings")
        token_sequences = cluster.get_cluster_attr("tokens")
        semantic_embeddings = cluster.get_cluster_attr("embeddings")

        # Calculate cluster centroid
        cluster_centroid = delta_embeddings.mean(axis=0)

        # Find sequence with highest similarity to centroid
        similarities = np.dot(delta_embeddings, cluster_centroid)
        best_idx = np.argmax(similarities)

        cluster.representative_sequence = token_sequences[best_idx]
        cluster.representative_semantic_embedding = semantic_embeddings[best_idx]
        cluster.representative_trajectory_embedding = delta_embeddings[best_idx]
        cluster.representative_entropy = cluster.get_cluster_attr("entropy_sums")[best_idx]
