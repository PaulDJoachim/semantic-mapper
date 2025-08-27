from clustering.cluster_analyzer import ClusterAnalyzer, ClusteringResult
import numpy as np
from config.config import get_config
from typing import List, Any
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_distances


class HierarchicalAnalyzer(ClusterAnalyzer):
    """Hierarchical clustering analyzer."""

    def __init__(self):
        config = get_config()
        self.min_sample_ratio = config.getfloat("clustering", "min_sample_ratio")
        self.linkage_criterion = config.get("hi-clustering", "linkage_criterion")
        self.cut_distance = config.getfloat("hi-clustering", "cut_distance")

    def analyze_clusters(self, embeddings: np.ndarray) -> ClusteringResult:
        """Cluster embeddings using hierarchical clustering."""
        if len(embeddings) < 2:
            return ClusteringResult([], 0, False, embeddings)

        print(f"\n--- Hierarchical Clustering Analysis (n={len(embeddings)}) ---")

        # Build dendrogram
        distances = pdist(embeddings, metric='cosine')
        linkage_matrix = linkage(distances, method=self.linkage_criterion)

        # Cut dendrogram at selected distance
        labels = fcluster(linkage_matrix, self.cut_distance, criterion='distance')
        labels = labels - 1  # Convert to 0-based indexing

        initial_clusters = len(set(labels))
        print(f"Initial clusters after cut: {initial_clusters}")

        # Filter out small clusters (noise)
        filtered_labels = self._filter_small_clusters(labels)

        # Calculate final statistics
        valid_clusters = set(filtered_labels[filtered_labels >= 0])
        num_clusters = len(valid_clusters)
        noise_points = np.sum(filtered_labels == -1)
        has_branching = num_clusters >= 2

        print(f"Final result: {num_clusters} clusters, {noise_points} noise points")
        print(f"Cluster sizes: {[np.sum(filtered_labels == label) for label in sorted(valid_clusters)]}")
        print(f"Has branching: {has_branching}")

        return ClusteringResult(
            labels=filtered_labels.tolist(),
            num_clusters=num_clusters,
            has_branching=has_branching,
            embeddings=embeddings
        )

    def _filter_small_clusters(self, labels: np.ndarray) -> np.ndarray:
        """Filter out clusters smaller than min_sample_ratio, marking them as noise (-1)."""
        unique_labels = np.unique(labels)
        min_samples = max(2, int(self.min_sample_ratio * len(labels)))

        print(f"Filtering clusters: minimum {min_samples} samples per cluster (ratio={self.min_sample_ratio})")

        # Identify valid clusters by size
        valid_clusters = []
        cluster_info = {}

        for label in unique_labels:
            cluster_size = np.sum(labels == label)
            cluster_info[label] = cluster_size

            if cluster_size >= min_samples:
                valid_clusters.append(label)

        print(f"All cluster sizes: {cluster_info}")
        print(f"Valid clusters: {valid_clusters}")

        # Create new labels: valid clusters get new sequential IDs, small ones become noise (-1)
        filtered_labels = np.full(len(labels), -1, dtype=int)

        for new_id, old_label in enumerate(valid_clusters):
            mask = labels == old_label
            filtered_labels[mask] = new_id

        filtered_points = np.sum(filtered_labels >= 0)
        noise_points = np.sum(filtered_labels == -1)

        print(f"After filtering: {len(valid_clusters)} valid clusters ({filtered_points} points), {noise_points} noise points")

        return filtered_labels

    def get_cluster_representatives(self, items: List[Any],
                                 clustering_result: ClusteringResult) -> List[Any]:
        """Get representative items using centroid proximity."""
        embeddings = clustering_result.embeddings
        labels = clustering_result.labels

        if not items or embeddings is None:
            return []

        # Only get representatives for valid clusters (not noise points)
        valid_labels = [label for label in set(labels) if label >= 0]
        representatives = []

        for label in sorted(valid_labels):
            cluster_indices = [i for i, l in enumerate(labels) if l == label]
            cluster_embeddings = embeddings[cluster_indices]

            # Find item closest to cluster centroid
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = cosine_distances([centroid], cluster_embeddings)[0]
            closest_idx = cluster_indices[np.argmin(distances)]
            representatives.append(items[closest_idx])

        return representatives