from clustering.cluster_analyzer import ClusterAnalyzer, ClusteringResult
import numpy as np
from config.config import get_config
from typing import List, Any
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances


class DBSCANClusterAnalyzer(ClusterAnalyzer):
    """DBSCAN-based clustering analyzer."""

    def __init__(self):
        config = get_config()
        self.eps = config.getfloat("DBSCAN-clustering", "eps")
        self.min_sample_ratio = config.getfloat("clustering", "min_sample_ratio")

    def analyze_clusters(self, embeddings: np.ndarray) -> ClusteringResult:
        """Cluster embeddings using DBSCAN."""
        total_stems = len(embeddings)
        if total_stems == 0:
            return ClusteringResult([], 0, False, embeddings)

        min_samples = max(2, int(self.min_sample_ratio * total_stems))

        clustering = DBSCAN(eps=self.eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        has_branching = num_clusters >= 2

        return ClusteringResult(
            labels=labels.tolist(),
            num_clusters=num_clusters,
            has_branching=has_branching,
            embeddings=embeddings
        )

    def get_cluster_representatives(self, items: List[Any],
                                    clustering_result: ClusteringResult) -> List[Any]:
        """Get representative items using embedding distances."""
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
            similarities = np.dot(cluster_embeddings, centroid)
            closest_idx = cluster_indices[np.argmax(similarities)]
            representatives.append(items[closest_idx])

        return representatives