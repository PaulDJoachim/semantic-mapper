from clustering.cluster_analyzer import ClusterAnalyzer, ClusteringResult
import numpy as np
from typing import List, Any


class MockClusterAnalyzer(ClusterAnalyzer):
    """Mock analyzer for testing."""

    def __init__(self, mode: str = "simple", seed: int = None):
        self.mode = mode
        if seed is not None:
            np.random.seed(seed)

    def analyze_clusters(self, embeddings: np.ndarray) -> ClusteringResult:
        """Generate mock clustering results."""
        if len(embeddings) == 0:
            return ClusteringResult([], 0, False, embeddings)

        if self.mode == "simple":
            return self._simple_clustering(embeddings)
        elif self.mode == "alternating":
            return self._alternating_clustering(embeddings)
        elif self.mode == "random":
            return self._random_clustering(embeddings)
        else:  # no_clusters
            return ClusteringResult([0] * len(embeddings), 1, False, embeddings)

    def get_cluster_representatives(self, items: List[Any], clustering_result: ClusteringResult) -> List[Any]:
        """Select first item from each cluster."""
        if not items:
            return []

        labels = clustering_result.labels
        cluster_reps = {}

        for i, label in enumerate(labels):
            if label not in cluster_reps and label != -1:
                cluster_reps[label] = items[i]

        return list(cluster_reps.values())

    def _simple_clustering(self, embeddings: np.ndarray) -> ClusteringResult:
        """Split into two clusters based on position."""
        mid = len(embeddings) // 2
        labels = [0] * mid + [1] * (len(embeddings) - mid)
        return ClusteringResult(labels, 2, True, embeddings)

    def _alternating_clustering(self, embeddings: np.ndarray) -> ClusteringResult:
        """Alternate between 2-3 clusters."""
        num_clusters = min(3, max(2, len(embeddings) // 8))
        labels = [i % num_clusters for i in range(len(embeddings))]
        return ClusteringResult(labels, num_clusters, num_clusters >= 2, embeddings)

    def _random_clustering(self, embeddings: np.ndarray) -> ClusteringResult:
        """Random clustering with 2-4 clusters."""
        num_clusters = np.random.randint(2, min(4, len(embeddings) // 5))
        labels = [np.random.randint(0, num_clusters) for _ in range(len(embeddings))]
        return ClusteringResult(labels, num_clusters, num_clusters >= 2, embeddings)