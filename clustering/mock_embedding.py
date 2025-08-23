import random
from typing import List, Any
from clustering.embedding_analyzer import EmbeddingAnalyzer, ClusteringResult


class MockEmbeddingAnalyzer(EmbeddingAnalyzer):
    """Mock analyzer for testing without dependencies."""

    def __init__(self, mode: str = "simple", seed: int = None):
        """
        Args:
            mode: 'simple', 'alternating', 'random', 'no_clusters'
        """
        self.mode = mode
        if seed is not None:
            random.seed(seed)

    def cluster_stems(self, stem_texts: List[str]) -> ClusteringResult:
        """Generate mock clustering results."""
        if not stem_texts:
            return ClusteringResult([], 0, False)

        if self.mode == "simple":
            return self._simple_clustering(stem_texts)
        elif self.mode == "alternating":
            return self._alternating_clustering(stem_texts)
        elif self.mode == "random":
            return self._random_clustering(stem_texts)
        else:  # no_clusters
            return ClusteringResult([0] * len(stem_texts), 1, False)

    def _simple_clustering(self, stem_texts: List[str]) -> ClusteringResult:
        """Split into two clusters based on position."""
        mid = len(stem_texts) // 2
        labels = [0] * mid + [1] * (len(stem_texts) - mid)
        return ClusteringResult(labels, 2, True)

    def _alternating_clustering(self, stem_texts: List[str]) -> ClusteringResult:
        """Alternate between 2-3 clusters."""
        num_clusters = min(3, max(2, len(stem_texts) // 8))
        labels = [i % num_clusters for i in range(len(stem_texts))]
        return ClusteringResult(labels, num_clusters, num_clusters >= 2)

    def _random_clustering(self, stem_texts: List[str]) -> ClusteringResult:
        """Random clustering with 2-4 clusters."""
        num_clusters = random.randint(2, min(4, len(stem_texts) // 5))
        labels = [random.randint(0, num_clusters - 1) for _ in stem_texts]
        return ClusteringResult(labels, num_clusters, num_clusters >= 2)

    def get_cluster_representatives(self, stem_tokens: List[Any],
                                  clustering_result: ClusteringResult) -> List[Any]:
        """Select first token from each cluster."""
        if not stem_tokens:
            return []

        labels = clustering_result.labels
        cluster_reps = {}

        for i, label in enumerate(labels):
            if label not in cluster_reps and label != -1:
                cluster_reps[label] = stem_tokens[i]

        return list(cluster_reps.values())