from clustering.cluster_analyzer import ClusterAnalyzer, ClusteringResult
import numpy as np
from config.config import get_config
from sklearn.cluster import DBSCAN


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
            representatives={},
            num_clusters=num_clusters,
            has_branching=has_branching,
            embeddings=embeddings
        )
