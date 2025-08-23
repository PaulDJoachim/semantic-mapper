from clustering.cluster_analyzer import ClusterAnalyzer, ClusteringResult
import numpy as np
from config.config import get_config
from typing import List, Any

class DBSCANClusterAnalyzer(ClusterAnalyzer):
    """DBSCAN-based clustering analyzer."""

    def __init__(self, eps: float = None, min_sample_ratio: float = None, min_clusters: int = None):
        config = get_config()
        self.eps = eps if eps is not None else config.getfloat("clustering", "eps", 0.20)
        self.min_sample_ratio = (min_sample_ratio if min_sample_ratio is not None 
                               else config.getfloat("clustering", "min_sample_ratio", 0.15))
        self.min_clusters = (min_clusters if min_clusters is not None 
                           else config.getint("clustering", "min_clusters", 2))

    def analyze_clusters(self, embeddings: np.ndarray) -> ClusteringResult:
        """Cluster embeddings using DBSCAN."""
        from sklearn.cluster import DBSCAN
        from sklearn.metrics.pairwise import cosine_distances

        if len(embeddings) == 0:
            return ClusteringResult([], 0, False)

        # Calculate minimum samples based on total stems generated
        config = get_config()
        total_stems = config.getint("generation", "num_stems", 50)
        min_samples = max(2, int(self.min_sample_ratio * total_stems))

        # Use cosine distance for semantic similarity
        clustering = DBSCAN(eps=self.eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        has_branching = num_clusters >= self.min_clusters

        return ClusteringResult(labels.tolist(), num_clusters, has_branching)

    def get_cluster_representatives(self, items: List[Any], clustering_result: ClusteringResult,
                                  embeddings: np.ndarray = None) -> List[Any]:
        """Get representative items using embedding distances."""
        from sklearn.metrics.pairwise import cosine_distances

        if embeddings is None:
            raise ValueError("embeddings required for DBSCAN representative selection")

        labels = clustering_result.labels
        unique_labels = set(label for label in labels if label != -1)

        if not unique_labels:
            return [items[0]] if items else []

        representatives = []

        for cluster_id in unique_labels:
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
            cluster_embeddings = embeddings[cluster_indices]

            # Find centroid and closest item
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = cosine_distances([centroid], cluster_embeddings)[0]
            closest_idx = cluster_indices[np.argmin(distances)]
            representatives.append(items[closest_idx])

        return representatives