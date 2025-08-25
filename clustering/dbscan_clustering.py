from clustering.cluster_analyzer import ClusterAnalyzer, ClusteringResult
import numpy as np
from config.config import get_config
from typing import List, Any
from sklearn.cluster import DBSCAN


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
        if len(embeddings) == 0:
            return ClusteringResult([], 0, False, embeddings)

        config = get_config()
        total_stems = config.getint("generation", "num_stems", 50)
        min_samples = max(2, int(self.min_sample_ratio * total_stems))

        clustering = DBSCAN(eps=self.eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        has_branching = num_clusters >= self.min_clusters

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
        if embeddings is None:
            raise ValueError("embeddings required for representative selection")

        labels = clustering_result.labels
        unique_labels = [label for label in set(labels) if label != -1]

        if not unique_labels:
            return [items[0]] if items else []

        representatives = []

        try:
            from sklearn.metrics.pairwise import cosine_distances
            
            for cluster_id in unique_labels:
                cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
                cluster_embeddings = embeddings[cluster_indices]

                centroid = np.mean(cluster_embeddings, axis=0)
                distances = cosine_distances([centroid], cluster_embeddings)[0]
                closest_idx = cluster_indices[np.argmin(distances)]
                representatives.append(items[closest_idx])

        except ImportError:
            for cluster_id in unique_labels:
                cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
                representatives.append(items[cluster_indices[0]])

        return representatives