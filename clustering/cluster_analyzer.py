from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Any, NamedTuple, Dict
import numpy as np


@dataclass
class ClusteringResult:
    """Clustering analysis results."""
    labels: List[int]
    representatives: Dict[int, List[int]]
    num_clusters: int
    has_branching: bool
    embeddings: np.ndarray

    def update_cluster_representatives(self, items: List[Any]):
        """Find representative items using centroid proximity."""
        embeddings = self.embeddings
        labels = np.array(self.labels)

        # Get unique valid cluster labels (excluding noise points)
        unique_labels = np.unique(labels)
        valid_labels = unique_labels[unique_labels >= 0]

        representatives = {}

        for label in valid_labels:
            cluster_mask = labels == label
            cluster_indices = np.where(cluster_mask)[0]
            cluster_embeddings = embeddings[cluster_mask]

            # Compute centroid and find closest point
            centroid = cluster_embeddings.mean(axis=0)

            # Use dot product for cosine similarity
            similarities = np.dot(cluster_embeddings, centroid)
            closest_local_idx = np.argmax(similarities)
            closest_global_idx = cluster_indices[closest_local_idx]

            representatives[label] = items[closest_global_idx]

        self.representatives = representatives


class ClusterAnalyzer(ABC):
    """Abstract interface for clustering analysis."""

    @abstractmethod
    def analyze_clusters(self, embeddings: np.ndarray) -> ClusteringResult:
        """Cluster embeddings and determine branching."""
        pass

