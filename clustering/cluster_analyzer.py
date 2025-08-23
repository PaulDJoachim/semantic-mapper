from abc import ABC, abstractmethod
from typing import List, Any, NamedTuple
import numpy as np
from config.config import get_config


class ClusteringResult(NamedTuple):
    """Clustering analysis results."""
    labels: List[int]
    num_clusters: int
    has_branching: bool


class ClusterAnalyzer(ABC):
    """Abstract interface for clustering analysis."""

    @abstractmethod
    def analyze_clusters(self, embeddings: np.ndarray) -> ClusteringResult:
        """Cluster embeddings and determine branching."""
        pass

    @abstractmethod
    def get_cluster_representatives(self, items: List[Any], clustering_result: ClusteringResult,
                                  embeddings: np.ndarray = None) -> List[Any]:
        """Get representative items for each cluster."""
        pass
