from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Any, Optional
import numpy as np
from utilities.stem import StemPack


@dataclass
class Cluster:
    """Cluster object containing cluster label, mask for cluster member positions, and reference to stem pack of cluster."""
    label: int
    cluster_mask: np.ndarray  # mask for cluster member positions
    stem_pack: StemPack  # reference to stem pack of cluster
    size: int  # Number of token sequences in cluster

    # TODO Representative object? Move these to TreeNode?
    representative_sequence: Optional[List[int]] = None  # tokens
    representative_semantic_embedding: np.ndarray = None
    representative_trajectory_embedding: np.ndarray = None  # Assigned to TreeNode
    representative_entropy: Optional[float] = None

    def get_cluster_attr(self, attr_name: str):
        return self.stem_pack.get_cluster_attr(attr_name, self.cluster_mask)


class ClusterAnalyzer(ABC):
    """Abstract interface for clustering analysis."""

    @abstractmethod
    def analyze_clusters(self, embeddings: np.ndarray,
                         delta_embeddings: np.ndarray,
                         token_sequences: np.ndarray,
                         token_mask: np.ndarray,
                         text_sequences: List[str]) -> List[Cluster]:
        """Cluster embeddings and determine branching."""
        pass
