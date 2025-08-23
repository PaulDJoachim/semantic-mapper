from abc import ABC, abstractmethod
from typing import List, NamedTuple, Any, Optional
from config.config import get_config


class ClusteringResult(NamedTuple):
    """Clustering analysis results."""
    labels: List[int]
    num_clusters: int
    has_branching: bool


class EmbeddingAnalyzer(ABC):
    """Abstract interface for semantic clustering."""

    @abstractmethod
    def cluster_stems(self, stem_texts: List[str]) -> ClusteringResult:
        """Cluster stems by semantic similarity."""
        pass

    @abstractmethod
    def get_cluster_representatives(self, stem_tokens: List[Any], 
                                  clustering_result: ClusteringResult) -> List[Any]:
        """Get representative stems for each cluster."""
        pass


def create_analyzer(analyzer_type: Optional[str] = None, **kwargs) -> EmbeddingAnalyzer:
    """Factory for creating embedding analyzers."""
    config = get_config()
    analyzer_type = analyzer_type or config.get("embeddings", "type", "sentence")

    if analyzer_type == "mock":
        from clustering.mock_embedding import MockEmbeddingAnalyzer
        return MockEmbeddingAnalyzer(**kwargs)

    elif analyzer_type == "sentence":
        try:
            from sentence_embedding import SentenceEmbeddingAnalyzer
            return SentenceEmbeddingAnalyzer(**kwargs)
        except ImportError as e:
            print(f"SentenceEmbeddingAnalyzer not available: {e}")
            print("Falling back to MockEmbeddingAnalyzer")
            from clustering.mock_embedding import MockEmbeddingAnalyzer
            return MockEmbeddingAnalyzer(**kwargs)

    else:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")


def create_generator(**kwargs):
    """Create DivergentGenerator with appropriate model and analyzer."""
    from models.model_interface import create_model
    from divergent import DivergentGenerator

    model_type = kwargs.pop('model_type', None)
    analyzer_type = kwargs.pop('analyzer_type', None)

    model = create_model(model_type=model_type, **kwargs.get('model_kwargs', {}))
    analyzer = create_analyzer(analyzer_type=analyzer_type, **kwargs.get('analyzer_kwargs', {}))

    return DivergentGenerator(model_interface=model, analyzer=analyzer)