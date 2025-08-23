from typing import List, Tuple, Any
from abc import ABC, abstractmethod
from config.config import get_config


class ModelInterface(ABC):
    """Abstract interface for language models used in divergent generation."""

    @abstractmethod
    def encode(self, text: str) -> Any:
        """Encode text to token representation."""
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        pass

    @abstractmethod
    def generate_stems(self, input_ids: Any, num_stems: int, stem_length: int,
                      temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> List[Tuple[Any, Any]]:
        """Generate multiple continuation stems and their representations."""
        pass


def create_model(model_type: str = None, **kwargs) -> ModelInterface:
    """Create model instance of specified type."""
    config = get_config()
    model_type = model_type or config.get("model", "type", "mock")

    if model_type == "mock":
        from models.mock_model import MockModel
        return MockModel(**kwargs)

    elif model_type == "gpt2":
        from gpt_two import GPT2Interface
        return GPT2Interface(**kwargs)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_generator(model_type: str = None, **kwargs):
    """Create DivergentGenerator with specified components."""
    from divergent import DivergentGenerator
    from semantic_embedding.mock_embedding import MockEmbeddingProvider
    from semantic_embedding.sentence_embedding import SentenceEmbeddingProvider
    from clustering.mock_clustering import MockClusterAnalyzer
    from clustering.dbscan_clustering import DBSCANClusterAnalyzer

    config = get_config()
    analyzer_type = kwargs.pop('analyzer_type', None) or config.get("embeddings", "type", "sentence")

    model = create_model(model_type, **kwargs.get('model_kwargs', {}))

    if analyzer_type == "mock":
        embedding_provider = MockEmbeddingProvider(**kwargs.get('embedding_kwargs', {}))
        cluster_analyzer = MockClusterAnalyzer(**kwargs.get('cluster_kwargs', {}))
    elif analyzer_type == "sentence":
        try:
            embedding_provider = SentenceEmbeddingProvider(**kwargs.get('embedding_kwargs', {}))
            cluster_analyzer = DBSCANClusterAnalyzer(**kwargs.get('cluster_kwargs', {}))
        except ImportError as e:
            print(f"SentenceEmbedding not available: {e}, falling back to mock")
            embedding_provider = MockEmbeddingProvider(**kwargs.get('embedding_kwargs', {}))
            cluster_analyzer = MockClusterAnalyzer(**kwargs.get('cluster_kwargs', {}))
    else:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")

    return DivergentGenerator(
        model_interface=model,
        embedding_provider=embedding_provider,
        cluster_analyzer=cluster_analyzer
    )