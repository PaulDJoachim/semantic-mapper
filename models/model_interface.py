from typing import List, Tuple, Any
from abc import ABC, abstractmethod
from config.config import get_config


class ModelInterface(ABC):
    """Abstract interface for language models used in divergent generation."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Return list of token IDs"""
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode list of token IDs to text"""
        pass

    @abstractmethod
    def generate_stems(self, input_ids: Any, num_stems: int, stem_length: int,
                      temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> List[List[int]]:
        """Return stems as lists of token IDs"""
        pass


def get_model(model_name: str = None, **kwargs) -> ModelInterface:
    """Create model instance of specified type."""
    model_name = model_name

    try:
        from models.generic_transformer import GenericTransformer
        return GenericTransformer(model_name, **kwargs)

    except ValueError:
        raise ValueError(f"Unknown model: {model_name}")


def get_embedder(embedding_model: str = None, **kwargs):
    """Create embedding model of specified type."""

    try:
        from semantic_embedding.sentence_embedding import SentenceEmbeddingProvider
        embedding_provider = SentenceEmbeddingProvider(model_name=embedding_model, **kwargs.get('embedding_kwargs', {}))
    except ValueError:
        raise ValueError(f"Unknown embedding model: {embedding_model}")

    return embedding_provider


def get_grouper(clustering_type: str = None, **kwargs):
    cluster_kwargs = kwargs.get('cluster_kwargs', {})

    if clustering_type == "hierarchical":
        from clustering.hierarchical_clustering import HierarchicalAnalyzer
        return HierarchicalAnalyzer(**cluster_kwargs)
    else:
        raise ValueError(f"Unknown clustering type: {clustering_type}")


def create_generator(inference_model: str, embedding_model: str, cluster_type: str, **kwargs):
    """Create DivergentGenerator with specified components."""
    from divergent import DivergentGenerator

    inference_model = get_model(inference_model, **kwargs.get('model_kwargs', {}))
    embedding_provider = get_embedder(embedding_model, **kwargs.get('embedder_kwargs', {}))
    cluster_analyzer = get_grouper(cluster_type, **kwargs.get('cluster_kwargs', {}))

    return DivergentGenerator(
        inference_model=inference_model,
        embedding_provider=embedding_provider,
        cluster_analyzer=cluster_analyzer
    )