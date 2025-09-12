from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np


class EmbeddingProvider(ABC):
    """Abstract interface for generating text embeddings."""
    tokenizer: Any

    @abstractmethod
    def get_embeddings(self, texts: List[str], normalize: bool) -> np.ndarray:
        """Generate embeddings for input texts."""
        pass


def get_delta_embeddings(embeddings: np.ndarray, parent_embedding: np.ndarray, normalize: bool) -> np.ndarray:
    """Get the difference between embeddings and parent embedding.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        parent_embedding: Array of shape (embedding_dim,) or (1, embedding_dim)
        normalize: Whether to normalize the delta embeddings

    Returns:
        Delta embeddings, optionally normalized
    """
    # Compute raw delta embeddings
    delta_embeddings = embeddings - parent_embedding

    if normalize:
        # Compute L2 norm for each delta embedding
        norms = np.linalg.norm(delta_embeddings, axis=1, keepdims=True)

        # Avoid division by zero for zero vectors
        # Set norm to 1 for zero vectors (they'll remain zero after division)
        norms = np.where(norms == 0, 1, norms)

        # Normalize each delta embedding
        delta_embeddings = delta_embeddings / norms

    # TODO: move the jitterizer to a utility module something. It's hidden and violates SoC
    # Add noise to prevent exact duplicates
    jitter = np.random.normal(0, 1e-8, delta_embeddings.shape)
    delta_embeddings += jitter

    return delta_embeddings


def get_weighted_embeddings(embeddings: np.ndarray, parent_embedding: np.ndarray) -> np.ndarray:
    """Combine stems and parent embedding using a weighted average."""
    parent_weight = 0.3
    stem_weight = 0.7
    weighted_embeddings = parent_embedding * parent_weight + embeddings * stem_weight

    return weighted_embeddings
