from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EmbeddingProvider(ABC):
    """Abstract interface for generating text embeddings."""

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for input texts."""
        pass

