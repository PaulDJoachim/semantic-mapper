from semantic_embedding.embedding_provider import EmbeddingProvider
from typing import List
import numpy as np


class SentenceEmbeddingProvider(EmbeddingProvider):
    """Sentence transformer embedding provider."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 128):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using sentence transformers."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        return np.array(embeddings)