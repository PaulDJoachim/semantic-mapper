from semantic_embedding.embedding_provider import EmbeddingProvider
from typing import List
import numpy as np
from config.config import get_config


class SentenceEmbeddingProvider(EmbeddingProvider):
    """Sentence transformer embedding provider."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer, models

        config = get_config()
        device = config.get("model", "device")
        self.model_name = model_name
        self.normalize = config.getboolean("embeddings", "normalize")
        self.batch_size = config.getint("embeddings", "batch_size")
        self.custom_pooling = config.get("embeddings", "custom_pooling")

        # Build transformer + pooling
        embedding_model = models.Transformer(model_name)
        embedding_dimension = embedding_model.get_word_embedding_dimension()
        pooling_modes = {"max": "max", "mean": "mean", "lasttoken": "lasttoken"}

        if self.custom_pooling not in pooling_modes:
            raise ValueError(
                f"Unknown pooling: {self.custom_pooling}. Valid options: {list(pooling_modes.keys())}")

        pooling_model = models.Pooling(embedding_dimension, pooling_mode=pooling_modes[self.custom_pooling])

        self.model = SentenceTransformer(modules=[embedding_model, pooling_model], device=device)
        self.tokenizer = self.model.tokenizer

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using sentence transformers with configured pooling."""

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=self.normalize
        )

        return embeddings
