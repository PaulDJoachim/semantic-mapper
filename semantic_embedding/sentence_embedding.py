from semantic_embedding.embedding_provider import EmbeddingProvider
from typing import List
import numpy as np
from config.config import get_config
import torch


class SentenceEmbeddingProvider(EmbeddingProvider):
    """Sentence transformer embedding provider."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        config = get_config()
        device = config.get("model", "device")
        self.model_name = model_name
        self.normalize = config.getboolean("embeddings", "normalize")
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = config.getint("embeddings", "batch_size")
        self.custom_pooling = config.get("embeddings", "custom_pooling")

        self.tokenizer = self.model.tokenizer

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using sentence transformers with max pooling."""

        token_embeddings = self.model.encode(
            texts,
            output_value="token_embeddings",
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # max pooling
        pooled_embeddings = torch.stack([
            torch.max(emb, dim=0)[0] for emb in token_embeddings
        ])

        return pooled_embeddings.cpu().numpy()