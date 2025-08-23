from embedding_provider import EmbeddingProvider
from typing import List
import numpy as np


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, embedding_dim: int = 384, seed: int = None):
        self.embedding_dim = embedding_dim
        if seed is not None:
            np.random.seed(seed)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings based on text content."""
        embeddings = []

        for text in texts:
            # Create deterministic embeddings based on text content
            text_hash = hash(text.lower()) % 1000000
            np.random.seed(text_hash)

            # Generate base embedding
            embedding = np.random.randn(self.embedding_dim)

            # Add semantic clustering hints based on keywords
            if any(word in text.lower() for word in ['freedom', 'individual', 'autonomy', 'choice']):
                embedding[:50] += 2.0  # Individualistic cluster
            elif any(word in text.lower() for word in ['society', 'collective', 'community', 'group']):
                embedding[50:100] += 2.0  # Collective cluster
            elif any(word in text.lower() for word in ['good', 'great', 'excellent', 'wonderful']):
                embedding[100:150] += 2.0  # Positive cluster
            elif any(word in text.lower() for word in ['bad', 'terrible', 'awful', 'horrible']):
                embedding[150:200] += 2.0  # Negative cluster

            embeddings.append(embedding)

        return np.array(embeddings)
