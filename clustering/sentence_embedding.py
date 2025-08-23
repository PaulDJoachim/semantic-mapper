import torch
from sentence_transformers import SentenceTransformer
from typing import List, Any
from embedding_analyzer import EmbeddingAnalyzer, ClusteringResult
from config.config import get_config


class SentenceEmbeddingAnalyzer(EmbeddingAnalyzer):
    """Real embedding analyzer using sentence transformers."""

    def __init__(self, model_name: str = None):
        self.config = get_config()
        model_name = model_name or self.config.get("embeddings", "model_name", "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(model_name)
        self._last_stem_texts = None
        self._last_clustering_result = None

    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings for texts."""
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.config.getint("embeddings", "batch_size", 32),
            convert_to_tensor=True,
            show_progress_bar=False
        )
        return embeddings

    def cluster_stems(self, stem_texts: List[str]) -> ClusteringResult:
        """Cluster stems using DBSCAN on embeddings."""
        from sklearn.cluster import DBSCAN

        if not stem_texts:
            return ClusteringResult([], 0, False)

        if (self._last_stem_texts == stem_texts and 
            self._last_clustering_result is not None):
            return self._last_clustering_result

        embeddings = self.get_embeddings(stem_texts)
        embeddings_np = embeddings.cpu().numpy()

        eps = self.config.getfloat("clustering", "eps", 0.5)
        min_sample_ratio = self.config.getfloat("clustering", "min_sample_ratio", 0.1)
        min_samples = int(min_sample_ratio * self.config.getint("generation", "num_stems"))
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings_np)

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        min_clusters = self.config.getint("clustering", "min_clusters", 2)
        has_branching = num_clusters >= min_clusters

        result = ClusteringResult(labels.tolist(), num_clusters, has_branching)
        print(f"Found {num_clusters} clusters")

        self._last_stem_texts = stem_texts.copy()
        self._last_clustering_result = result
        return result

    def get_cluster_representatives(self, stem_tokens: List[Any],
                                  clustering_result: ClusteringResult) -> List[Any]:
        """Get representative stems using embedding distances."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_distances

        labels = clustering_result.labels
        unique_labels = set(label for label in labels if label != -1)

        if not unique_labels:
            return [stem_tokens[0]] if stem_tokens else []

        representatives = []
        stem_texts = [self.embedding_model.tokenizer.decode(tokens.tolist()) 
                     for tokens in stem_tokens]
        embeddings = self.get_embeddings(stem_texts).cpu().numpy()

        for cluster_id in unique_labels:
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
            cluster_embeddings = embeddings[cluster_indices]

            centroid = np.mean(cluster_embeddings, axis=0)
            distances = cosine_distances([centroid], cluster_embeddings)[0]
            closest_idx = cluster_indices[np.argmin(distances)]
            representatives.append(stem_tokens[closest_idx])

        return representatives