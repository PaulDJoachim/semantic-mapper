import torch
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional, NamedTuple
from config import get_config


class ClusteringResult(NamedTuple):
    """Encapsulates clustering analysis results."""
    labels: List[int]
    num_clusters: int
    has_branching: bool


class EmbeddingAnalyzer:
    """Analyzes semantic clusters using sentence embeddings."""

    def __init__(self, model_name: str = None):
        self.config = get_config()
        model_name = model_name or self.config.get("embeddings", "model_name", "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(model_name)

        # Cache last clustering result to avoid redundant computation
        self._last_stem_texts = None
        self._last_clustering_result = None

    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings for a batch of texts."""
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.config.getint("embeddings", "batch_size", 32),
            convert_to_tensor=True,
            show_progress_bar=False
        )
        return embeddings

    def cluster_stems(self, stem_texts: List[str]) -> ClusteringResult:
        """
        Cluster stems by semantic similarity using embeddings.
        Results are cached to avoid redundant computation.

        Returns:
            ClusteringResult with labels, cluster count, and branching status
        """
        from sklearn.cluster import DBSCAN

        if not stem_texts:
            return ClusteringResult([], 0, False)

        # Check cache first
        if (self._last_stem_texts is not None and
            self._last_stem_texts == stem_texts and
            self._last_clustering_result is not None):
            return self._last_clustering_result

        # Get embeddings for all stem texts
        embeddings = self.get_embeddings(stem_texts)
        embeddings_np = embeddings.cpu().numpy()

        # Cluster using DBSCAN
        eps = self.config.getfloat("clustering", "eps", 0.5)
        min_sample_ratio = self.config.getfloat("clustering", "min_sample_ratio", 0.1)
        min_samples = int(min_sample_ratio * len(stem_texts))
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings_np)

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        min_clusters = self.config.getint("clustering", "min_clusters", 2)
        has_branching = num_clusters >= min_clusters

        result = ClusteringResult(labels.tolist(), num_clusters, has_branching)

        print(f"Found {num_clusters} clusters")

        # Cache the result
        self._last_stem_texts = stem_texts.copy()
        self._last_clustering_result = result

        return result

    def get_cluster_representatives(self, stem_tokens: List[torch.Tensor],
                                   clustering_result: ClusteringResult) -> List[torch.Tensor]:
        """Get representative stems for each cluster using precomputed clustering."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_distances

        labels = clustering_result.labels
        unique_labels = set(label for label in labels if label != -1)

        if not unique_labels:
            # Fallback: return first stem if all are noise
            return [stem_tokens[0]] if stem_tokens else []

        representatives = []

        # We need to recompute embeddings here since we don't cache them
        # This is unavoidable for the distance calculation
        stem_texts = [self.embedding_model.tokenizer.decode(tokens.tolist()) for tokens in stem_tokens]
        embeddings = self.get_embeddings(stem_texts).cpu().numpy()

        for cluster_id in unique_labels:
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
            cluster_embeddings = embeddings[cluster_indices]

            # Find centroid and closest point
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = cosine_distances([centroid], cluster_embeddings)[0]
            closest_idx = cluster_indices[np.argmin(distances)]
            representatives.append(stem_tokens[closest_idx])

        return representatives