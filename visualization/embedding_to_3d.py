import numpy as np
from typing import List, Dict, Any
from clustering.cluster_analyzer import ClusteringResult
from sklearn.decomposition import PCA
from config.config import get_config

class EmbeddingTo3D:
    """Convert high-dimensional embeddings to 3D positions for visualization."""

    CLUSTER_COLORS = [
        '#ff6b6b', '#96e338', '#45b7d1', '#ffc619',
        '#9c46ec', '#ce7e00', '#2c81db', '#74b9ff',
        '#a29bfe', '#6c5ce7'
    ]
    NOISE_COLOR = '#c1bbb9'

    @staticmethod
    def create_visualization_data(clustering_result: ClusteringResult,
                                  texts: List[str]) -> Dict[str, Any]:
        """Convert clustering result to 3D visualization data."""
        if not texts or len(clustering_result.embeddings) == 0:
            return {'samples': [], 'stats': {}}

        # Generate 3D positions
        positions_3d = EmbeddingTo3D._embeddings_to_3d(clustering_result.embeddings)

        # Create samples with 3D data
        samples = []
        for text, label, pos_3d in zip(texts, clustering_result.labels, positions_3d):
            color = EmbeddingTo3D._get_cluster_color(label)

            samples.append({
                'text': text,
                'direction': pos_3d.tolist(),
                'cluster': int(label),
                'color': color,
            })

        # Generate cluster statistics
        stats = EmbeddingTo3D._generate_cluster_stats(clustering_result.labels)

        return {
            'samples': samples,
            'stats': stats
        }

    @staticmethod
    def _embeddings_to_3d(embeddings: np.ndarray) -> np.ndarray:
        """Reduce embeddings to 3D using PCA."""
        if embeddings.shape[0] < 3:
            # Handle edge case: pad with zeros if too few samples
            positions = np.zeros((len(embeddings), 3))
            for i in range(len(embeddings)):
                positions[i, i % 3] = 1.0  # Place on different axes
            return positions

        pca = PCA(n_components=3)
        positions_3d = pca.fit_transform(embeddings)

        pca_normalization = get_config().get("visualization", "pca_normalization")
        if pca_normalization == 'unit_sphere':
            # Normalize to unit sphere for consistent visualization
            norms = np.linalg.norm(positions_3d, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            return positions_3d / norms
        else:
            return positions_3d

    @staticmethod
    def _get_cluster_color(cluster_id: int) -> str:
        """Get color for cluster ID."""
        if cluster_id == -1:
            return EmbeddingTo3D.NOISE_COLOR
        return EmbeddingTo3D.CLUSTER_COLORS[cluster_id % len(EmbeddingTo3D.CLUSTER_COLORS)]

    @staticmethod
    def _generate_cluster_stats(labels: List[int]) -> Dict[str, int]:
        """Generate cluster statistics for UI display."""
        stats = {}
        for label in labels:
            key = 'noise' if label == -1 else f'cluster_{label}'
            stats[key] = stats.get(key, 0) + 1
        return stats