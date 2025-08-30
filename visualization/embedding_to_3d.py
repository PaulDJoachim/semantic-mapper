import numpy as np
from typing import List, Dict, Any
from clustering.cluster_analyzer import ClusteringResult
from sklearn.decomposition import PCA


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
        for i, (text, label, pos_3d) in enumerate(zip(texts, clustering_result.labels, positions_3d)):
            color = EmbeddingTo3D._get_cluster_color(label)
            confidence = EmbeddingTo3D._estimate_confidence(clustering_result.embeddings[i], positions_3d)

            samples.append({
                'text': text,
                'direction': pos_3d.tolist(),  # Normalized to unit sphere
                'cluster': int(label),  # Convert numpy int to Python int
                'color': color,
                'confidence': float(confidence)  # Convert numpy float to Python float
            })

        # Generate cluster statistics
        stats = EmbeddingTo3D._generate_cluster_stats(clustering_result.labels)

        return {
            'samples': samples,
            'stats': stats
        }

    @staticmethod
    def _embeddings_to_3d(embeddings: np.ndarray) -> np.ndarray:
        """Reduce embeddings to 3D using PCA, normalized to unit sphere."""
        if embeddings.shape[0] < 3:
            # Handle edge case: pad with zeros if too few samples
            positions = np.zeros((len(embeddings), 3))
            for i in range(len(embeddings)):
                positions[i, i % 3] = 1.0  # Place on different axes
            return positions

        pca = PCA(n_components=3)
        positions_3d = pca.fit_transform(embeddings)

        # Normalize to unit sphere for consistent visualization
        norms = np.linalg.norm(positions_3d, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return positions_3d / norms

    @staticmethod
    def _get_cluster_color(cluster_id: int) -> str:
        """Get color for cluster ID."""
        if cluster_id == -1:
            return EmbeddingTo3D.NOISE_COLOR
        return EmbeddingTo3D.CLUSTER_COLORS[cluster_id % len(EmbeddingTo3D.CLUSTER_COLORS)]

    @staticmethod
    def _estimate_confidence(embedding: np.ndarray, all_positions: np.ndarray) -> float:
        """Estimate confidence based on embedding magnitude."""
        # Simple heuristic: longer embeddings = higher confidence
        magnitude = np.linalg.norm(embedding)
        # Normalize to 0-1 range based on typical embedding magnitudes
        return min(1.0, magnitude / 10.0)

    @staticmethod
    def _generate_cluster_stats(labels: List[int]) -> Dict[str, int]:
        """Generate cluster statistics for UI display."""
        stats = {}
        for label in labels:
            key = 'noise' if label == -1 else f'cluster_{label}'
            stats[key] = stats.get(key, 0) + 1
        return stats