import numpy as np
from typing import List, Dict, Any
from clustering.cluster_analyzer import Cluster
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
    def create_visualization_data(cluster_list: List[Cluster]) -> Dict[str, Any]:
        """Convert cluster list to 3D visualization data."""

        # Get shared data from first cluster
        stem_pack = cluster_list[0].stem_pack
        mean_entropy = stem_pack.mean_entropy
        all_embeddings = stem_pack.delta_embeddings
        texts = stem_pack.texts
        n_samples = len(all_embeddings)

        # Convert to 3D once
        positions_3d = EmbeddingTo3D._embeddings_to_3d(all_embeddings)

        # TODO: move color stuff to frontend
        # Pre-compute color mapping for unique cluster IDs
        unique_labels = set()
        for cluster in cluster_list:
            unique_labels.add(cluster.label)
        unique_labels.add(-1)  # Add noise label

        color_map = {label: EmbeddingTo3D._get_cluster_color(label) for label in unique_labels}

        # Build cluster labels array
        cluster_labels = np.full(n_samples, -1, dtype=int)  # Initialize with noise
        for cluster in cluster_list:
            cluster_labels[cluster.cluster_mask] = cluster.label

        # color assignment
        colors = np.array([color_map[label] for label in cluster_labels])

        # Convert all positions to lists at once (more efficient than per-row conversion)
        directions_list = positions_3d.tolist()

        # Build samples list
        samples = []
        for i in range(n_samples):
            samples.append({
                'text': texts[i] + f'[{mean_entropy[i]:.2f}]',
                'direction': directions_list[i],
                'cluster': int(cluster_labels[i]),
                'color': colors[i],
            })

        stats = EmbeddingTo3D._generate_cluster_stats(cluster_labels)

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
    def _generate_cluster_stats(labels: np.ndarray) -> Dict[str, int]:
        """Generate cluster statistics for UI display."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        stats = {}
        for label, count in zip(unique_labels, counts):
            key = 'noise' if label == -1 else f'cluster_{label}'
            stats[key] = int(count)
        return stats
