import numpy as np
from typing import List


class EmbeddingTo3D:
    """Converts high-dimensional embeddings to 3D positions using PCA."""
    
    def __init__(self):
        self.pca = None
        self._is_fitted = False
    
    def convert(self, embeddings: np.ndarray) -> np.ndarray:
        """Convert embeddings to 3D positions."""
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            return self._fallback_conversion(embeddings)
        
        if len(embeddings) < 3:
            return self._fallback_conversion(embeddings)
        
        # Fit PCA to reduce to 3D
        pca = PCA(n_components=3)
        positions_3d = pca.fit_transform(embeddings)
        
        # Normalize to unit sphere for consistent ray lengths
        norms = np.linalg.norm(positions_3d, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero
        positions_3d = positions_3d / norms[:, np.newaxis]
        
        return positions_3d
    
    def _fallback_conversion(self, embeddings: np.ndarray) -> np.ndarray:
        """Fallback conversion when PCA unavailable or insufficient data."""
        if len(embeddings) == 0:
            return np.array([]).reshape(0, 3)
        
        # Use first 3 dimensions or pad with zeros
        if embeddings.shape[1] >= 3:
            positions = embeddings[:, :3]
        else:
            positions = np.zeros((len(embeddings), 3))
            positions[:, :embeddings.shape[1]] = embeddings
        
        # Normalize
        norms = np.linalg.norm(positions, axis=1)
        norms[norms == 0] = 1
        return positions / norms[:, np.newaxis]
    
    @staticmethod
    def generate_cluster_colors(num_clusters: int, include_noise: bool = True) -> List[str]:
        """Generate distinct colors for clusters."""
        colors = [
            '#ff6b6b',  # Red
            '#4ecdc4',  # Teal
            '#45b7d1',  # Blue
            '#96ceb4',  # Green
            '#fd79a8',  # Pink
            '#fdcb6e',  # Yellow
            '#e17055',  # Orange
            '#74b9ff',  # Light blue
            '#a29bfe',  # Purple
            '#6c5ce7'   # Dark purple
        ]
        
        cluster_colors = []
        for i in range(num_clusters):
            cluster_colors.append(colors[i % len(colors)])
        
        if include_noise:
            cluster_colors.append('#ffeaa7')  # Light yellow for noise
        
        return cluster_colors