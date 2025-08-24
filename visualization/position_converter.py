import numpy as np
from typing import List, Dict, Any, Optional
from clustering.cluster_analyzer import ClusteringResult


class EmbeddingTo3D:
    """Converts clustering results to 3D visualization data."""
    
    @staticmethod
    def create_visualization_data(clustering_result: ClusteringResult, 
                                 stem_texts: List[str], 
                                 confidences: Optional[List[float]] = None) -> Dict[str, Any]:
        """Create 3D visualization data from clustering results."""
        if clustering_result.embeddings is None:
            return {'samples': [], 'stats': {}}
            
        positions_3d = EmbeddingTo3D._embeddings_to_3d(clustering_result.embeddings)
        colors = EmbeddingTo3D._generate_cluster_colors(clustering_result.num_clusters)
        
        if confidences is None:
            confidences = [1.0] * len(stem_texts)
            
        samples = []
        for i, (text, position, label, confidence) in enumerate(
            zip(stem_texts, positions_3d, clustering_result.labels, confidences)):
            color = colors[label] if 0 <= label < len(colors) else '#ffeaa7'  # noise color
            samples.append({
                'text': text,
                'direction': position.tolist(),
                'cluster': label,
                'color': color,
                'confidence': confidence
            })

        stats = EmbeddingTo3D._compute_cluster_stats(clustering_result.labels)
        
        return {
            'samples': samples,
            'stats': stats,
            'num_clusters': clustering_result.num_clusters
        }
    
    @staticmethod
    def _embeddings_to_3d(embeddings: np.ndarray) -> np.ndarray:
        """Convert high-dimensional embeddings to normalized 3D positions."""
        if len(embeddings) < 3:
            return EmbeddingTo3D._fallback_3d(embeddings)
            
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            positions_3d = pca.fit_transform(embeddings)
        except ImportError:
            return EmbeddingTo3D._fallback_3d(embeddings)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(positions_3d, axis=1)
        norms[norms == 0] = 1
        return positions_3d / norms[:, np.newaxis]
    
    @staticmethod
    def _fallback_3d(embeddings: np.ndarray) -> np.ndarray:
        """Fallback 3D conversion without PCA."""
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
    def _generate_cluster_colors(num_clusters: int) -> List[str]:
        """Generate distinct colors for clusters."""
        colors = [
            '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#fd79a8',
            '#fdcb6e', '#e17055', '#74b9ff', '#a29bfe', '#6c5ce7'
        ]
        return [colors[i % len(colors)] for i in range(num_clusters)]
    
    @staticmethod
    def _compute_cluster_stats(labels: List[int]) -> Dict[str, int]:
        """Compute cluster statistics for visualization."""
        stats = {}
        for label in set(labels):
            key = f'cluster_{label}' if label >= 0 else 'noise'
            stats[key] = labels.count(label)
        return stats