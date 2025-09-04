from clustering.cluster_analyzer import ClusterAnalyzer, ClusteringResult
import numpy as np
from config.config import get_config
from sklearn.cluster import OPTICS


class OPTICSClusterAnalyzer(ClusterAnalyzer):
    """OPTICS-based clustering analyzer."""

    def __init__(self):
        config = get_config()
        self.min_sample_ratio = config.getfloat("clustering", "min_sample_ratio")
        self.eps = config.getfloat("OPTICS-clustering", "eps")
        self.xi = config.getfloat("OPTICS-clustering", "xi")
        self.n_jobs = config.getint("OPTICS-clustering", "n_jobs")

    def analyze_clusters(self, embeddings: np.ndarray) -> ClusteringResult:
        """Cluster embeddings using OPTICS."""
        total_stems = len(embeddings)
        if total_stems == 0:
            return ClusteringResult([], 0, False, embeddings)

        print(f"\n--- OPTICS Clustering Analysis (n={total_stems}) ---")

        min_samples = max(2, int(self.min_sample_ratio * total_stems))
        print(f"Using min_samples: {min_samples}")

        # Configure OPTICS
        optics_kwargs = {
            'min_samples': min_samples,
            'metric': 'cosine',
            'algorithm': 'auto',
            'n_jobs': -1,
            'cluster_method': 'xi',
            'xi': self.xi
        }

        embeddings = self._handle_duplicates(embeddings)
        optics = OPTICS(**optics_kwargs)
        labels = optics.fit_predict(embeddings)

        # Calculate statistics
        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_points = np.sum(labels == -1) if -1 in labels else 0
        has_branching = num_clusters >= 2

        # Get cluster sizes
        if num_clusters > 0:
            valid_labels = [label for label in unique_labels if label != -1]
            cluster_sizes = [int(np.sum(labels == label)) for label in sorted(valid_labels)]
            print(f"Cluster sizes: {cluster_sizes}")
        else:
            print("No clusters found")

        print(f"Final result: {num_clusters} clusters, {noise_points} noise points")
        print(f"Has branching: {has_branching}")

        # Provide some OPTICS-specific diagnostics
        if hasattr(optics, 'reachability_'):
            finite_reach = optics.reachability_[np.isfinite(optics.reachability_)]
            if len(finite_reach) > 0:
                print(f"Reachability range: {finite_reach.min():.3f} - {finite_reach.max():.3f}")
                print(f"Mean reachability: {finite_reach.mean():.3f}")

        return ClusteringResult(
            labels=labels.tolist(),
            representatives={},
            num_clusters=num_clusters,
            has_branching=has_branching,
            embeddings=embeddings
        )

    def _handle_duplicates(self, embeddings: np.ndarray) -> np.ndarray:
        """Add small random noise to duplicate embeddings to avoid numerical issues."""
        # Find duplicate rows
        unique_embeddings, inverse_indices = np.unique(embeddings, axis=0, return_inverse=True)

        if len(unique_embeddings) < len(embeddings):
            duplicates_found = len(embeddings) - len(unique_embeddings)
            print(f"Found {duplicates_found} duplicate embeddings, adding small noise")

            # Add small random noise to break ties
            noise_scale = 1e-8
            noise = np.random.RandomState(42).normal(0, noise_scale, embeddings.shape)
            embeddings_deduped = embeddings + noise

            return embeddings_deduped

        return embeddings