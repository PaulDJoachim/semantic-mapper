from clustering.cluster_analyzer import ClusterAnalyzer, ClusteringResult
import numpy as np
from config.config import get_config
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, cosine


class HierarchicalAnalyzer(ClusterAnalyzer):
    """Hierarchical clustering analyzer with adaptive distance threshold."""

    def __init__(self):
        config = get_config()
        self.min_sample_ratio = config.getfloat("clustering", "min_sample_ratio")
        self.cluster_top_p = config.getfloat("hi-clustering", "cluster_top_p")
        self.linkage_criterion = config.get("hi-clustering", "linkage_criterion")
        self.cut_distance = config.getfloat("hi-clustering", "cut_distance")
        self.force_cluster = config.getboolean("hi-clustering", "force_cluster")

        # New adaptive distance parameters
        self.use_adaptive_distance = config.getboolean("hi-clustering", "use_adaptive_distance", fallback=False)
        self.variance_scaling_factor = config.getfloat("hi-clustering", "variance_scaling_factor", fallback=1.0)

    def analyze_clusters(self, embeddings: np.ndarray) -> ClusteringResult:
        """Cluster embeddings using hierarchical clustering."""
        if len(embeddings) < 2:
            return ClusteringResult([], 0, False, embeddings)

        print(f"\n--- Hierarchical Clustering Analysis (n={len(embeddings)}) ---")

        # Calculate adaptive distance threshold if enabled
        if self.use_adaptive_distance:
            cut_distance = self._calculate_adaptive_distance(embeddings)
            print(f"Adaptive cut distance: {cut_distance:.3f} (base: {self.cut_distance:.3f})")
        else:
            cut_distance = self.cut_distance
            print(f"Using fixed cut distance: {cut_distance:.3f}")

        # Build dendrogram
        distances = pdist(embeddings, metric='cosine')
        linkage_matrix = linkage(distances, method=self.linkage_criterion)

        # Cut dendrogram at selected distance
        labels = fcluster(linkage_matrix, cut_distance, criterion='distance')
        labels = labels - 1  # Convert to 0-based indexing

        initial_clusters = len(set(labels))
        print(f"Initial clusters after cut: {initial_clusters}")

        # Filter out small clusters (noise)
        filtered_labels = self._filter_clusters_top_p(labels)

        # Calculate final statistics
        valid_clusters = set(filtered_labels[filtered_labels >= 0])
        num_clusters = len(valid_clusters)
        noise_points = np.sum(filtered_labels == -1)
        has_branching = num_clusters >= 2

        print(f"Final result: {num_clusters} clusters, {noise_points} noise points")
        print(f"Cluster sizes: {[int(np.sum(filtered_labels == label)) for label in sorted(valid_clusters)]}")
        print(f"Has branching: {has_branching}")

        return ClusteringResult(
            labels=filtered_labels.tolist(),
            representatives={},
            num_clusters=num_clusters,
            has_branching=has_branching,
            embeddings=embeddings
        )

    def _calculate_adaptive_distance(self, embeddings: np.ndarray) -> float:
        """Calculate adaptive distance threshold based on dataset variance."""
        # Calculate centroid of all embeddings
        centroid = np.mean(embeddings, axis=0)

        # Calculate distance from each embedding to centroid
        distances_to_centroid = [cosine(emb, centroid) for emb in embeddings]

        # Use standard deviation as variance measure
        # TODO: Print this to the UI somewhere
        variance_measure = np.std(distances_to_centroid)

        # # Scale base distance by variance
        # variance_scaling = 1 + self.variance_scaling_factor * variance_measure

        # Testing non-linear scaling
        variance_scaling = 1 + self.variance_scaling_factor * variance_measure ** 2

        adaptive_distance = self.cut_distance * variance_scaling

        print(f"Dataset variance (std of centroid distances): {variance_measure:.3f}")
        print(f"Variance scaling: {variance_scaling:.3f}")

        return adaptive_distance

    def _filter_clusters_top_p(self, labels: np.ndarray) -> np.ndarray:
        """Keep largest clusters until target coverage is reached, subject to minimum size threshold."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        min_samples = max(2, int(self.min_sample_ratio * total_samples))

        # Sort clusters by size descending
        sorted_idx = np.argsort(counts)[::-1]
        sorted_counts = counts[sorted_idx]
        sorted_labels = unique_labels[sorted_idx]

        # Filter by minimum size threshold
        size_mask = sorted_counts >= min_samples
        if not size_mask.any() and self.force_cluster:
            # Force include largest cluster if none meet threshold
            size_mask[0] = True

        valid_counts = sorted_counts[size_mask]
        valid_labels = sorted_labels[size_mask]

        # Find clusters needed to reach target coverage
        cumulative_samples = np.cumsum(valid_counts)
        coverage_threshold = self.cluster_top_p * total_samples
        keep_mask = (cumulative_samples <= coverage_threshold) | (np.arange(len(cumulative_samples)) == 0)

        # Always keep at least one cluster if any are valid
        if keep_mask.any():
            # Include the first cluster that pushes us over threshold
            first_over_idx = np.searchsorted(cumulative_samples, coverage_threshold, side='right')
            if first_over_idx < len(keep_mask):
                keep_mask[first_over_idx] = True

        final_labels = valid_labels[keep_mask]
        final_coverage = cumulative_samples[keep_mask][-1] if keep_mask.any() else 0

        # label remapping
        filtered_labels = np.full(len(labels), -1, dtype=int)
        for new_id, old_label in enumerate(final_labels):
            filtered_labels[labels == old_label] = new_id

        coverage = final_coverage / total_samples
        print(f"Kept {len(final_labels)} clusters covering {coverage:.1%} of samples")

        return filtered_labels