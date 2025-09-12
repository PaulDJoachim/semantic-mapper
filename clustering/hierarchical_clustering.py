import numpy as np
from typing import List
from config.config import get_config
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, cosine
from clustering.cluster_analyzer import ClusterAnalyzer, Cluster
from utilities.stem import StemPack


# TODO: Refactor - too much responsibility
class HierarchicalAnalyzer(ClusterAnalyzer):
    """Hierarchical clustering analyzer with adaptive distance threshold."""

    def __init__(self):
        self.config = get_config()
        self.min_sample_ratio = self.config.getfloat("clustering", "min_sample_ratio")
        self.cluster_top_p = self.config.getfloat("clustering", "cluster_top_p")
        self.linkage_criterion = self.config.get("clustering", "linkage_criterion")
        self.cut_distance = self.config.getfloat("clustering", "cut_distance")
        self.force_cluster = self.config.getboolean("clustering", "force_cluster")

    def analyze_clusters(self, stem_pack: StemPack) -> List[Cluster]:

        """Cluster delta embeddings and return list of Cluster objects."""
        print(f"\n--- Hierarchical Clustering Analysis (n={len(stem_pack.texts)}) ---")

        # Build dendrogram
        metric = 'cosine' if self.config.normalize_post_delta else 'euclidean'
        distances = pdist(stem_pack.delta_embeddings, metric=metric)  # type: ignore
        linkage_matrix = linkage(distances, method=self.linkage_criterion)

        # Cut dendrogram at selected distance
        # TODO: play with 'inconsistent' criterion
        labels = fcluster(linkage_matrix, self.cut_distance, criterion='distance')
        labels = labels - 1  # Convert to 0-based indexing

        initial_clusters = len(set(labels))
        print(f"Initial clusters after cut: {initial_clusters}")

        # Filter out small clusters (noise)
        filtered_labels = self._filter_clusters_top_p(labels)

        # Build Cluster objects from filtered results
        clusters = self._build_cluster_objects(filtered_labels, stem_pack)

        valid_clusters = len(clusters)
        noise_points = np.sum(filtered_labels == -1)
        
        print(f"Final result: {valid_clusters} clusters, {noise_points} noise points")
        print(f"Cluster sizes: {[cluster.size for cluster in clusters]}")

        return clusters

    def _build_cluster_objects(self, labels: np.ndarray, stem_pack: StemPack) -> List[Cluster]:
        """Group data by cluster labels and build Cluster objects."""
        clusters = []

        unique_labels = np.unique(labels)

        for label in sorted(unique_labels):
            cluster_mask = np.equal(labels, label)
            cluster_indices = np.where(cluster_mask)[0]

            cluster = Cluster(
                label=label,
                cluster_mask=cluster_mask,
                stem_pack=stem_pack,
                size=len(cluster_indices)
            )
            clusters.append(cluster)

        return clusters

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
