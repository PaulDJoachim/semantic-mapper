from clustering.cluster_analyzer import ClusterAnalyzer, ClusteringResult
import numpy as np
from config.config import get_config
from typing import List, Any
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_distances


class HierarchicalAnalyzer(ClusterAnalyzer):
    """Hierarchical clustering analyzer using complete linkage."""

    def __init__(self):
        config = get_config()

        self.linkage_criterion = (config.get("hi-clustering", "linkage_criterion"))
        self.distance_threshold = (config.getfloat("hi-clustering", "distance_threshold"))
        self.min_sample_ratio = (config.getfloat("clustering", "min_sample_ratio"))
        self.merge_threshold = (config.getfloat("hi-clustering", "merge_threshold"))
        self.min_clusters = (config.getint("clustering", "min_clusters"))

    def analyze_clusters(self, embeddings: np.ndarray) -> ClusteringResult:
        """Cluster embeddings using hierarchical clustering with gap detection."""
        if len(embeddings) < 2:
            return ClusteringResult([], 0, False, embeddings)

        print(f"\n--- Hierarchical Clustering Debug (n={len(embeddings)}) ---")

        # Build dendrogram
        distances = pdist(embeddings, metric='cosine')
        linkage_matrix = linkage(distances, method=self.linkage_criterion)
        merge_distances = linkage_matrix[:, 2]

        print(f"Merge distances: {merge_distances[-10:]}")  # Last 10 merges
        print(f"Distance range: {merge_distances.min():.4f} - {merge_distances.max():.4f}")

        # Find natural cut point using gap detection
        cut_distance = self._find_cut_distance(linkage_matrix)
        print(f"Selected cut distance: {cut_distance}")

        if cut_distance is None:
            print("No natural structure found")
            return ClusteringResult([0] * len(embeddings), 1, False, embeddings)

        # Cut dendrogram at selected distance
        labels = fcluster(linkage_matrix, cut_distance, criterion='distance')
        labels = labels - 1  # Convert to 0-based indexing
        initial_clusters = len(set(labels))
        print(f"Initial clusters after cut: {initial_clusters}")

        # Validate clusters
        valid_clusters = self._validate_clusters(embeddings, labels)
        print(f"Validation passed: {valid_clusters}")

        if not valid_clusters:
            print("Clusters failed validation - returning single cluster")
            return ClusteringResult([0] * len(embeddings), 1, False, embeddings)

        num_clusters = len(set(labels))
        has_branching = num_clusters >= self.min_clusters
        print(f"Final result: {num_clusters} clusters, branching={has_branching}")

        return ClusteringResult(
            labels=labels.tolist(),
            num_clusters=num_clusters,
            has_branching=has_branching,
            embeddings=embeddings
        )

    def _find_cut_distance(self, linkage_matrix: np.ndarray) -> float:
        """Find cut distance using simple threshold approach."""
        merge_distances = linkage_matrix[:, 2]

        print(f"Looking for merges below threshold {self.merge_threshold}")

        # Find largest merge distance below threshold
        valid_distances = merge_distances[merge_distances < self.merge_threshold]

        if len(valid_distances) == 0:
            print("No merges below threshold - all points remain separate")
            return merge_distances.min() - 1e-10  # Cut below smallest merge to keep all separate

        cut_distance = valid_distances.max()
        print(f"Cutting at distance {cut_distance:.6f}")
        return cut_distance

    def _validate_clusters(self, embeddings: np.ndarray, labels: np.ndarray) -> bool:
        """Validate that clusters meet quality criteria."""
        unique_labels = set(labels)
        min_samples = max(2, int(self.min_sample_ratio * len(embeddings)))

        print(f"  Validation: need {min_samples} samples per cluster (ratio={self.min_sample_ratio})")

        # Check cluster sizes
        valid_cluster_count = 0
        for label in unique_labels:
            cluster_size = np.sum(labels == label)
            print(f"  Cluster {label}: {cluster_size} samples")
            if cluster_size >= min_samples:
                valid_cluster_count += 1

        print(f"  Valid clusters by size: {valid_cluster_count}/{len(unique_labels)}")

        # Check inter-cluster distances
        if len(unique_labels) > 1:
            centroids = []
            for label in unique_labels:
                cluster_mask = labels == label
                if np.sum(cluster_mask) >= min_samples:
                    centroid = np.mean(embeddings[cluster_mask], axis=0)
                    centroids.append(centroid)

            if len(centroids) >= 2:
                centroid_distances = cosine_distances(centroids)
                min_centroid_distance = np.min(centroid_distances[np.triu_indices_from(centroid_distances, k=1)])
                print(f"  Min centroid distance: {min_centroid_distance:.6f} (need >{self.distance_threshold})")

                if min_centroid_distance < self.distance_threshold:
                    print(f"  Failed: centroids too close")
                    return False

        return True

    def get_cluster_representatives(self, items: List[Any],
                                 clustering_result: ClusteringResult) -> List[Any]:
        """Get representative items using centroid proximity."""
        embeddings = clustering_result.embeddings
        labels = clustering_result.labels

        if not items or embeddings is None:
            return []

        unique_labels = list(set(labels))
        representatives = []

        for label in unique_labels:
            cluster_indices = [i for i, l in enumerate(labels) if l == label]
            cluster_embeddings = embeddings[cluster_indices]

            # Find item closest to cluster centroid
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = cosine_distances([centroid], cluster_embeddings)[0]
            closest_idx = cluster_indices[np.argmin(distances)]
            representatives.append(items[closest_idx])

        return representatives