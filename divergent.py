from typing import List, Any, Tuple
from tree_utils import TreeNode, TreeOperations
from semantic_embedding.embedding_provider import EmbeddingProvider, get_delta_embeddings
from utilities.cluster_reps import assign_cluster_representatives
from utilities.entropy_distance import get_entropy_distance_mask
from models.generic_transformer import GenericTransformer
from clustering.cluster_analyzer import ClusterAnalyzer, Cluster
from config.config import get_config
from reporting.analysis_report import AnalysisReport
from visualization.embedding_to_3d import EmbeddingTo3D
import numpy as np


class DivergentGenerator:
    """Generates text trees by exploring semantic branching points."""

    def __init__(self, inference_model: GenericTransformer,
                 embedding_provider: EmbeddingProvider,
                 cluster_analyzer: ClusterAnalyzer):

        self.config = get_config()
        self.model = inference_model
        self.embedding_provider = embedding_provider
        self.cluster_analyzer = cluster_analyzer

    def set_seed(self, seed: int) -> None:
        """Set random seed for deterministic generation."""
        import random

        # Set Python and NumPy seeds
        random.seed(seed)
        np.random.seed(seed)

        # Set PyTorch seeds
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                # Force deterministic CUDA operations
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass  # PyTorch not available

        # Set model-specific seed if supported
        if hasattr(self.model, 'set_seed'):
            self.model.set_seed(seed)

    def generate_tree(self, prompt: str) -> TreeNode:
        """Explore semantic branching by generating and clustering stems."""

        self.set_seed(self.config.getint("generation", "seed"))

        # max_depth = self.config.max_depth
        max_entropy_depth = self.config.max_entropy_depth
        entropy_budget = self.config.entropy_budget
        gap_threshold = self.config.max_proportion_gap
        stem_length = self.config.stem_length
        num_stems = self.config.num_stems
        temperature = self.config.temperature
        normalize_pre_delta = self.config.normalize_pre_delta
        normalize_post_delta = self.config.normalize_post_delta
        top_k = self.config.top_k
        top_p = self.config.top_p

        #  encode initial prompt into tokens
        input_ids = self.model.encode(prompt)
        prompt_embedding = self.embedding_provider.get_embeddings([prompt], normalize_pre_delta)
        # create root node
        root_node = TreeNode(token_id=-1,
                             tokens=input_ids,
                             token_text=prompt,
                             semantic_embedding=prompt_embedding[0],
                             proportion=1.0,
                             token_depth=0,
                             entropy_depth=0)

        active_nodes = [root_node]

        while active_nodes and any(node.entropy_depth < max_entropy_depth for node in active_nodes):
            new_nodes = []

            for this_node in active_nodes:
                # if node has reached max depth, add it to the tree without generating
                if this_node.entropy_depth >= max_entropy_depth:
                    new_nodes.append(this_node)
                    continue

                # Generate stems
                token_arr, entropy_arr = self.model.generate_stems(
                    input_ids=this_node.get_token_sequence(),
                    num_stems=num_stems,
                    stem_length=stem_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p)

                stem_pack = this_node.stem_pack
                stem_pack.tokens = token_arr
                stem_pack.entropies = entropy_arr

                # Create attention mask for decoder and decode
                stem_pack.entropy_mask, stem_pack.entropy_sums = get_entropy_distance_mask(stem_pack.entropies, entropy_budget)
                stem_pack.texts = self.model.masked_batch_decode(stem_pack)

                parent_text = this_node.get_text_sequence() + " "
                text_with_parent_arr = np.char.add(parent_text, stem_pack.texts)

                stem_pack.embeddings = self.embedding_provider.get_embeddings(text_with_parent_arr, normalize_pre_delta)
                stem_pack.delta_embeddings = get_delta_embeddings(stem_pack.embeddings, this_node.semantic_embedding, normalize_post_delta)

                this_node.child_clusters = self.cluster_analyzer.analyze_clusters(stem_pack)

                print(f"Node at entropy depth {this_node.entropy_depth:.2f}: {len(stem_pack.texts)} stems -> {len(this_node.child_clusters)} clusters")

                # TODO: this_node.assign_cluster_representatives()
                assign_cluster_representatives(this_node.child_clusters)

                viz_data = EmbeddingTo3D.create_visualization_data(this_node.child_clusters)
                this_node.cluster_data = viz_data
                filtered_clusters = self._filter_noise_clusters(this_node.child_clusters)
                filtered_clusters = self._filter_by_gap_threshold(filtered_clusters, gap_threshold)
                child_nodes = self._create_child_nodes(this_node, filtered_clusters)
                new_nodes.extend(child_nodes)

                this_node.clean_stem_pack()

            active_nodes = new_nodes

        return root_node

    def _create_child_nodes(self, parent_node: TreeNode, cluster_list: List[Cluster]) -> List[Tuple[TreeNode, Any]]:
        entropy_budget = self.config.getfloat("generation", "entropy_budget")

        new_nodes = []
        for cluster in cluster_list:
            proportion = cluster.size / self.config.getint("generation", "num_stems")
            child_text = self.model.decode(cluster.representative_sequence)

            child = TreeNode(
                token_id=cluster.representative_sequence[0],
                tokens=cluster.representative_sequence,
                token_text=child_text,
                semantic_embedding=cluster.representative_semantic_embedding,
                proportion=proportion,
                token_depth=parent_node.token_depth + len(cluster.representative_sequence),
                entropy_depth=parent_node.entropy_depth + cluster.representative_entropy
            )
            parent_node.add_child(child)

            new_nodes.append(child)

        return new_nodes

    def _filter_by_gap_threshold(self, cluster_list: List[Cluster], gap_threshold: float) -> List[Cluster]:
        """Filter clusters by gap threshold, keeping those within threshold of next largest."""
        if not cluster_list:
            return []

        total_samples = sum(cluster.size for cluster in cluster_list)

        # Calculate proportions and sort by size descending
        cluster_proportions = [(cluster.size / total_samples, cluster) for cluster in cluster_list]
        sorted_clusters = sorted(cluster_proportions, key=lambda x: x[0], reverse=True)

        proportions = np.array([item[0] for item in sorted_clusters])
        gaps = proportions[:-1] - proportions[1:]

        # Find cutoff point
        large_gaps = np.where(gaps > gap_threshold)[0]
        cutoff_idx = large_gaps[0] + 1 if len(large_gaps) > 0 else len(sorted_clusters)

        return [cluster for _, cluster in sorted_clusters[:cutoff_idx]]

    def _filter_noise_clusters(self, cluster_list: List[Cluster]) -> List[Cluster]:
        return [cluster for cluster in cluster_list if cluster.label != -1]

    def full_analysis(self, prompt: str, **kwargs) -> AnalysisReport:
        root = self.generate_tree(prompt, **kwargs)
        stats = TreeOperations.get_statistics(root)
        paths = TreeOperations.get_all_paths(root, min_depth=5)

        return AnalysisReport(
            root=root,
            prompt=prompt,
            generation_params=kwargs,
            tree_statistics=stats,
            sample_paths=paths[:10],
            branching_ratio=stats['branching_points'] / max(stats['total_nodes'], 1),
            average_path_length=sum(len(path) for path in paths) / max(len(paths), 1),
            model_info=self.model.get_mode_info() if hasattr(self.model, 'get_mode_info') else str(type(self.model))
        )