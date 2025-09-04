from typing import List, Optional, Dict, Any, Tuple
from tree_utils import TreeNode, TreeOperations
from models.model_interface import ModelInterface
from semantic_embedding.embedding_provider import EmbeddingProvider
from clustering.cluster_analyzer import ClusterAnalyzer, ClusteringResult
from config.config import get_config
from reporting.analysis_report import AnalysisReport
from visualization.embedding_to_3d import EmbeddingTo3D
import numpy as np


class DivergentGenerator:
    """Generates text trees by exploring semantic branching points."""

    def __init__(self, inference_model: ModelInterface,
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

    def _create_child_branches(self, parent_node: TreeNode, branch_sequence: Any,
                               clustering_result: ClusteringResult) -> List[Tuple[TreeNode, Any]]:
        """Create child nodes and new branch sequences from cluster representatives."""
        representatives = clustering_result.representatives
        labels = clustering_result.labels

        new_branches = []
        num_samples = len(labels)
        for cluster_label, rep_tokens in representatives.items():
            child_text = self.model.decode(rep_tokens)
            cluster_proportion = labels.count(cluster_label) / num_samples

            child = TreeNode(
                token_id=rep_tokens[0] if hasattr(rep_tokens, '__getitem__') else 0,
                token_text=child_text,
                proportion=cluster_proportion,
                depth=parent_node.depth + len(rep_tokens))
            parent_node.add_child(child)

            new_sequence = branch_sequence + rep_tokens
            new_branches.append((child, new_sequence))

        return new_branches

    def explore_topology(self, prompt: str,
                         max_depth: int = None,
                         stem_length: int = None,
                         num_stems: int = None,
                         temperature: float = None,
                         top_k: int = None,
                         top_p: float = None,
                         max_stems_per_node: int = None,
                         print_stems: bool = False,
                         seed: int = None) -> TreeNode:
        """Explores semantic branching by generating and clustering stems."""
        if seed is not None:
            self.set_seed(seed)

        max_depth = max_depth or self.config.max_depth
        stem_length = stem_length or self.config.getint("generation", "stem_length")
        num_stems = num_stems or self.config.getint("generation", "num_stems")
        temperature = temperature if temperature is not None else self.config.getfloat("generation", "temperature")
        top_k = top_k if top_k is not None else self.config.getint("generation", "top_k")
        top_p = top_p if top_p is not None else self.config.getfloat("generation", "top_p")
        max_stems_per_node = max_stems_per_node or self.config.getint("generation", "max_stems_per_node")

        input_ids = self.model.encode(prompt)
        root = TreeNode(token_id=-1, token_text=prompt, proportion=1.0, depth=0)

        active_branches = [(root, input_ids)]
        total_nodes = 1

        while active_branches and any(branch[0].depth < max_depth for branch in active_branches):
            new_branches = []

            for branch_node, branch_sequence in active_branches:
                if branch_node.depth >= max_depth:
                    new_branches.append((branch_node, branch_sequence))
                    continue

                stem_tokens, stem_texts, clustering_result, total_generated = (
                    self._generate_stems_until_clustered(branch_sequence,
                                                         num_stems,
                                                         stem_length,
                                                         temperature,
                                                         top_k,
                                                         top_p,
                                                         max_stems_per_node))

                if print_stems:
                    self._print_stems(stem_texts, branch_node, prompt)

                print(f"Node at depth {branch_node.depth}: {total_generated} stems -> {clustering_result.num_clusters} clusters")
                clustering_result.update_cluster_representatives(stem_tokens)

                viz_data = EmbeddingTo3D.create_visualization_data(clustering_result, stem_texts)
                branch_node.cluster_data = viz_data

                child_branches = self._create_child_branches(branch_node, branch_sequence, clustering_result)

                new_branches.extend(child_branches)
                total_nodes += len(child_branches)

            active_branches = new_branches

            if total_nodes % 10 == 0:
                print(f"Explored {total_nodes} nodes...")

        return root

    def _generate_stems_until_clustered(self, branch_sequence: Any,
                                        initial_num_stems: int,
                                        stem_length: int,
                                        temperature: float,
                                        top_k: int,
                                        top_p: float,
                                        max_total_stems: int = 2000) -> Tuple[List[Any], List[str], ClusteringResult, int]:
        """
        Generate stems iteratively until clusters are found or max limit reached.

        Returns:
            tuple of (stem_tokens, stem_texts, clustering_result, total_stems_generated)
        """
        all_stem_tokens = []
        all_stem_texts = []
        all_stem_embeddings = None
        current_batch_size = initial_num_stems
        total_generated = 0

        while total_generated < max_total_stems:
            # Generate current batch with entropy data
            stems, entropies = self.model.generate_stems(
                input_ids=branch_sequence,
                num_stems=current_batch_size,
                stem_length=stem_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p)

            # Prune stems based on entropy peaks
            pruned_stems = self._prune_stems_by_entropy(stems, entropies, stem_length)

            # Extract texts and embeddings
            batch_texts = [self.model.decode(tokens) for tokens in pruned_stems]
            batch_embeddings = self.embedding_provider.get_embeddings(batch_texts)

            # Add to accumulated results
            all_stem_tokens.extend(pruned_stems)
            all_stem_texts.extend(batch_texts)

            if all_stem_embeddings is None:
                all_stem_embeddings = batch_embeddings
            else:
                all_stem_embeddings = np.concatenate(
                    [all_stem_embeddings, batch_embeddings], axis=0)

            total_generated += current_batch_size

            clustering_result = self.cluster_analyzer.analyze_clusters(all_stem_embeddings)

            if clustering_result.num_clusters > 0:
                # Found clusters, we're done
                break

            if total_generated >= max_total_stems:
                # Hit maximum, stop here
                break

            # No clusters found, double the batch size for next iteration
            current_batch_size = min(current_batch_size * 2, max_total_stems - total_generated)
            print(f"No clusters found with {total_generated} stems, generating {current_batch_size} more...")

        return all_stem_tokens, all_stem_texts, clustering_result, total_generated

    def _prune_stems_by_entropy(self, stems: List[List[int]], entropies: List[List[float]],
                               original_stem_length: int) -> List[List[int]]:
        """Prune stems at token entropy peaks."""
        prune_range = self.config.getfloat("generation", "prune_range")

        pruned_stems = []
        search_range = max(1, int(prune_range * original_stem_length))
        min_length = original_stem_length - search_range

        for stem, entropy_sequence in zip(stems, entropies):

            # Look for entropy peak in the final search_range tokens
            search_start = min_length
            search_entropies = entropy_sequence[search_start:]

            if not search_entropies or len(search_entropies) <= 1:
                # No search range, keep full stem
                pruned_stems.append(stem)
                continue

            # TODO test using a moving average of entropy to filter out noise
            # Find largest entropy jump in the search range
            entropy_diffs = np.diff(search_entropies)
            max_jump_idx_in_range = np.argmax(entropy_diffs)

            # Prune stem before largest entropy jump in range
            cut_idx = search_start + max_jump_idx_in_range + 1
            pruned_stem = stem[:cut_idx]
            pruned_stems.append(pruned_stem)

        return pruned_stems

    def _print_stems(self, stem_texts: List[str], branch_node: TreeNode, original_prompt: str):
        """Print stems for inspection."""
        for i, stem_text in enumerate(stem_texts):
            print(f"{i+1:3d}: {repr(stem_text)}")

    def full_analysis(self, prompt: str, **kwargs) -> AnalysisReport:
        root = self.explore_topology(prompt, **kwargs)
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