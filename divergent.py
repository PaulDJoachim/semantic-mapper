import random
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from tree_utils import TreeNode, TreeOperations
from models.model_interface import ModelInterface
from semantic_embedding.embedding_provider import EmbeddingProvider
from clustering.cluster_analyzer import ClusterAnalyzer, ClusteringResult
from config.config import get_config
from reporting.analysis_report import AnalysisReport
from sklearn.decomposition import PCA


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
        random.seed(seed)
        np.random.seed(seed)
        if hasattr(self.model, 'set_seed'):
            self.model.set_seed(seed)

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
        root = TreeNode(token_id=-1, token_text=prompt, probability=1.0, depth=0)

        active_branches = [(root, input_ids)]
        total_nodes = 1

        while active_branches and any(branch[0].depth < max_depth for branch in active_branches):
            new_branches = []

            for branch_node, branch_sequence in active_branches:
                if branch_node.depth >= max_depth:
                    new_branches.append((branch_node, branch_sequence))
                    continue

                # Generate stems with dynamic sampling
                stem_tokens, stem_texts, clustering_result, total_generated = (
                    self._generate_stems_until_clustered(branch_sequence,
                                                         num_stems,
                                                         stem_length,
                                                         temperature,
                                                         top_k,
                                                         top_p,
                                                         max_stems_per_node))

                # Print stems if requested
                if print_stems:
                    self._print_stems(stem_texts, branch_node, prompt)

                print(f"Node at depth {branch_node.depth}: {total_generated} stems -> {clustering_result.num_clusters} clusters")

                if clustering_result.has_branching:
                    representatives = self.cluster_analyzer.get_cluster_representatives(
                        stem_tokens, clustering_result
                    )

                    for rep_tokens in representatives:
                        # Create child node with representative stem
                        child_text = self.model.decode(rep_tokens)
                        child = TreeNode(
                            token_id=rep_tokens[0] if hasattr(rep_tokens, '__getitem__') else 0,
                            token_text=child_text,
                            probability=1.0 / len(representatives),
                            depth=branch_node.depth + stem_length
                        )
                        branch_node.add_child(child)

                        # Create new sequence for this branch
                        new_sequence = branch_sequence + rep_tokens
                        new_branches.append((child, new_sequence))
                        total_nodes += 1
                else:
                    # No semantic branching - continue with single representative
                    representatives = self.cluster_analyzer.get_cluster_representatives(
                        stem_tokens, clustering_result
                    )

                    if representatives:
                        rep_tokens = representatives[0]
                        child_text = self.model.decode(rep_tokens)
                        child = TreeNode(
                            token_id=rep_tokens[0] if hasattr(rep_tokens, '__getitem__') else 0,
                            token_text=child_text,
                            probability=1.0,
                            depth=branch_node.depth + stem_length
                        )
                        branch_node.add_child(child)

                        new_sequence = branch_sequence + rep_tokens
                        new_branches.append((child, new_sequence))
                        total_nodes += 1

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
            tuple of (stem_tokens, stem_texts, total_stems_generated)
        """
        all_stem_tokens = []
        all_stem_texts = []
        all_stem_embeddings = None
        current_batch_size = initial_num_stems
        total_generated = 0

        while total_generated < max_total_stems:
            # Generate current batch
            stems = self.model.generate_stems(branch_sequence, current_batch_size, stem_length,
                                            temperature=temperature, top_k=top_k, top_p=top_p)

            # Extract tokens, texts, and embeddings
            batch_tokens = [stem[0] for stem in stems]
            batch_texts = [self.model.decode(tokens) for tokens in batch_tokens]
            batch_embeddings = self.embedding_provider.get_embeddings(batch_texts)

            # Add to accumulated results
            all_stem_tokens.extend(batch_tokens)
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
