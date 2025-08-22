import torch
import random
import numpy as np
from typing import List, Optional, Dict, Any
from tree_utils import TreeNode, TreeOperations
from model_interface import ModelInterface, GPT2Interface
from embedding_analyzer import EmbeddingAnalyzer
from visualization import TreeVisualizer, TreePrinter
from config import get_config


class DivergentGenerator:
    """Generates text trees by exploring semantic branching points."""

    def __init__(self, model_interface: Optional[ModelInterface] = None,
                 visualizer: Optional[TreeVisualizer] = None):
        self.config = get_config()
        self.model = model_interface or GPT2Interface()
        self.visualizer = visualizer or TreeVisualizer()
        self.analyzer = EmbeddingAnalyzer()
        self.printer = TreePrinter()

    def set_seed(self, seed: int) -> None:
        """Set random seed for deterministic generation."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _generate_stems_until_clustered(self, branch_sequence: torch.Tensor,
                                       initial_num_stems: int,
                                       stem_length: int,
                                       temperature: float,
                                       top_k: int,
                                       top_p: float,
                                       max_total_stems: int = 2000) -> tuple[List[torch.Tensor], List[str], int]:
        """
        Generate stems iteratively until clusters are found or max limit reached.

        Returns:
            tuple of (stem_tokens, stem_texts, total_stems_generated)
        """
        all_stem_tokens = []
        all_stem_texts = []
        current_batch_size = initial_num_stems
        total_generated = 0

        while total_generated < max_total_stems:
            # Generate current batch
            stems = self.model.generate_stems(branch_sequence, current_batch_size, stem_length,
                                            temperature=temperature, top_k=top_k, top_p=top_p)

            # Extract tokens and texts
            batch_tokens = [stem[0] for stem in stems]
            batch_texts = [self.model.decode(tokens.tolist()) for tokens in batch_tokens]

            # Add to accumulated results
            all_stem_tokens.extend(batch_tokens)
            all_stem_texts.extend(batch_texts)
            total_generated += current_batch_size

            # Check for clustering
            clustering_result = self.analyzer.cluster_stems(all_stem_texts)

            if clustering_result.num_clusters > 0:
                # Found clusters, we're done
                break

            if total_generated >= max_total_stems:
                # Hit maximum, stop here
                print(f"Warning: Hit maximum stems ({max_total_stems}) without finding clusters")
                break

            # No clusters found, double the batch size for next iteration
            current_batch_size = min(current_batch_size * 2, max_total_stems - total_generated)
            print(f"No clusters found with {total_generated} stems, generating {current_batch_size} more...")

        return all_stem_tokens, all_stem_texts, total_generated

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
        """
        Explores semantic branching by periodically generating and clustering stems.
        Uses dynamic sampling to ensure reliable cluster detection.
        """
        # Set seed for deterministic generation if provided
        if seed is not None:
            self.set_seed(seed)
            print(f"Set random seed to {seed} for deterministic generation")
        elif hasattr(self.config, 'getint') and self.config.getint("generation", "seed", None) is not None:
            config_seed = self.config.getint("generation", "seed")
            self.set_seed(config_seed)
            print(f"Set random seed to {config_seed} from config for deterministic generation")

        max_depth = max_depth or self.config.max_depth
        stem_length = stem_length or self.config.getint("generation", "stem_length", 10)
        num_stems = num_stems or self.config.getint("generation", "num_stems", 50)
        temperature = temperature if temperature is not None else self.config.getfloat("generation", "temperature", 1.0)
        top_k = top_k if top_k is not None else self.config.getint("generation", "top_k", 0)
        top_p = top_p if top_p is not None else self.config.getfloat("generation", "top_p", 1.0)
        max_stems_per_node = max_stems_per_node or self.config.getint("generation", "max_stems_per_node", 2000)

        print(f"Generation settings: temp={temperature}, top_k={top_k}, top_p={top_p}")
        print(f"Dynamic sampling: initial={num_stems}, max_per_node={max_stems_per_node}")

        input_ids = self.model.encode(prompt)
        root = TreeNode(token_id=-1, token_text=prompt, probability=1.0, depth=0)

        # Initialize active branches with the root
        active_branches = [(root, input_ids)]
        total_nodes = 1

        while active_branches and any(branch[0].depth < max_depth for branch in active_branches):
            new_branches = []

            for branch_node, branch_sequence in active_branches:
                if branch_node.depth >= max_depth:
                    new_branches.append((branch_node, branch_sequence))
                    continue

                # Generate stems with dynamic sampling
                stem_tokens, stem_texts, total_generated = self._generate_stems_until_clustered(
                    branch_sequence, num_stems, stem_length, temperature, top_k, top_p, max_stems_per_node
                )

                # Print stems if requested
                if print_stems:
                    self._print_stems(stem_texts, branch_node, prompt)

                # Final clustering analysis
                clustering_result = self.analyzer.cluster_stems(stem_texts)

                print(f"Node at depth {branch_node.depth}: {total_generated} stems -> {clustering_result.num_clusters} clusters")

                if clustering_result.has_branching:
                    # Found genuine semantic divergence - create branches
                    representatives = self.analyzer.get_cluster_representatives(stem_tokens, clustering_result)

                    for rep_tokens in representatives:
                        # Create child node with representative stem
                        child_text = self.model.decode(rep_tokens.tolist())
                        child = TreeNode(
                            token_id=rep_tokens[0].item(),
                            token_text=child_text,
                            probability=1.0 / len(representatives),
                            depth=branch_node.depth + stem_length
                        )
                        branch_node.add_child(child)

                        # Create new sequence for this branch
                        new_sequence = torch.cat([branch_sequence, rep_tokens.unsqueeze(0)], dim=1)
                        new_branches.append((child, new_sequence))
                        total_nodes += 1
                else:
                    # No semantic branching - continue with single representative
                    representatives = self.analyzer.get_cluster_representatives(stem_tokens, clustering_result)

                    if representatives:
                        rep_tokens = representatives[0]
                        child_text = self.model.decode(rep_tokens.tolist())
                        child = TreeNode(
                            token_id=rep_tokens[0].item(),
                            token_text=child_text,
                            probability=1.0,
                            depth=branch_node.depth + stem_length
                        )
                        branch_node.add_child(child)

                        new_sequence = torch.cat([branch_sequence, rep_tokens.unsqueeze(0)], dim=1)
                        new_branches.append((child, new_sequence))
                        total_nodes += 1

            active_branches = new_branches

            if total_nodes % 10 == 0:
                print(f"Explored {total_nodes} nodes...")

        return root

    def _print_stems(self, stem_texts: List[str], branch_node: TreeNode, original_prompt: str):
        """Print all stems for inspection."""
        print(f"\n{'='*80}")
        print(f"STEMS at depth {branch_node.depth}")
        print(f"Context: {original_prompt}{branch_node.get_text_sequence()}")
        print(f"{'='*80}")

        for i, stem_text in enumerate(stem_texts):
            print(f"{i+1:3d}: {repr(stem_text)}")

        print(f"{'='*80}\n")

    def analyze_tree(self, root: TreeNode) -> Dict[str, Any]:
        """Analyze an existing tree and return detailed metrics."""
        stats = TreeOperations.get_statistics(root)
        min_path_depth = self.config.getint("analysis", "min_path_depth", 5)
        paths = TreeOperations.get_all_paths(root, min_depth=min_path_depth)

        return {
            'statistics': stats,
            'sample_paths': paths[:self.config.getint("analysis", "sample_paths_count", 10)],
            'branching_ratio': stats['branching_points'] / max(stats['total_nodes'], 1),
            'average_path_length': sum(len(path) for path in paths) / max(len(paths), 1)
        }

    def generate_and_export(self, prompt: str, output_dir: str = None,
                           **explore_kwargs) -> str:
        """Generate tree and export visualization."""
        output_dir = output_dir or self.config.output_dir

        print(f"Exploring semantic topology for: '{prompt}'")
        root = self.explore_topology(prompt, **explore_kwargs)

        self.printer.print_statistics(root)
        output_path = self.visualizer.quick_export(root, prompt, output_dir)
        return output_path

    def full_analysis(self, prompt: str, **explore_kwargs) -> Dict[str, Any]:
        """Generate tree and return complete analysis including the tree itself."""
        root = self.explore_topology(prompt, **explore_kwargs)
        analysis = self.analyze_tree(root)
        analysis['root'] = root
        return analysis