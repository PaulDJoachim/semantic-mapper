import torch
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

    def explore_topology(self, prompt: str,
                         max_depth: int = None,
                         stem_length: int = None,
                         num_stems: int = None,
                         temperature: float = None,
                         top_k: int = None,
                         top_p: float = None,
                         print_stems: bool = False) -> TreeNode:
        """
        Explores semantic branching by periodically generating and clustering stems.
        """
        max_depth = max_depth or self.config.max_depth
        stem_length = stem_length or self.config.getint("generation", "stem_length", 10)
        num_stems = num_stems or self.config.getint("generation", "num_stems", 50)
        temperature = temperature if temperature is not None else self.config.getfloat("generation", "temperature", 1.0)
        top_k = top_k if top_k is not None else self.config.getint("generation", "top_k", 0)
        top_p = top_p if top_p is not None else self.config.getfloat("generation", "top_p", 1.0)

        print(f"Generation settings: temp={temperature}, top_k={top_k}, top_p={top_p}")

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

                # Generate stems from this branch point
                stems = self.model.generate_stems(branch_sequence, num_stems, stem_length,
                                                  temperature=temperature,
                                                  top_k=top_k,
                                                  top_p=top_p)

                # Extract tokens and texts for analysis
                stem_tokens = [stem[0] for stem in stems]
                stem_texts = [self.model.decode(tokens.tolist()) for tokens in stem_tokens]

                # Print stems if requested
                if print_stems:
                    self._print_stems(stem_texts, branch_node, prompt)

                # Compute clustering once and use results for both branching analysis and representatives
                clustering_result = self.analyzer.cluster_stems(stem_texts)

                if clustering_result.has_branching:
                    # Found genuine semantic divergence - create branches
                    representatives = self.analyzer.get_cluster_representatives(stem_tokens, clustering_result)

                    print(f"Found {clustering_result.num_clusters} semantic clusters at depth {branch_node.depth}")

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