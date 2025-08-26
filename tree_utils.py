from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json


@dataclass
class TreeNode:
    """Represents a single point in the generation tree with cluster visualization data."""
    token_id: int
    token_text: str
    probability: float
    depth: int
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    cluster_data: Optional[Dict[str, Any]] = None

    def get_sequence(self) -> List[int]:
        """Returns the token sequence from root to this node."""
        sequence = []
        node = self
        while node and node.parent:
            sequence.append(node.token_id)
            node = node.parent
        return list(reversed(sequence))

    def get_text_sequence(self) -> str:
        """Returns the text sequence from root to this node."""
        text_parts = []
        node = self
        while node and node.parent:
            text_parts.append(node.token_text)
            node = node.parent
        return ''.join(reversed(text_parts))

    def add_child(self, child: 'TreeNode') -> None:
        """Add a child node and set its parent reference."""
        child.parent = self
        self.children.append(child)

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children)."""
        return len(self.children) == 0

    def is_branching_point(self) -> bool:
        """Check if this node has multiple children (branching point)."""
        return len(self.children) > 1


class TreeOperations:
    """Utility operations for working with generation trees."""

    @staticmethod
    def traverse_depth_first(root: TreeNode, visit_func=None):
        """Traverse tree depth-first, optionally applying visit_func to each node."""
        if visit_func:
            visit_func(root)

        for child in root.children:
            TreeOperations.traverse_depth_first(child, visit_func)

    @staticmethod
    def get_all_paths(root: TreeNode, min_depth: Optional[int] = None) -> List[str]:
        """Returns all unique paths in the tree."""
        paths = []

        def traverse(node: TreeNode, current_path: str):
            if node.parent:
                current_path += node.token_text

            if node.is_leaf():
                if not min_depth or node.depth >= min_depth:
                    paths.append(current_path)
            else:
                for child in node.children:
                    traverse(child, current_path)

        traverse(root, "")
        return paths

    @staticmethod
    def get_statistics(root: TreeNode) -> Dict[str, Any]:
        """Gather statistics about the generated tree."""
        stats = {
            "total_nodes": 0, 
            "max_depth": 0, 
            "leaf_nodes": 0, 
            "branching_points": 0,
            "unique_paths": 0,
            "nodes_with_clusters": 0
        }

        def count_node(node: TreeNode):
            stats["total_nodes"] += 1
            stats["max_depth"] = max(stats["max_depth"], node.depth)
            
            if node.cluster_data:
                stats["nodes_with_clusters"] += 1

            if node.is_leaf():
                stats["leaf_nodes"] += 1
            elif node.is_branching_point():
                stats["branching_points"] += 1

        TreeOperations.traverse_depth_first(root, count_node)
        stats["unique_paths"] = len(TreeOperations.get_all_paths(root))
        return stats

    @staticmethod
    def print_tree(node: TreeNode, prefix: str = "", is_last: bool = True,
                   max_depth: Optional[int] = None):
        """Pretty prints the generation tree."""
        if max_depth and node.depth > max_depth:
            return

        if node.parent:
            connector = "└── " if is_last else "├── "
            cluster_info = ""
            if node.cluster_data:
                cluster_info = f" [{node.cluster_data['num_clusters']} clusters]"
            print(f"{prefix}{connector}{repr(node.token_text)} (p={node.probability:.3f}, d={node.depth}){cluster_info}")
            prefix += "    " if is_last else "│   "

        for i, child in enumerate(node.children):
            TreeOperations.print_tree(child, prefix, i == len(node.children) - 1, max_depth)

    @staticmethod
    def to_dict(node: TreeNode, compress_linear: bool = True) -> Dict[str, Any]:
        """Convert TreeNode to dictionary, including cluster data."""
        node_dict = {
            "token_id": node.token_id,
            "token_text": node.token_text,
            "probability": node.probability,
            "depth": node.depth,
            "children": [TreeOperations.to_dict(child, compress_linear) for child in node.children]
        }
        
        if node.cluster_data:
            node_dict["cluster_data"] = node.cluster_data
            
        return node_dict

    @staticmethod
    def to_json(root: TreeNode, compress_linear: bool = True, indent: int = 2) -> str:
        """Convert tree to JSON string."""
        return json.dumps(TreeOperations.to_dict(root, compress_linear), indent=indent)