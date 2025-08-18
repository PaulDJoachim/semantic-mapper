from pathlib import Path
from typing import Optional
from tree_utils import TreeNode, TreeOperations
from config import get_config


class TreeVisualizer:
    """Handles visualization and export of generation trees."""

    def __init__(self, template_path: str = None):
        self.config = get_config()
        template_path = template_path or self.config.template_path
        self.template_path = Path(template_path)

        if not self.template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

    def export_to_html(self, root: TreeNode, output_path: str,
                      title: str = "Divergent Tree Visualization",
                      compress_linear: bool = None) -> None:
        """Export tree to interactive HTML visualization."""
        compress_linear = compress_linear if compress_linear is not None else self.config.compress_linear
        tree_json = TreeOperations.to_json(root, compress_linear)

        template_content = self.template_path.read_text(encoding='utf-8')
        html_content = template_content.format(title=title, tree_data=tree_json)

        Path(output_path).write_text(html_content, encoding='utf-8')
        print(f"Visualization exported to {output_path}")

    def quick_export(self, root: TreeNode, prompt: str,
                    output_dir: str = None, **kwargs) -> str:
        """Generate a filename and export visualization in one step."""
        output_dir = output_dir or self.config.output_dir

        # Create safe filename from prompt
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c.isspace())
        safe_prompt = safe_prompt.replace(" ", "_")[:20]
        output_path = Path(output_dir) / f"tree_{safe_prompt}.html"

        title = kwargs.pop("title", f"Tree: {prompt}")
        self.export_to_html(root, str(output_path), title, **kwargs)
        return str(output_path)


class TreePrinter:
    """Handles text-based tree printing and display."""

    def __init__(self):
        self.config = get_config()

    def print_tree(self, root: TreeNode, max_depth: Optional[int] = None):
        """Print tree in ASCII format."""
        max_depth = max_depth or self.config.getint("visualization", "max_display_depth", 10)
        TreeOperations.print_tree(root, max_depth=max_depth)

    def print_statistics(self, root: TreeNode):
        """Print tree statistics."""
        stats = TreeOperations.get_statistics(root)
        print("Tree Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    def print_sample_paths(self, root: TreeNode, num_paths: int = None,
                          min_depth: Optional[int] = None, prompt: str = ""):
        """Print sample complete paths from the tree."""
        num_paths = num_paths or self.config.getint("analysis", "sample_paths_count", 10)
        min_depth = min_depth or self.config.getint("analysis", "min_path_depth", 5)

        paths = TreeOperations.get_all_paths(root, min_depth)
        print(f"\nSample paths ({min(num_paths, len(paths))} of {len(paths)}):")

        for i, path in enumerate(paths[:num_paths]):
            print(f"\nPath {i + 1}: {prompt}{path}")