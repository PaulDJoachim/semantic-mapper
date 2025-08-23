from pathlib import Path
from typing import Optional
from tree_utils import TreeNode, TreeOperations
from config.config import get_config


class TreeVisualizer:
    """Interactive HTML visualization for trees."""

    def __init__(self, template_path: Optional[str] = None):
        self.config = get_config()
        
        if template_path:
            self.template_path = Path(template_path)
        else:
            # Find template relative to project structure
            self.template_path = self._find_template()

        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {self.template_path}")

    def _find_template(self) -> Path:
        """Locate template file in project structure."""
        template_name = self.config.get("visualization", "template_path", "tree_template.html")
        
        # Start from this file's directory
        current = Path(__file__).parent
        
        # Look for templates directory
        for parent in [current] + list(current.parents):
            template_path = parent / "templates" / Path(template_name).name
            if template_path.exists():
                return template_path
            
            # Also check direct path from config
            template_path = parent / template_name
            if template_path.exists():
                return template_path
        
        raise FileNotFoundError(f"Template {template_name} not found")

    def export_to_html(self, root: TreeNode, output_path: str,
                      title: str = "Divergent Tree Visualization",
                      compress_linear: bool = None) -> None:
        """Export tree to HTML visualization."""
        compress_linear = compress_linear if compress_linear is not None else self.config.compress_linear
        tree_json = TreeOperations.to_json(root, compress_linear)

        template_content = self.template_path.read_text(encoding='utf-8')
        html_content = template_content.format(title=title, tree_data=tree_json)

        Path(output_path).write_text(html_content, encoding='utf-8')
        print(f"Visualization exported to {output_path}")

    def quick_export(self, root: TreeNode, prompt: str,
                    output_dir: str = None, **kwargs) -> str:
        """Generate filename and export in one step."""
        output_dir = output_dir or self.config.output_dir

        safe_prompt = "".join(c for c in prompt if c.isalnum() or c.isspace())
        safe_prompt = safe_prompt.replace(" ", "_")[:20]
        output_path = Path(output_dir) / f"tree_{safe_prompt}.html"
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        title = kwargs.pop("title", f"Tree: {prompt}")
        self.export_to_html(root, str(output_path), title, **kwargs)
        return str(output_path)


class TreePrinter:
    """Text-based tree display."""

    def __init__(self):
        self.config = get_config()

    def print_tree(self, root: TreeNode, max_depth: Optional[int] = None):
        """Print ASCII tree."""
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
        """Print sample paths."""
        num_paths = num_paths or self.config.getint("analysis", "sample_paths_count", 10)
        min_depth = min_depth or self.config.getint("analysis", "min_path_depth", 5)

        paths = TreeOperations.get_all_paths(root, min_depth)
        print(f"\nSample paths ({min(num_paths, len(paths))} of {len(paths)}):")

        for i, path in enumerate(paths[:num_paths]):
            print(f"\nPath {i + 1}: {prompt}{path}")