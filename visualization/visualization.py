from pathlib import Path
from typing import Optional
import re
from tree_utils import TreeOperations
from reporting.analysis_report import AnalysisReport


class TreeVisualizer:
    """Interactive HTML visualization for analysis reports."""

    def __init__(self):
        self.templates_dir = Path("./templates")
        self.required_files = {
            'base': 'tree_base.html',
            'styles': 'tree_styles.css',
            'script': 'tree_script.js'
        }

        # Verify all template files exist
        for name, filename in self.required_files.items():
            file_path = self.templates_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Template file not found: {file_path}")

    def _load_template_parts(self):
        """Load all template components."""
        parts = {}
        for name, filename in self.required_files.items():
            file_path = self.templates_dir / filename
            parts[name] = file_path.read_text(encoding='utf-8')
        return parts

    def export(self, report: AnalysisReport, output_dir: str) -> str:
        """Export report to HTML and return the output path."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename from prompt
        safe_prompt = re.sub(r'[^\w\s-]', '', report.prompt)[:30].strip()
        safe_prompt = re.sub(r'[-\s]+', '_', safe_prompt)
        timestamp = report.timestamp.split('T')[0].replace('-', '')

        filename = f"tree_{timestamp}_{safe_prompt}.html"
        output_path = output_dir / filename

        # Load template parts
        parts = self._load_template_parts()

        # Substitute tree data in the script first
        processed_script = parts['script'].format(
            tree_data=TreeOperations.to_json(report.root)
        )

        # Assemble final HTML
        html_content = parts['base'].format(
            title=f"Tree: {report.prompt[:50]}...",
            styles=parts['styles'],
            script=processed_script
        )

        output_path.write_text(html_content, encoding='utf-8')
        return str(output_path)


class TreePrinter:
    """Text-based tree display for analysis reports."""

    def print_tree(self, report: AnalysisReport, max_depth: Optional[int] = None):
        """Print ASCII tree from report."""
        TreeOperations.print_tree(report.root, max_depth=max_depth)

    def print_statistics(self, report: AnalysisReport):
        """Print tree statistics from report."""
        print("Analysis Statistics:")
        for key, value in report.tree_statistics.items():
            print(f"  {key}: {value}")
        print(f"  branching_ratio: {report.branching_ratio:.3f}")
        print(f"  average_path_length: {report.average_path_length:.1f}")

    def print_sample_paths(self, report: AnalysisReport, num_paths: Optional[int] = None):
        """Print sample paths from report."""
        paths = report.sample_paths
        display_count = num_paths or min(10, len(paths))

        print(f"\nSample paths ({display_count} of {len(paths)}):")
        for i, path in enumerate(paths[:display_count]):
            print(f"\nPath {i + 1}: {report.prompt}{path}")

    def print_cluster_summary(self, report: AnalysisReport):
        """Print cluster analysis summary from report."""
        cluster_summary = report._extract_cluster_summary()

        print(f"\nCluster Summary:")
        print(f"  Total samples: {cluster_summary['total_samples']}")
        print(f"  Nodes with clusters: {cluster_summary['nodes_with_clusters']}")

        if cluster_summary['cluster_stats']:
            print("  Cluster distribution:")
            for cluster, count in cluster_summary['cluster_stats'].items():
                print(f"    {cluster}: {count}")

    def full_report(self, report: AnalysisReport):
        """Print complete analysis report."""
        print(f"\n{'='*70}")
        print(f"ANALYSIS REPORT: {report.prompt}")
        print(f"Generated: {report.timestamp}")
        print(f"Model: {report.model_info}")
        print(f"{'='*70}")

        self.print_statistics(report)
        self.print_cluster_summary(report)
        self.print_sample_paths(report)