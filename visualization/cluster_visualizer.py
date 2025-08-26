from pathlib import Path
import json
import re
from typing import Dict, Any, List
from reporting.analysis_report import AnalysisReport


class Cluster3DVisualizer:
    """3D semantic cluster visualization for analysis reports."""

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        self.template_path = self._find_template()

    def _find_template(self) -> Path:
        """Find 3D template in project structure."""
        current = Path(__file__).parent
        for parent in [current] + list(current.parents):
            template_path = parent / "templates" / "3d_template.html"
            if template_path.exists():
                return template_path
        raise FileNotFoundError("3d_template.html not found")

    def export_to_html(self, report: AnalysisReport, output_path: str) -> None:
        """Export analysis report to 3D cluster HTML visualization."""
        cluster_data = self._prepare_cluster_data(report)

        template_content = self.template_path.read_text(encoding='utf-8')
        html_content = template_content.format(
            title=f"3D Clusters: {report.prompt[:50]}...",
            cluster_data=json.dumps(cluster_data, indent=2)
        )

        Path(output_path).write_text(html_content, encoding='utf-8')

    def quick_export(self, report: AnalysisReport, suffix: str = "") -> str:
        """Generate filename and export report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        safe_prompt = re.sub(r'[^\w\s-]', '', report.prompt)[:30].strip()
        safe_prompt = re.sub(r'[-\s]+', '_', safe_prompt)
        timestamp = report.timestamp.split('T')[0].replace('-', '')

        filename = f"clusters_{timestamp}_{safe_prompt}{suffix}.html"
        output_path = self.output_dir / filename

        self.export_to_html(report, str(output_path))
        return str(output_path)

    def _prepare_cluster_data(self, report: AnalysisReport) -> Dict[str, Any]:
        """Convert report cluster data to 3D visualization format."""
        cluster_summary = report._extract_cluster_summary()
        samples = cluster_summary.get('all_samples', [])

        if not samples:
            return {'samples': [], 'stats': {}}

        colors = self._generate_colors(cluster_summary.get('cluster_stats', {}))

        # Add colors to samples (3D positions already computed)
        viz_samples = []
        for sample in samples:
            cluster_id = sample.get('cluster', -1)
            color = colors.get(cluster_id, '#cccccc')

            viz_samples.append({
                'text': sample.get('text', ''),
                'direction': sample.get('position_3d', [0, 0, 1]),
                'cluster': cluster_id,
                'color': color,
                'confidence': sample.get('confidence', 1.0)
            })

        return {
            'samples': viz_samples,
            'stats': cluster_summary.get('cluster_stats', {}),
            'num_clusters': len(cluster_summary.get('cluster_stats', {}))
        }

    def _generate_colors(self, cluster_stats: Dict[str, int]) -> Dict[int, str]:
        """Generate colors for cluster IDs."""
        colors = [
            '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#fd79a8',
            '#fdcb6e', '#e17055', '#74b9ff', '#a29bfe', '#6c5ce7'
        ]

        color_map = {}
        cluster_ids = [int(k.split('_')[1]) for k in cluster_stats.keys() if k.startswith('cluster_')]

        for i, cluster_id in enumerate(sorted(cluster_ids)):
            color_map[cluster_id] = colors[i % len(colors)]

        color_map[-1] = '#ffeaa7'  # Noise color
        return color_map