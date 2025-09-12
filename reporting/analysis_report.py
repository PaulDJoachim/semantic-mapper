from dataclasses import dataclass, field
from typing import Dict, Any, List
import json
import numpy as np
from datetime import datetime
from tree_utils import TreeNode, TreeOperations


@dataclass
class AnalysisReport:
    """Complete analysis report with tree structure, embeddings, and clustering data."""
    
    # Core data
    root: TreeNode
    prompt: str
    generation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis results
    tree_statistics: Dict[str, Any] = field(default_factory=dict)
    sample_paths: List[str] = field(default_factory=list)
    branching_ratio: float = 0.0
    average_path_length: float = 0.0
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_info: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            'metadata': {
                'prompt': self.prompt,
                'timestamp': self.timestamp,
                'model_info': self.model_info,
                'generation_params': self.generation_params
            },
            'analysis': {
                'tree_statistics': self.tree_statistics,
                'sample_paths': self.sample_paths,
                'branching_ratio': self.branching_ratio,
                'average_path_length': self.average_path_length
            },
            'tree_data': TreeOperations.to_dict(self.root),
            'cluster_summary': self._extract_cluster_summary()
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export report as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=self._json_serializer)
    
    def save_json(self, output_dir: str = None) -> str:
        """Save report to JSON file with auto-generated filename."""
        import re
        from pathlib import Path

        output_dir = Path(output_dir or "./output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create safe filename from prompt and timestamp
        safe_prompt = re.sub(r'[^\w\s-]', '', self.prompt)[:30].strip()
        safe_prompt = re.sub(r'[-\s]+', '_', safe_prompt)
        timestamp = self.timestamp.split('T')[0].replace('-', '')

        filename = f"analysis_{timestamp}_{safe_prompt}.json"
        filepath = output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

        return str(filepath)
    
    def _extract_cluster_summary(self) -> Dict[str, Any]:
        """Extract aggregated cluster data from tree for 3D visualization."""
        all_samples = []
        cluster_nodes = []
        
        def collect_cluster_data(node):
            if node.cluster_data and 'samples' in node.cluster_data:
                cluster_nodes.append({
                    'node_text': node.token_text,
                    'token_depth': node.token_depth,
                    'num_samples': len(node.cluster_data['samples'])
                })
                all_samples.extend(node.cluster_data['samples'])
        
        TreeOperations.traverse_depth_first(self.root, collect_cluster_data)
        
        # Aggregate statistics
        total_samples = len(all_samples)
        cluster_counts = {}
        if all_samples:
            for sample in all_samples:
                cluster_id = sample.get('cluster', -1)
                key = f'cluster_{cluster_id}' if cluster_id >= 0 else 'noise'
                cluster_counts[key] = cluster_counts.get(key, 0) + 1
        
        return {
            'total_samples': total_samples,
            'cluster_stats': cluster_counts,
            'nodes_with_clusters': len(cluster_nodes),
            'cluster_nodes_info': cluster_nodes,
            'all_samples': all_samples  # For full 3D visualization
        }
    
    def _json_serializer(self, obj):
        """Handle numpy arrays and other non-serializable objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@classmethod
def from_json(cls, json_str: str) -> 'AnalysisReport':
    """Create AnalysisReport from JSON string."""
    # This would require TreeNode deserialization logic
    # Implement if needed for loading saved reports
    raise NotImplementedError("Loading from JSON not implemented yet")