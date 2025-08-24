"""Test the hybrid visualization with 3D cluster data."""

from models.model_interface import create_generator
from pathlib import Path


def test_hybrid_export():
    """Test exporting tree with 3D cluster data to hybrid template."""
    print("Testing hybrid visualization export...")
    
    generator = create_generator(
        model_type="mock",
        model_kwargs={"mode": "semantic_clusters", "seed": 42},
        analyzer_type="mock"
    )
    
    analysis = generator.full_analysis(
        "Should AI prioritize individual privacy or collective security?",
        max_depth=8,
        stem_length=3,
        num_stems=20
    )
    
    # Export with hybrid template
    html_path = generator.visualizer.quick_export(
        analysis['root'], 
        "hybrid_test",
        title="Hybrid 3D Cluster Test"
    )
    
    # Verify the file contains cluster data
    content = Path(html_path).read_text()
    
    cluster_indicators = [
        "cluster_data",
        "Cluster3DViewer", 
        "showClusters",
        "direction",
        "samples"
    ]
    
    found_indicators = [ind for ind in cluster_indicators if ind in content]
    
    print(f"✓ Exported to: {html_path}")
    print(f"✓ Found {len(found_indicators)}/{len(cluster_indicators)} cluster features")
    print(f"✓ Nodes with clusters: {analysis['statistics']['nodes_with_clusters']}")
    
    if len(found_indicators) == len(cluster_indicators):
        print("✓ Hybrid template fully integrated!")
    else:
        print(f"⚠ Missing: {set(cluster_indicators) - set(found_indicators)}")
    
    return html_path


if __name__ == "__main__":
    test_hybrid_export()
    print("\nOpen the HTML file in a browser and click nodes with green borders!")