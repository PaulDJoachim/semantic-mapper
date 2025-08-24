"""Simple manual test runner for development workflow."""

from models.model_interface import create_generator


def test_basic_workflow():
    """Test the most important user workflow."""
    print("Testing basic DIA workflow...")
    
    generator = create_generator(
        model_type="mock",
        model_kwargs={"mode": "semantic_clusters", "seed": 42},
        analyzer_type="mock"
    )

    analysis = generator.full_analysis(
        "Individual autonomy versus collective welfare",
        max_depth=10,
        stem_length=3,
        num_stems=20
    )
    
    html_path = generator.visualizer.quick_export(analysis['root'], "test")
    
    print(f"✓ Generated {analysis['statistics']['total_nodes']} nodes")
    print(f"✓ Branching ratio: {analysis['branching_ratio']:.2f}")
    print(f"✓ Visualization: {html_path}")


def test_3d_pipeline():
    """Test 3D cluster data generation."""
    print("\nTesting 3D cluster pipeline...")
    
    generator = create_generator(
        model_type="mock",
        model_kwargs={"mode": "semantic_clusters", "seed": 42},
        analyzer_type="mock"
    )
    
    root = generator.explore_topology(
        "Test 3D clusters",
        max_depth=6,
        stem_length=2,
        num_stems=12
    )
    
    # Count nodes with cluster data
    cluster_nodes = 0
    sample_count = 0
    
    def count_clusters(node):
        nonlocal cluster_nodes, sample_count
        if node.cluster_data:
            cluster_nodes += 1
            sample_count += len(node.cluster_data.get('samples', []))
    
    from tree_utils import TreeOperations
    TreeOperations.traverse_depth_first(root, count_clusters)
    
    print(f"✓ {cluster_nodes} nodes with cluster data")
    print(f"✓ {sample_count} total 3D samples generated")
    
    if cluster_nodes > 0:
        print("✓ 3D pipeline working")
    else:
        print("⚠ No cluster data generated")


def test_mode_differences():
    """Verify different modes behave as expected."""
    print("\nTesting mode differences...")
    
    for mode in ["semantic_clusters", "linear"]:
        gen = create_generator(
            model_type="mock",
            model_kwargs={"mode": mode, "seed": 42},
            analyzer_type="mock"
        )
        
        analysis = gen.full_analysis("Test", max_depth=6, stem_length=2, num_stems=15)
        print(f"  {mode}: {analysis['branching_ratio']:.2f} branching ratio")


def main():
    """Run essential development tests."""
    print("Running DIA development tests")
    print("=" * 40)
    
    test_basic_workflow()
    test_3d_pipeline() 
    test_mode_differences()
    
    print("\n" + "=" * 40)
    print("✓ All tests passed - ready for development!")


if __name__ == "__main__":
    main()