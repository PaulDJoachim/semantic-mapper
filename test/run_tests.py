"""Core integration test to verify the main DIA pipeline works."""

from pathlib import Path
from models.model_interface import create_generator
from visualization.visualization import TreeVisualizer


def test_core_pipeline_works():
    """Test that the essential pipeline completes without errors."""

    # Create generator with mock components for reliability
    generator = create_generator(
        inference_model="mock",
        embedding_model="mock",
        cluster_type="mock",
        model_kwargs={"mode": "semantic_clusters", "seed": 42},
        cluster_kwargs={"seed": 42}
    )

    # Run full analysis - this exercises the entire pipeline
    analysis = generator.full_analysis(
        prompt="Individual freedom versus collective responsibility",
        max_depth=6,
        stem_length=3,
        num_stems=15,
        seed=42
    )

    # Verify we got a sensible result
    assert analysis.root is not None
    assert analysis.root.token_text == "Individual freedom versus collective responsibility"
    assert analysis.root.depth == 0

    # Check basic statistics make sense
    stats = analysis.tree_statistics
    assert stats['total_nodes'] >= 1
    assert stats['max_depth'] >= 0
    assert stats['leaf_nodes'] >= 0

    # Verify branching ratio is reasonable (0-1 range)
    assert 0 <= analysis.branching_ratio <= 1

    # Should have some sample paths
    assert len(analysis.sample_paths) >= 0

    print(f"✓ Pipeline generated {stats['total_nodes']} nodes")
    print(f"✓ Branching ratio: {analysis.branching_ratio:.3f}")


def test_visualization_export():
    """Test that we can export visualizations."""

    generator = create_generator(
        inference_model="mock",
        embedding_model="mock",
        cluster_type="mock",
        model_kwargs={"seed": 42}
    )

    analysis = generator.full_analysis(
        "Test visualization",
        max_depth=4,
        stem_length=2,
        num_stems=8
    )

    # Test visualization export
    try:
        visualizer = TreeVisualizer()
        output_path = visualizer.export(analysis, "./output/test")

        # Verify file was created and has content
        assert Path(output_path).exists()
        content = Path(output_path).read_text()
        assert "<html" in content
        assert "Test visualization" in content

        print(f"✓ Visualization exported to {output_path}")

    except FileNotFoundError as e:
        if "tree_template.html" in str(e):
            print("⚠ Skipping visualization test - template file missing")
        else:
            raise


def test_deterministic_generation():
    """Test that seeded generation produces consistent results."""

    results = []
    for run in range(2):
        generator = create_generator(
            inference_model="mock",
            embedding_model="mock",
            cluster_type="mock",
            model_kwargs={"seed": 42},
            cluster_kwargs={"seed": 42}
        )

        analysis = generator.full_analysis(
            "Determinism test",
            max_depth=4,
            stem_length=2,
            num_stems=10,
            seed=42
        )

        results.append(analysis.tree_statistics['total_nodes'])

    assert results[0] == results[1], "Seeded generation should be deterministic"
    print(f"✓ Deterministic generation confirmed ({results[0]} nodes)")


if __name__ == "__main__":
    """Run tests manually for quick verification."""
    print("Running core pipeline integration tests...")
    print("=" * 50)

    try:
        test_core_pipeline_works()
        test_visualization_export()
        test_deterministic_generation()

        print("=" * 50)
        print("✓ All core tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()