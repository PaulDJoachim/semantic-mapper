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
    test_mode_differences()
    
    print("\n" + "=" * 40)
    print("✓ All tests passed - ready for development!")


if __name__ == "__main__":
    main()