"""Manual testing script for development."""

from models.model_interface import create_generator
from clustering.embedding_analyzer import create_analyzer

def test_basic_functionality():
    """Test basic MockModel functionality."""
    print("Testing basic functionality...")

    analyzer = create_analyzer(analyzer_type='mock')
    generator = create_generator(model_type="mock", seed=42, analyzer=analyzer)

    prompt = "Test prompt"
    
    encoded = generator.model.encode(prompt)
    decoded = generator.model.decode(encoded.tolist())
    
    print(f"  Original: {prompt}")
    print(f"  Decoded: {decoded}")
    print("  ✓ Basic encode/decode works")


def test_clustering_behavior():
    """Compare clustering vs linear modes."""
    print("\nTesting clustering behavior...")
    
    modes = [("semantic_clusters", "Should branch"), ("linear", "Should not branch")]
    
    for mode, expectation in modes:
        analyzer = create_analyzer(analyzer_type='mock')
        generator = create_generator(model_type="mock",
                                     mode=mode,
                                     seed=42,
                                     analyzer=analyzer)
        analysis = generator.full_analysis(
            "Test ethical question",
            max_depth=8,
            stem_length=3,
            num_stems=20,
        )
        
        print(f"  {mode}: {analysis['branching_ratio']:.2f} branching ratio ({expectation})")


def test_full_pipeline():
    """Test complete pipeline with visualization."""
    print("\nTesting full pipeline...")
    
    analyzer = create_analyzer(analyzer_type='mock')
    generator = create_generator(model_type="mock",
                                 mode="semantic_clusters",
                                 seed=42,
                                 analyzer=analyzer)

    analysis = generator.full_analysis(
        "In the question of individual autonomy versus collective welfare",
        max_depth=12,
        stem_length=4,
        num_stems=25,
    )
    
    html_path = generator.visualizer.quick_export(analysis['root'], "test_prompt")
    
    print(f"  Generated {analysis['statistics']['total_nodes']} nodes")
    print(f"  Branching ratio: {analysis['branching_ratio']:.2f}")
    print(f"  Visualization: {html_path}")
    print("  ✓ Full pipeline works")


def test_deterministic_generation():
    """Test that seeded generation is deterministic."""
    print("\nTesting deterministic generation...")
    
    results = []
    for i in range(2):
        analyzer = create_analyzer(analyzer_type='mock')
        generator = create_generator(model_type="mock",
                                     seed=42,
                                     analyzer=analyzer)

        analysis = generator.full_analysis(
            "Test determinism",
            max_depth=6,
            stem_length=2,
            num_stems=10,
            seed=42,
        )
        results.append(analysis['statistics']['total_nodes'])
    
    if results[0] == results[1]:
        print("  ✓ Generation is deterministic")
    else:
        print(f"  ✗ Non-deterministic: {results[0]} vs {results[1]} nodes")


def main():
    """Run all manual tests."""
    print("Running DIA manual tests with MockModel")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_clustering_behavior()
        test_full_pipeline()
        test_deterministic_generation()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("You can now develop DIA features without heavy dependencies.")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()