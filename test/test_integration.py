"""Integration tests for DIA pipeline using MockModel."""

import pytest
from models.model_interface import create_generator


def test_mock_model_basic():
    """Test MockModel basic functionality."""
    generator = create_generator(model_type="mock", seed=42)
    
    prompt = "Test prompt"
    encoded = generator.model.encode(prompt)
    decoded = generator.model.decode(encoded.tolist())
    
    assert isinstance(decoded, str)
    assert len(encoded.tolist()) > 0


def test_full_pipeline_clustering_mode():
    """Test complete pipeline with clustering mode."""
    generator = create_generator(
        model_type="mock", 
        mode="semantic_clusters", 
        seed=42
    )
    
    analysis = generator.full_analysis(
        "Test ethical question",
        max_depth=10,
        stem_length=3,
        num_stems=20
    )
    
    assert analysis['root'] is not None
    assert analysis['statistics']['total_nodes'] >= 1
    assert 'branching_ratio' in analysis


def test_full_pipeline_linear_mode():
    """Test pipeline with linear mode (should not branch much)."""
    generator = create_generator(
        model_type="mock", 
        mode="linear", 
        seed=42
    )
    
    analysis = generator.full_analysis(
        "Test prompt",
        max_depth=8,
        stem_length=3,
        num_stems=15
    )
    
    # Linear mode should have low branching
    assert analysis['branching_ratio'] <= 0.5


def test_visualization_export():
    """Test that visualization exports successfully."""
    generator = create_generator(model_type="mock", seed=42)
    
    root = generator.explore_topology(
        "Test visualization",
        max_depth=6,
        stem_length=2,
        num_stems=10
    )
    
    output_path = generator.visualizer.quick_export(root, "test")
    assert output_path.endswith('.html')


def test_different_mock_modes():
    """Test all MockModel modes work."""
    modes = ["semantic_clusters", "linear", "random"]
    
    for mode in modes:
        generator = create_generator(model_type="mock", mode=mode, seed=42)
        
        root = generator.explore_topology(
            f"Test {mode}",
            max_depth=5,
            stem_length=2,
            num_stems=8
        )
        
        assert root is not None
        assert root.children is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])