"""Integration tests for DIA pipeline using MockModel."""

import pytest
from models.model_interface import create_generator


def test_mock_model_basic():
    """Test MockModel basic functionality."""
    generator = create_generator(
        model_type="mock",
        model_kwargs={"seed": 42},
        analyzer_type="mock"
    )
    
    prompt = "Test prompt"
    encoded = generator.model.encode(prompt)
    decoded = generator.model.decode(encoded.tolist())
    
    assert isinstance(decoded, str)
    assert len(encoded.tolist()) > 0


def test_full_pipeline_clustering_mode():
    """Test complete pipeline with clustering mode."""
    generator = create_generator(
        model_type="mock",
        model_kwargs={"mode": "semantic_clusters", "seed": 42},
        analyzer_type="mock"
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
        model_kwargs={"mode": "linear", "seed": 42},
        analyzer_type="mock"
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
    generator = create_generator(
        model_type="mock",
        model_kwargs={"seed": 42},
        analyzer_type="mock"
    )
    
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
        generator = create_generator(
            model_type="mock",
            model_kwargs={"mode": mode, "seed": 42},
            analyzer_type="mock"
        )
        
        root = generator.explore_topology(
            f"Test {mode}",
            max_depth=5,
            stem_length=2,
            num_stems=8
        )
        
        assert root is not None
        assert root.children is not None


def test_component_interfaces():
    """Test that embedding and clustering components work together."""
    generator = create_generator(
        model_type="mock",
        model_kwargs={"seed": 42},
        analyzer_type="mock"
    )
    
    test_texts = ["individual freedom", "collective welfare", "research data"]
    
    # Test embedding provider
    embeddings = generator.embedding_provider.get_embeddings(test_texts)
    assert embeddings.shape[0] == len(test_texts)
    assert embeddings.shape[1] > 0
    
    # Test cluster analyzer
    clustering_result = generator.cluster_analyzer.analyze_clusters(embeddings)
    assert hasattr(clustering_result, 'labels')
    assert hasattr(clustering_result, 'num_clusters')
    assert hasattr(clustering_result, 'has_branching')
    
    # Test representative selection
    representatives = generator.cluster_analyzer.get_cluster_representatives(
        test_texts, clustering_result, embeddings
    )
    assert isinstance(representatives, list)
    assert len(representatives) <= len(test_texts)


def test_end_to_end_with_real_dbscan():
    """Test with real DBSCAN if available."""
    try:
        generator = create_generator(
            model_type="mock",
            model_kwargs={"mode": "semantic_clusters", "seed": 42},
            analyzer_type="sentence"
        )
        
        analysis = generator.full_analysis(
            "Individual rights versus collective welfare",
            max_depth=6,
            stem_length=2,
            num_stems=12
        )
        
        assert analysis['root'] is not None
        assert 'branching_ratio' in analysis
        
    except ImportError:
        pytest.skip("Sentence transformers not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])