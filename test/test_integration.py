"""Integration tests for DIA pipeline."""

import pytest
from pathlib import Path
from models.model_interface import create_generator


class TestPipelineIntegration:
    """Test complete pipeline integration."""
    
    @pytest.fixture
    def generator(self):
        """Create generator with deterministic components."""
        return create_generator(
            model_type="mock",
            model_kwargs={"mode": "semantic_clusters", "seed": 42},
            analyzer_type="mock",
            cluster_kwargs={"seed": 42}
        )

    def test_full_analysis_pipeline(self, generator):
        """Test complete analysis pipeline produces expected structure."""
        analysis = generator.full_analysis(
            "Test ethical question",
            max_depth=10,
            stem_length=3,
            num_stems=20
        )
        
        # Verify analysis structure
        required_keys = ['root', 'statistics', 'branching_ratio', 'average_path_length']
        assert all(key in analysis for key in required_keys)
        
        # Verify tree structure
        root = analysis['root']
        assert root.token_text == "Test ethical question"
        assert root.depth == 0
        
        # Verify statistics make sense
        stats = analysis['statistics']
        assert stats['total_nodes'] >= 1
        assert stats['max_depth'] >= 0
        assert 0 <= analysis['branching_ratio'] <= 1

    def test_mode_affects_branching(self):
        """Test that different model modes produce different branching behavior."""
        modes = [
            ("semantic_clusters", lambda br: br > 0.1),  # Should branch
            ("linear", lambda br: br < 0.5)  # Should branch less
        ]
        
        for mode, branching_check in modes:
            gen = create_generator(
                model_type="mock",
                model_kwargs={"mode": mode, "seed": 42},
                analyzer_type="mock"
            )
            
            analysis = gen.full_analysis("Test", max_depth=8, stem_length=2, num_stems=15)
            assert branching_check(analysis['branching_ratio'])

    def test_visualization_export(self, generator, tmp_path):
        """Test visualization export creates valid HTML."""
        root = generator.explore_topology(
            "Test visualization",
            max_depth=6,
            stem_length=2,
            num_stems=10
        )
        
        output_path = generator.visualizer.quick_export(root, "test")
        
        # Verify file exists and has content
        assert Path(output_path).exists()
        content = Path(output_path).read_text()
        assert "<html" in content
        assert "const treeData" in content
        assert "token_text" in content  # Verify JSON data is present

    def test_deterministic_generation(self):
        """Test that seeded generation is reproducible."""
        results = []
        
        for _ in range(2):
            gen = create_generator(
                model_type="mock",
                model_kwargs={"seed": 42},
                analyzer_type="mock",
                cluster_kwargs={"seed": 42}
            )
            
            analysis = gen.full_analysis(
                "Test determinism",
                max_depth=6,
                stem_length=2,
                num_stems=10,
                seed=42
            )
            results.append(analysis['statistics']['total_nodes'])
        
        assert results[0] == results[1], "Generation should be deterministic"


class TestComponentCompatibility:
    """Test that different component combinations work together."""
    
    def test_mock_components_compatibility(self):
        """Test all mock components work together."""
        gen = create_generator(
            model_type="mock",
            analyzer_type="mock"
        )
        
        # Test basic component interfaces
        test_texts = ["freedom", "collective", "research"]
        
        embeddings = gen.embedding_provider.get_embeddings(test_texts)
        result = gen.cluster_analyzer.analyze_clusters(embeddings)
        reps = gen.cluster_analyzer.get_cluster_representatives(test_texts, result, embeddings)
        
        assert embeddings.shape[0] == len(test_texts)
        assert len(reps) <= len(test_texts)

    def test_real_components_fallback(self):
        """Test fallback to mock when real components unavailable."""
        try:
            gen = create_generator(
                model_type="mock",
                analyzer_type="sentence"
            )
            # If we get here, real components are available
            assert hasattr(gen.embedding_provider, 'model')
        except ImportError:
            # Should fall back to mock
            gen = create_generator(
                model_type="mock",
                analyzer_type="mock"
            )
            assert hasattr(gen.embedding_provider, 'embedding_dim')


def test_parameter_validation():
    """Test that invalid parameters are handled gracefully."""
    gen = create_generator(model_type="mock", analyzer_type="mock")
    
    # Test with minimal parameters
    root = gen.explore_topology("Test", max_depth=1, stem_length=1, num_stems=5)
    assert root is not None


def test_edge_case_inputs():
    """Test edge cases that might break the pipeline."""
    gen = create_generator(model_type="mock", analyzer_type="mock")
    
    edge_cases = [
        "",  # Empty string
        "A",  # Single character
        "Very long prompt " * 20  # Very long prompt
    ]
    
    for prompt in edge_cases:
        try:
            analysis = gen.full_analysis(
                prompt,
                max_depth=3,
                stem_length=1,
                num_stems=5
            )
            assert analysis['root'] is not None
        except Exception as e:
            pytest.fail(f"Pipeline failed on edge case '{prompt[:20]}...': {e}")