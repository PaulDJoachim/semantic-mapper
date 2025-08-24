"""Test 3D visualization pipeline end-to-end."""

import pytest
import numpy as np
from clustering.embedding_to_3d import EmbeddingTo3D
from clustering.cluster_analyzer import ClusteringResult
from clustering.mock_clustering import MockClusterAnalyzer
from semantic_embedding.mock_embedding import MockEmbeddingProvider
from models.model_interface import create_generator


class Test3DPipeline:
    """Test 3D cluster visualization data generation."""

    @pytest.fixture
    def semantic_stems(self):
        """Stems designed to form semantic clusters."""
        return [
            "individual freedom is paramount",
            "personal autonomy matters most", 
            "collective welfare comes first",
            "society needs unity",
            "community solidarity is key"
        ]

    @pytest.fixture
    def clustering_components(self):
        """Mock components for testing."""
        embedding_provider = MockEmbeddingProvider(seed=42)
        cluster_analyzer = MockClusterAnalyzer(mode="simple", seed=42)
        return embedding_provider, cluster_analyzer

    def test_embedding_to_3d_conversion(self, semantic_stems, clustering_components):
        """Test that embeddings convert to valid 3D positions."""
        embedding_provider, cluster_analyzer = clustering_components
        
        embeddings = embedding_provider.get_embeddings(semantic_stems)
        clustering_result = cluster_analyzer.analyze_clusters(embeddings)
        
        viz_data = EmbeddingTo3D.create_visualization_data(
            clustering_result, semantic_stems
        )
        
        # Verify structure
        assert 'samples' in viz_data
        assert 'stats' in viz_data
        assert len(viz_data['samples']) == len(semantic_stems)
        
        # Verify 3D positions
        for sample in viz_data['samples']:
            assert 'direction' in sample
            assert len(sample['direction']) == 3
            
            # Positions should be normalized (roughly unit vectors)
            direction = np.array(sample['direction'])
            magnitude = np.linalg.norm(direction)
            assert 0.8 <= magnitude <= 1.2  # Allow some numerical precision

    def test_cluster_color_assignment(self, semantic_stems, clustering_components):
        """Test that clusters get distinct colors."""
        embedding_provider, cluster_analyzer = clustering_components
        
        embeddings = embedding_provider.get_embeddings(semantic_stems)
        clustering_result = cluster_analyzer.analyze_clusters(embeddings)
        
        viz_data = EmbeddingTo3D.create_visualization_data(
            clustering_result, semantic_stems
        )
        
        # Extract unique colors
        colors = set(sample['color'] for sample in viz_data['samples'])
        
        # Should have colors for each cluster
        assert len(colors) >= clustering_result.num_clusters

    def test_full_generator_with_3d_data(self):
        """Test complete generator pipeline stores 3D data."""
        generator = create_generator(
            model_type="mock",
            model_kwargs={"mode": "semantic_clusters", "seed": 42},
            analyzer_type="mock",
            cluster_kwargs={"seed": 42}
        )
        
        root = generator.explore_topology(
            "Test semantic branching",
            max_depth=6,
            stem_length=2,
            num_stems=10,
            seed=42
        )
        
        # Find nodes with cluster data
        nodes_with_clusters = []
        def collect_cluster_nodes(node):
            if node.cluster_data:
                nodes_with_clusters.append(node)
        
        from tree_utils import TreeOperations
        TreeOperations.traverse_depth_first(root, collect_cluster_nodes)
        
        assert len(nodes_with_clusters) > 0, "Should have nodes with cluster data"
        
        # Verify cluster data structure
        cluster_node = nodes_with_clusters[0]
        assert 'samples' in cluster_node.cluster_data
        assert 'stats' in cluster_node.cluster_data
        
        for sample in cluster_node.cluster_data['samples']:
            assert all(key in sample for key in ['text', 'direction', 'cluster', 'color'])

    def test_3d_fallback_without_sklearn(self, semantic_stems):
        """Test 3D conversion works without sklearn."""
        # Create mock embeddings
        embeddings = np.random.randn(len(semantic_stems), 384)
        labels = [0, 0, 1, 1, 1]
        
        clustering_result = ClusteringResult(labels, 2, True, embeddings)
        
        # This should work even if sklearn unavailable
        viz_data = EmbeddingTo3D.create_visualization_data(
            clustering_result, semantic_stems
        )
        
        assert len(viz_data['samples']) == len(semantic_stems)
        for sample in viz_data['samples']:
            direction = np.array(sample['direction'])
            assert len(direction) == 3
            assert np.linalg.norm(direction) > 0

    def test_edge_cases(self):
        """Test edge cases that could break 3D conversion."""
        # Empty clustering result
        empty_result = ClusteringResult([], 0, False, np.array([]).reshape(0, 384))
        viz_data = EmbeddingTo3D.create_visualization_data(empty_result, [])
        assert viz_data['samples'] == []
        
        # Single item
        single_embedding = np.random.randn(1, 384)
        single_result = ClusteringResult([0], 1, False, single_embedding)
        viz_data = EmbeddingTo3D.create_visualization_data(single_result, ["single"])
        assert len(viz_data['samples']) == 1


def test_3d_data_in_json_export():
    """Test that 3D data is included in tree JSON export."""
    generator = create_generator(
        model_type="mock", 
        model_kwargs={"seed": 42},
        analyzer_type="mock"
    )
    
    root = generator.explore_topology(
        "Test export", 
        max_depth=4, 
        stem_length=1, 
        num_stems=6
    )
    
    from tree_utils import TreeOperations
    tree_json = TreeOperations.to_json(root)
    
    # Should contain cluster_data
    assert 'cluster_data' in tree_json


if __name__ == "__main__":
    # Simple manual runner like your existing tests
    print("Testing 3D visualization pipeline...")
    
    test = Test3DPipeline()
    stems = test.semantic_stems()
    components = test.clustering_components()
    
    print("✓ Testing 3D conversion...")
    test.test_embedding_to_3d_conversion(stems, components)
    
    print("✓ Testing color assignment...")  
    test.test_cluster_color_assignment(stems, components)
    
    print("✓ Testing full pipeline...")
    test.test_full_generator_with_3d_data()
    
    print("✓ Testing edge cases...")
    test.test_edge_cases()
    
    print("✓ Testing JSON export...")
    test_3d_data_in_json_export()
    
    print("\n3D pipeline tests passed! 🎉")