"""Test clustering functionality with focused, non-redundant tests."""

import numpy as np
import pytest
from clustering.mock_clustering import MockClusterAnalyzer
from semantic_embedding.mock_embedding import MockEmbeddingProvider


class TestMockComponents:
    """Test mock components work correctly."""

    @pytest.fixture
    def embedding_provider(self):
        return MockEmbeddingProvider(seed=42)

    @pytest.fixture
    def test_stems(self):
        return [
            "individual freedom is paramount",
            "personal autonomy should be protected",
            "collective welfare comes first",
            "society needs unity",
            "research shows evidence",
            "data supports this view"
        ]

    def test_embedding_semantic_clustering(self, embedding_provider, test_stems):
        """Test that semantically similar texts produce similar embeddings."""
        embeddings = embedding_provider.get_embeddings(test_stems)

        # Verify embeddings have expected properties
        assert embeddings.shape == (len(test_stems), 384)
        assert np.all(np.isfinite(embeddings))

        # Test semantic similarity hints work
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)

        # Individual freedom stems (0,1) should be more similar than individual vs collective (0,2)
        assert similarities[0][1] > similarities[0][2]

    def test_cluster_analyzer_modes(self, embedding_provider, test_stems):
        """Test different clustering modes produce expected results."""
        embeddings = embedding_provider.get_embeddings(test_stems)

        test_cases = [
            ("simple", True, 2),  # Should find 2 clusters and branch
            ("no_clusters", False, 1),  # Should find 1 cluster, no branching
            ("alternating", True, lambda x: 2 <= x <= 3)  # Should find 2-3 clusters
        ]

        for mode, should_branch, expected_clusters in test_cases:
            analyzer = MockClusterAnalyzer(mode=mode, seed=42)
            result = analyzer.analyze_clusters(embeddings)

            assert result.has_branching == should_branch
            if callable(expected_clusters):
                assert expected_clusters(result.num_clusters)
            else:
                assert result.num_clusters == expected_clusters

    def test_representative_selection(self, embedding_provider, test_stems):
        """Test cluster representative selection."""
        embeddings = embedding_provider.get_embeddings(test_stems)
        analyzer = MockClusterAnalyzer(mode="simple", seed=42)

        result = analyzer.analyze_clusters(embeddings)
        representatives = analyzer.get_cluster_representatives(test_stems, result, embeddings)

        # Should get one representative per cluster
        assert len(representatives) == result.num_clusters
        assert all(rep in test_stems for rep in representatives)


class TestRealComponents:
    """Test real components when available."""

    @pytest.fixture
    def real_components(self):
        """Skip if real components not available."""
        try:
            from clustering.dbscan_clustering import DBSCANClusterAnalyzer
            from semantic_embedding.sentence_embedding import SentenceEmbeddingProvider
            return SentenceEmbeddingProvider(), DBSCANClusterAnalyzer(eps=0.35, min_sample_ratio=0.2)
        except ImportError:
            pytest.skip("Real components not available")

    def test_dbscan_clustering(self, real_components):
        """Test DBSCAN with sentence embeddings."""
        embedding_provider, cluster_analyzer = real_components

        test_stems = [
            "individual freedom matters most",
            "personal autonomy is key",
            "collective welfare comes first",
            "society needs unity"
        ]

        embeddings = embedding_provider.get_embeddings(test_stems)
        result = cluster_analyzer.analyze_clusters(embeddings)
        representatives = cluster_analyzer.get_cluster_representatives(test_stems, result, embeddings)

        # Basic sanity checks
        assert len(result.labels) == len(test_stems)
        assert result.num_clusters >= 0
        assert len(representatives) <= len(test_stems)


def test_embedding_determinism():
    """Test that embeddings are deterministic with same input."""
    provider = MockEmbeddingProvider(seed=42)
    text = "test text for determinism"

    emb1 = provider.get_embeddings([text])
    emb2 = provider.get_embeddings([text])

    np.testing.assert_array_equal(emb1, emb2)


def test_edge_cases():
    """Test edge cases that could break the system."""
    provider = MockEmbeddingProvider()
    analyzer = MockClusterAnalyzer()

    # Empty input
    empty_embeddings = provider.get_embeddings([])
    result = analyzer.analyze_clusters(empty_embeddings)
    assert result.num_clusters == 0
    assert not result.has_branching

    # Single item
    single_embedding = provider.get_embeddings(["single text"])
    result = analyzer.analyze_clusters(single_embedding)
    reps = analyzer.get_cluster_representatives(["single text"], result, single_embedding)
    assert len(reps) <= 1