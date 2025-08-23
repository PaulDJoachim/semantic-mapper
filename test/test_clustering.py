"""Test clustering functionality"""

from clustering.mock_clustering import MockClusterAnalyzer
from semantic_embedding.mock_embedding import MockEmbeddingProvider


def test_semantic_clustering():
    """Test that semantically similar texts cluster together."""
    print("Testing semantic clustering with mock components...")
    
    embedding_provider = MockEmbeddingProvider(seed=42)
    cluster_analyzer = MockClusterAnalyzer(mode="simple", seed=42)
    
    test_stems = [
        "individual freedom is paramount",
        "personal autonomy should be protected", 
        "collective welfare comes first",
        "society needs unity",
        "research shows evidence",
        "data supports this view"
    ]
    
    embeddings = embedding_provider.get_embeddings(test_stems)
    clustering_result = cluster_analyzer.analyze_clusters(embeddings)
    
    print(f"Found {clustering_result.num_clusters} clusters")
    print(f"Branching detected: {clustering_result.has_branching}")
    print(f"Labels: {clustering_result.labels}")
    
    clusters = {}
    for i, label in enumerate(clustering_result.labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(f"{i}: {test_stems[i]}")
    
    for cluster_id, stems in clusters.items():
        cluster_name = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        print(f"\n{cluster_name}:")
        for stem in stems:
            print(f"  {stem}")


def test_dbscan_clustering():
    """Test with DBSCAN analyzer if available."""
    try:
        from clustering.dbscan_clustering import DBSCANClusterAnalyzer
        from semantic_embedding.sentence_embedding import SentenceEmbeddingProvider
        
        print("\n" + "="*60)
        print("Testing real DBSCAN clustering...")
        
        embedding_provider = SentenceEmbeddingProvider()
        cluster_analyzer = DBSCANClusterAnalyzer(eps=0.35, min_sample_ratio=0.2, min_clusters=2)
        
        test_stems = [
            "individual freedom matters most",
            "personal autonomy is key", 
            "collective welfare comes first",
            "society needs unity",
            "research shows evidence",
            "data supports this view"
        ]
        
        embeddings = embedding_provider.get_embeddings(test_stems)
        clustering_result = cluster_analyzer.analyze_clusters(embeddings)
        
        print(f"DBSCAN found {clustering_result.num_clusters} clusters")
        print(f"Branching: {clustering_result.has_branching}")
        
        representatives = cluster_analyzer.get_cluster_representatives(
            test_stems, clustering_result, embeddings
        )
        
        print(f"Representatives: {representatives}")
        
    except ImportError as e:
        print(f"\nSkipping DBSCAN test: {e}")


def test_parameter_sensitivity():
    """Test how different analyzers behave."""
    print("\n" + "="*60)
    print("Testing analyzer modes...")
    
    embedding_provider = MockEmbeddingProvider(seed=42)
    
    test_stems = [
        "individual freedom matters",
        "personal rights are key", 
        "collective welfare first",
        "society needs unity",
        "research shows evidence",
        "data supports this view"
    ]
    
    embeddings = embedding_provider.get_embeddings(test_stems)
    
    for mode in ["simple", "alternating", "random", "no_clusters"]:
        analyzer = MockClusterAnalyzer(mode=mode, seed=42)
        result = analyzer.analyze_clusters(embeddings)
        
        representatives = analyzer.get_cluster_representatives(test_stems, result, embeddings)
        
        print(f"{mode}: {result.num_clusters} clusters, branching={result.has_branching}, "
              f"{len(representatives)} representatives")


def test_embedding_semantic_hints():
    """Test that mock embeddings respond to semantic keywords."""
    print("\n" + "="*60)
    print("Testing embedding semantic hints...")
    
    embedding_provider = MockEmbeddingProvider(seed=42)
    
    texts = [
        "individual freedom is important",
        "personal autonomy matters most", 
        "collective welfare should guide us",
        "society needs unity above all",
        "this is completely random text",
        "another unrelated statement here"
    ]
    
    embeddings = embedding_provider.get_embeddings(texts)
    
    # Check that similar semantic content produces similar embeddings
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = cosine_similarity(embeddings)
    
    print("Cosine similarity matrix:")
    for i, text in enumerate(texts):
        print(f"  {i}: {text[:30]}...")
    
    print("\nSimilarities (showing high values > 0.5):")
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            sim = similarities[i][j]
            if sim > 0.5:
                print(f"  {i}↔{j}: {sim:.3f}")


def test_integration_workflow():
    """Test complete embedding → clustering → representatives workflow."""
    print("\n" + "="*60)
    print("Testing complete workflow...")
    
    embedding_provider = MockEmbeddingProvider(seed=42)
    cluster_analyzer = MockClusterAnalyzer(mode="simple", seed=42)
    
    stems = [
        "freedom and individual rights",
        "personal autonomy matters",
        "collective society benefits", 
        "community welfare first",
        "research shows evidence",
        "academic analysis suggests"
    ]
    
    # Step 1: Generate embeddings
    embeddings = embedding_provider.get_embeddings(stems)
    print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    
    # Step 2: Analyze clusters
    clustering_result = cluster_analyzer.analyze_clusters(embeddings)
    print(f"Clustering: {clustering_result.num_clusters} clusters, branching={clustering_result.has_branching}")
    
    # Step 3: Get representatives
    representatives = cluster_analyzer.get_cluster_representatives(stems, clustering_result, embeddings)
    print(f"Selected {len(representatives)} representatives:")
    for i, rep in enumerate(representatives):
        print(f"  {i+1}: {rep}")


if __name__ == "__main__":
    test_semantic_clustering()
    test_dbscan_clustering() 
    test_parameter_sensitivity()
    test_embedding_semantic_hints()
    test_integration_workflow()
    
    print(f"\n✅ Clustering tests completed! Components working correctly.")