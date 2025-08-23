#!/usr/bin/env python3
"""Test the MockEmbeddingAnalyzer with real DBSCAN clustering."""

import numpy as np
from clustering.mock_clustering import MockEmbeddingAnalyzer


def test_semantic_clustering():
    """Test that semantically similar texts cluster together."""
    print("Testing semantic clustering with DBSCAN...")
    
    analyzer = MockEmbeddingAnalyzer(eps=0.4, min_sample_ratio=0.2, seed=42)
    
    # Create stems with clear semantic groups
    test_stems = [
        # Individual rights cluster
        "individual freedom is paramount",
        "personal autonomy should be protected", 
        "rights of the individual matter most",
        "each person should choose freely",
        
        # Collective welfare cluster  
        "society benefits when we work together",
        "community needs come first",
        "collective action is most effective",
        "group welfare over individual wants",
        
        # Academic/research cluster
        "research shows evidence for this approach",
        "studies indicate the data supports",
        "analysis of the evidence suggests",
        
        # Mixed/noise
        "however we might consider both perspectives",
        "the situation is complex and nuanced"
    ]
    
    clustering_result = analyzer.cluster_stems(test_stems)
    
    print(f"Found {clustering_result.num_clusters} clusters")
    print(f"Branching detected: {clustering_result.has_branching}")
    print(f"Labels: {clustering_result.labels}")
    
    # Group stems by cluster
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


def test_parameter_sensitivity():
    """Test how eps parameter affects clustering."""
    print("\n" + "="*60)
    print("Testing parameter sensitivity...")
    
    test_stems = [
        "individual freedom matters",
        "personal rights are key", 
        "collective welfare first",
        "society needs unity",
        "research shows evidence",
        "data supports this view"
    ]
    
    for eps in [0.2, 0.4, 0.6, 0.8]:
        analyzer = MockEmbeddingAnalyzer(eps=eps, min_sample_ratio=0.15, seed=42)
        result = analyzer.cluster_stems(test_stems)
        noise_count = sum(1 for label in result.labels if label == -1)
        
        print(f"eps={eps}: {result.num_clusters} clusters, {noise_count} noise points")


def test_embedding_distances():
    """Examine actual distances between mock embeddings."""
    print("\n" + "="*60)
    print("Testing embedding distances...")
    
    analyzer = MockEmbeddingAnalyzer(seed=42)
    
    # Test semantically similar and different texts
    texts = [
        "individual freedom is important",
        "personal autonomy matters most", 
        "collective welfare should guide us",
        "society needs unity above all"
    ]
    
    embeddings = [analyzer._text_to_embedding(text) for text in texts]
    
    from sklearn.metrics.pairwise import cosine_distances
    distances = cosine_distances(embeddings)
    
    print("Cosine distance matrix:")
    print("Texts:")
    for i, text in enumerate(texts):
        print(f"  {i}: {text}")
    
    print("\nDistances:")
    for i in range(len(texts)):
        for j in range(len(texts)):
            print(f"  {i}→{j}: {distances[i][j]:.3f}")


def test_3d_visualization_data():
    """Generate data for 3D visualization testing."""
    print("\n" + "="*60) 
    print("Generating 3D visualization data...")
    
    analyzer = MockEmbeddingAnalyzer(eps=0.35, seed=42)
    
    # Generate a larger set of diverse stems
    stems = []
    
    # Individual freedom themes
    for i in range(8):
        stems.append(f"individual autonomy and personal freedom variation {i}")
    
    # Collective welfare themes  
    for i in range(8):
        stems.append(f"collective society and community welfare version {i}")
        
    # Academic/research themes
    for i in range(6):
        stems.append(f"research evidence and academic analysis study {i}")
        
    # Add some noise
    for i in range(4):
        stems.append(f"random mixed content neither here nor there {i}")
    
    # Cluster the stems
    clustering_result = analyzer.cluster_stems(stems)
    embeddings = analyzer.get_embeddings_for_visualization(stems)
    
    print(f"Generated {len(stems)} stems")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Clusters found: {clustering_result.num_clusters}")
    print(f"Noise points: {sum(1 for l in clustering_result.labels if l == -1)}")
    
    return stems, embeddings, clustering_result.labels


if __name__ == "__main__":
    test_semantic_clustering()
    test_parameter_sensitivity()
    test_embedding_distances()
    
    stems, embeddings, labels = test_3d_visualization_data()
    print(f"\n✅ Mock DBSCAN working! Ready for 3D visualization with:")
    print(f"   {len(stems)} samples, {embeddings.shape[1]}D embeddings")
    print(f"   Clusters: {set(labels)}")