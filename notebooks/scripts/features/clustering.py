from typing import Union

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import networkx as nx
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
from collections import defaultdict


def dino_clusterer(
    algorithm: str, features: np.ndarray, scaler: Union[None, str] = None, **kwargs
) -> np.ndarray:
    """Cluster the features using the specified algorithm.

    Args:
        algorithm: Clustering algorithm ('KMeans', 'DBSCAN', or 'Agglomerative').
        features: Input feature matrix.
        **kwargs: Additional arguments for the chosen algorithm.

    Returns:
        np.ndarray: Cluster labels for each feature.
    """
    if features.ndim != 2:
        raise ValueError(f"Input features should be a 2D array, got shape {features.shape}")

    # Apply scaling if specified
    if scaler == "StandardScaler":
        features = StandardScaler().fit_transform(features)
    elif scaler == "MinMaxScaler":
        features = MinMaxScaler().fit_transform(features)
    elif scaler is None or scaler == "None":
        pass
    else:
        raise ValueError(f"Unsupported scaler: {scaler}")

    if algorithm == "KMeans":
        clusterer = KMeans(**kwargs)
    elif algorithm == "DBSCAN":
        clusterer = DBSCAN(**kwargs)
    elif algorithm == "Agglomerative":
        clusterer = AgglomerativeClustering(**kwargs)
    elif algorithm == "HDBSCAN":
        clusterer = HDBSCAN(**kwargs)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return clusterer.fit_predict(features)

def build_view_graph(image_ids, features, matcher, min_inlier_matches_graph=10):
    """Builds a view graph based on pairwise matching."""
    from .matching import match_and_verify
    
    G = nx.Graph()
    G.add_nodes_from(image_ids)
    pairwise_matches = {} # Store matches for reuse in SfM

    print(f"Building view graph for {len(image_ids)} images...")
    # Optimized iteration: create pairs first
    pairs_to_match = []
    for i in range(len(image_ids)):
        for j in range(i + 1, len(image_ids)):
            pairs_to_match.append((image_ids[i], image_ids[j]))

    # Process pairs with progress bar
    match_results = {}
    for id1, id2 in tqdm(pairs_to_match, desc="Matching pairs"):

        kps1, descs1 = features.get(id1, (None, None))
        kps2, descs2 = features.get(id2, (None, None))

        if kps1 is None or kps2 is None:
            continue

        inlier_matches, num_inliers = match_and_verify(kps1, descs1, kps2, descs2, matcher)

        if inlier_matches is not None and num_inliers >= min_inlier_matches_graph:
             match_results[(id1, id2)] = (inlier_matches, num_inliers)
             G.add_edge(id1, id2, weight=num_inliers)
             # Store matches symmetrically for easier lookup during SfM
            #  pairwise_matches[(id1, id2)] = inlier_matches
             pairwise_matches[(id1, id2)] = [(m.queryIdx, m.trainIdx) for m in inlier_matches] # Keep query/train indices
             pairwise_matches[(id2, id1)] = [(m.trainIdx, m.queryIdx) for m in inlier_matches] # Swap query/train indices

    print(f"View graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G, pairwise_matches


def cluster_images(G, algorithm='ConnectedComponents', min_cluster_size=3):
    """Clusters the view graph and identifies outliers."""
    clusters = []
    outliers = []

    if G.number_of_nodes() == 0:
        return [], []

    print(f"Clustering graph using {algorithm}...")
    if algorithm == 'ConnectedComponents':
        # Use connected components as clusters
        components = list(nx.connected_components(G))
        for comp in components:
            if len(comp) >= min_cluster_size:
                clusters.append(list(comp))
            else:
                outliers.extend(list(comp)) # Add small components to outliers
    elif algorithm == 'Spectral':
         # Requires estimating number of clusters (k) - this is tricky
         # Heuristic: sqrt(nodes) or based on eigenvalues, but can be unstable
         num_nodes = G.number_of_nodes()
         if num_nodes < 2 : return [], list(G.nodes())

         # Estimate k (simple heuristic, might need improvement)
         k_estimated = max(2, min(10, int(np.sqrt(num_nodes / 5)))) # Cap at 10
         print(f"Estimating k={k_estimated} for Spectral Clustering")

         adj_matrix = nx.to_scipy_sparse_array(G, weight='weight', format='csr')
         sc = SpectralClustering(n_clusters=k_estimated,
                                 affinity='precomputed',
                                 assign_labels='kmeans', # or 'discretize'
                                 random_state=42)
         try:
             labels = sc.fit_predict(adj_matrix)
             # Group nodes by label
             temp_clusters = defaultdict(list)
             image_ids = list(G.nodes())
             for i, label in enumerate(labels):
                 temp_clusters[label].append(image_ids[i])

             for label, members in temp_clusters.items():
                 if len(members) >= min_cluster_size:
                     clusters.append(members)
                 else:
                     outliers.extend(members)

         except Exception as e:
              print(f"Spectral clustering failed: {e}. Falling back to Connected Components.")
              # Fallback
              components = list(nx.connected_components(G))
              for comp in components:
                  if len(comp) >= min_cluster_size:
                      clusters.append(list(comp))
                  else:
                      outliers.extend(list(comp))

    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    # Identify nodes that were in the original list but not in the graph (no edges)
    isolated_nodes = set(G.nodes()) - set(node for cluster in clusters for node in cluster) - set(outliers)
    outliers.extend(list(isolated_nodes))

    print(f"Found {len(clusters)} clusters and {len(outliers)} potential outliers.")
    return clusters, list(set(outliers)) # Ensure outliers are unique