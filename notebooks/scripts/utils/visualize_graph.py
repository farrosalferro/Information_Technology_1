import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import to_rgba
import matplotlib.cm as cm
from pathlib import Path

# Use absolute imports instead of relative imports
from scripts.utils import dataset, camera
from scripts.features import extraction, clustering, matching


def visualize_graph(G, clusters, outliers, output_path=None, title=None, figsize=(12, 12)):
    """
    Visualize the view graph with nodes colored by cluster.
    
    Args:
        G: NetworkX graph representing the view graph
        clusters: List of clusters (each cluster is a list of node ids)
        outliers: List of outlier node ids
        output_path: Path to save the figure (if None, displays interactively)
        title: Title for the visualization
        figsize: Figure size as (width, height) tuple
    """
    if G.number_of_nodes() == 0:
        print("Empty graph, nothing to visualize.")
        return
    
    plt.figure(figsize=figsize)
    
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title(f"View Graph Visualization: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", 
                 fontsize=16)
    
    # Generate a color palette for the clusters
    cmap = cm.get_cmap('tab10')
    colors = {node: [0.7, 0.7, 0.7, 0.8] for node in G.nodes()}  # Default color (gray) for all nodes
    
    # Color nodes by cluster, with different color for each cluster
    for i, cluster in enumerate(clusters):
        cluster_color = to_rgba(cmap(i % 10))
        for node in cluster:
            colors[node] = cluster_color
    
    # Mark outliers with a different color (red)
    for node in outliers:
        if node in G.nodes():
            colors[node] = [1.0, 0.0, 0.0, 0.8]
    
    # Get node colors in the correct order
    node_colors = [colors[node] for node in G.nodes()]
    
    # Get edge weights for thickness
    edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + 4.0 * (weight / max_weight) for weight in edge_weights]
    
    # Determine graph layout based on graph size
    if G.number_of_nodes() < 100:
        pos = nx.spring_layout(G, seed=42)  # More aesthetic but slower for larger graphs
    else:
        pos = nx.kamada_kawai_layout(G)  # Better for larger graphs
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')
    
    # Add labels only if the graph is small enough to make them readable
    if G.number_of_nodes() < 30:
        # Use short labels (extract just the filename without extension)
        labels = {node: Path(node).stem for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="black")
    
    # Add legend
    legend_items = []
    for i, cluster in enumerate(clusters):
        legend_items.append(plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=to_rgba(cmap(i % 10)), 
                                     markersize=10, label=f'Cluster {i+1} ({len(cluster)} images)'))
    if outliers:
        legend_items.append(plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor='red', 
                                     markersize=10, label=f'Outliers ({len(outliers)} images)'))
        
    plt.legend(handles=legend_items, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    
    # Add annotations about graph properties
    info_text = (
        f"Total images: {G.number_of_nodes()}\n"
        f"Connections: {G.number_of_edges()}\n"
        f"Clusters: {len(clusters)}\n"
        f"Outliers: {len(outliers)}"
    )
    plt.figtext(0.02, 0.02, info_text, va='bottom', ha='left', 
                bbox={'facecolor': 'white', 'alpha': 0.8, 'boxstyle': 'round,pad=0.5'})
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Graph visualization saved to {output_path}")
    else:
        plt.show()
    
    return plt


def visualize_dataset_graph(dataset_id, data_dir, 
                           feature_extractor_type='SIFT', 
                           matcher_type='FLANN',
                           min_inlier_matches=10,
                           clustering_algorithm='ConnectedComponents',
                           min_cluster_size=3,
                           output_dir=None,
                           show_fig=True):
    """
    Extract features, build and visualize the view graph for a dataset.
    
    Args:
        dataset_id: ID of the dataset to process
        data_dir: Root directory containing the dataset
        feature_extractor_type: Type of feature extractor to use
        matcher_type: Type of matcher to use
        min_inlier_matches: Minimum inlier matches to form an edge
        clustering_algorithm: Algorithm for clustering the graph
        min_cluster_size: Minimum cluster size
        output_dir: Directory to save the visualization
        show_fig: Whether to display the figure interactively
    
    Returns:
        The NetworkX graph object
    """
    
    # Get samples for the dataset
    all_samples = dataset.load_dataset(data_dir)
    if dataset_id not in all_samples:
        raise ValueError(f"Dataset '{dataset_id}' not found in {data_dir}")
    
    samples = all_samples[dataset_id]
    print(f"Processing dataset '{dataset_id}' with {len(samples)} images...")
    
    # Initialize feature extractor and matcher
    extractor = extraction.get_feature_extractor(feature_extractor_type)
    matcher = matching.get_matcher(matcher_type, feature_extractor_type)
    
    # Extract features
    test_image_dir = os.path.join(data_dir, "train")
    extracted_features, image_dims = extraction.load_and_extract_features_dataset(
        dataset_id, test_image_dir, extractor
    )
    image_ids = list(extracted_features.keys())
    
    # Build view graph
    G, pairwise_matches = clustering.build_view_graph(
        image_ids, extracted_features, matcher, min_inlier_matches_graph=min_inlier_matches
    )
    
    # Cluster images
    clusters, outliers = clustering.cluster_images(
        G, algorithm=clustering_algorithm, min_cluster_size=min_cluster_size
    )
    
    # Generate output path and create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, f"{dataset_id}_graph.png") if not show_fig else None
    
    # Visualize the graph
    title = f"View Graph for Dataset: {dataset_id}"
    fig = visualize_graph(G, clusters, outliers, output_dir, title)
    
    if show_fig:
        plt.show()
    
    return G, clusters, outliers
