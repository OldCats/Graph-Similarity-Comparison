import streamlit as st
import networkx as nx
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.linalg import eigvals

def create_sample_graphs():
    # First graph
    G1 = nx.Graph()
    G1.add_edges_from([
        (1, 2), (2, 3), (3, 4), (4, 1), (2, 4)
    ])
    
    # Second graph - similar structure but different node labels
    G2 = nx.Graph()
    G2.add_edges_from([
        (5, 6), (6, 7), (7, 8), (8, 5), (6, 8)
    ])
    
    return G1, G2

def parse_edge_input(edge_text):
    """Parse edge input text into a list of tuples"""
    edges = []
    try:
        # Split the input into lines and process each line
        lines = edge_text.strip().split('\n')
        for line in lines:
            if line.strip():
                # Convert "1,2" or "1 2" format to tuple
                nodes = line.replace(',', ' ').split()
                if len(nodes) >= 2:
                    edges.append((int(nodes[0]), int(nodes[1])))
        return edges
    except ValueError:
        st.error("Invalid input format. Please use format: 'node1 node2' or 'node1,node2' per line")
        return None

def compare_structural_similarity(G1, G2):
    """Compare graphs based on structural properties"""
    
    similarity_metrics = {
        'Number of Nodes Match': len(G1) == len(G2),
        'Number of Edges Match': len(G1.edges()) == len(G2.edges()),
        'Degree Sequence Match': sorted([d for n, d in G1.degree()]) == 
                                sorted([d for n, d in G2.degree()]),
        'Density Match': abs(nx.density(G1) - nx.density(G2)) < 1e-9,
        'Average Path Length Match': abs(nx.average_shortest_path_length(G1) - 
                                      nx.average_shortest_path_length(G2)) < 1e-9,
        'Diameter Match': nx.diameter(G1) == nx.diameter(G2)
    }
    
    return similarity_metrics

def compare_node_similarity(G1, G2):
    """Compare node-level properties between graphs"""
    
    deg_dist1 = Counter([d for n, d in G1.degree()])
    deg_dist2 = Counter([d for n, d in G2.degree()])
    
    clustering1 = nx.average_clustering(G1)
    clustering2 = nx.average_clustering(G2)
    
    # Calculate centrality measures
    betweenness1 = nx.betweenness_centrality(G1)
    betweenness2 = nx.betweenness_centrality(G2)
    
    closeness1 = nx.closeness_centrality(G1)
    closeness2 = nx.closeness_centrality(G2)
    
    node_metrics = {
        'Degree Distribution Match': deg_dist1 == deg_dist2,
        'Clustering Coefficient Difference': abs(clustering1 - clustering2),
        'Average Betweenness Difference': abs(sum(betweenness1.values())/len(betweenness1) - 
                                            sum(betweenness2.values())/len(betweenness2)),
        'Average Closeness Difference': abs(sum(closeness1.values())/len(closeness1) - 
                                          sum(closeness2.values())/len(closeness2))
    }
    
    return node_metrics

def plot_graphs(G1, G2):
    """Create a matplotlib figure with both graphs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot first graph with spring layout
    pos1 = nx.spring_layout(G1)
    nx.draw(G1, pos1, ax=ax1, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=16)
    ax1.set_title("Graph 1")
    
    # Plot second graph with spring layout
    pos2 = nx.spring_layout(G2)
    nx.draw(G2, pos2, ax=ax2, with_labels=True, node_color='lightgreen', 
            node_size=500, font_size=16)
    ax2.set_title("Graph 2")
    
    plt.tight_layout()
    return fig

def calculate_matrix_similarities(G1, G2):
    """Calculate similarities based on adjacency matrices"""
    # Get adjacency matrices
    A1 = nx.adjacency_matrix(G1).todense()
    A2 = nx.adjacency_matrix(G2).todense()
    
    # Jaccard similarity
    A1_flat = np.array(A1).flatten()
    A2_flat = np.array(A2).flatten()
    intersection = np.sum(np.logical_and(A1_flat, A2_flat))
    union = np.sum(np.logical_or(A1_flat, A2_flat))
    jaccard = intersection / union if union != 0 else 0
    
    # Cosine similarity
    cosine_sim = 1 - cosine(A1_flat, A2_flat) if np.any(A1_flat) and np.any(A2_flat) else 0
    
    # Spectral similarity (using top k eigenvalues)
    k = min(len(G1), len(G2))
    eig1 = sorted(np.real(eigvals(A1)))[-k:]
    eig2 = sorted(np.real(eigvals(A2)))[-k:]
    spectral_diff = np.linalg.norm(np.array(eig1) - np.array(eig2))
    spectral_sim = 1 / (1 + spectral_diff)
    
    # Graph Edit Distance similarity (normalized)
    try:
        ged = nx.graph_edit_distance(G1, G2)
        max_possible_ged = max(len(G1) + len(G2.edges()), len(G2) + len(G1.edges()))
        ged_sim = 1 - (ged / max_possible_ged if max_possible_ged > 0 else 0)
    except:
        ged_sim = 0  # Fallback if GED calculation fails
    
    return {
        'Jaccard Similarity': jaccard,
        'Cosine Similarity': cosine_sim,
        'Spectral Similarity': spectral_sim,
        'Graph Edit Distance Similarity': ged_sim
    }

def get_metric_explanations():
    """Return explanations for each metric"""
    base_explanations = {
        # Structural Metrics
        'Number of Nodes Match': 'Checks if both graphs have the same number of vertices/nodes.',
        'Number of Edges Match': 'Checks if both graphs have the same number of edges/connections.',
        'Degree Sequence Match': 'Compares the sorted sequence of node degrees (number of connections per node) between graphs.',
        'Density Match': 'Compares the density (ratio of actual edges to possible edges) between graphs.',
        'Average Path Length Match': 'Compares the average shortest path length between all pairs of nodes.',
        'Diameter Match': 'Compares the maximum shortest path length between any two nodes in the graphs.',
        
        # Node-level Metrics
        'Degree Distribution Match': 'Compares the frequency distribution of node degrees between graphs.',
        'Clustering Coefficient Difference': 'Measures the difference in how much nodes tend to cluster together.',
        'Average Betweenness Difference': 'Compares the average betweenness centrality, which measures how often a node acts as a bridge.',
        'Average Closeness Difference': 'Compares the average closeness centrality, measuring how close each node is to all other nodes.',
        
        # Matrix Similarity Metrics
        'Jaccard Similarity': 'Measures similarity as ratio of common edges to total edges (1.0 = identical graphs)',
        'Cosine Similarity': 'Measures similarity as cosine of angle between adjacency matrices (1.0 = identical graphs)',
        'Spectral Similarity': 'Compares graph structure using eigenvalues of adjacency matrices (1.0 = identical graphs)',
        'Graph Edit Distance Similarity': 'Measures similarity based on minimum operations to transform one graph to another (1.0 = identical graphs)'
    }
    
    return base_explanations

def main():
    st.title("Graph Similarity Comparison")
    st.write("Compare two graphs for structural and node-level similarity")
    
    # Add help text
    with st.expander("ℹ️ How to use this app"):
        st.write("""
        1. Enter the edges for each graph in the text areas below
        2. Use one edge per line in format: 'node1 node2' or 'node1,node2'
        3. Nodes should be numbered (e.g., 1, 2, 3...)
        4. The app will compare the graphs and show various similarity metrics
        """)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Graph 1 Edges")
        graph1_input = st.text_area(
            "Enter edges (one per line, format: 'node1 node2' or 'node1,node2')",
            value="1 2\n2 3\n3 4\n4 1\n2 4",
            key="graph1"
        )
    
    with col2:
        st.subheader("Graph 2 Edges")
        graph2_input = st.text_area(
            "Enter edges (one per line, format: 'node1 node2' or 'node1,node2')",
            value="5 6\n6 7\n7 8\n8 5\n6 8",
            key="graph2"
        )
    
    # Create graphs from input
    edges1 = parse_edge_input(graph1_input)
    edges2 = parse_edge_input(graph2_input)
    
    if edges1 and edges2:
        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_edges_from(edges1)
        G2.add_edges_from(edges2)
        
        # Get metric explanations
        explanations = get_metric_explanations()
        
        # Compare and display results
        st.subheader("Comparison Results")
        
        # Matrix-based similarities
        st.write("Matrix-based Similarity Metrics:")
        matrix_sim = calculate_matrix_similarities(G1, G2)
        for metric, value in matrix_sim.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"- {metric}")
                st.caption(explanations[metric])
            with col2:
                st.write(f"{value:.4f}")
        
        # Structural similarity with explanations
        st.write("\nStructural Similarity Metrics:")
        struct_sim = compare_structural_similarity(G1, G2)
        for metric, value in struct_sim.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"- {metric}")
                st.caption(explanations[metric])
            with col2:
                st.write(f"{'✅' if value else '❌'}")
        
        # Node similarity with explanations
        st.write("\nNode-level Similarity Metrics:")
        node_sim = compare_node_similarity(G1, G2)
        for metric, value in node_sim.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"- {metric}")
                st.caption(explanations[metric])
            with col2:
                if isinstance(value, bool):
                    st.write(f"{'✅' if value else '❌'}")
                else:
                    st.write(f"{value:.4f}")
        
        # Update interpretation guide
        with st.expander("📖 How to interpret the results"):
            st.write("""
            - ✅ indicates exact matches between the two graphs for that metric
            - ❌ indicates differences between the graphs
            - Numerical similarity values range from 0.0 to 1.0:
                - 1.0 means perfectly similar
                - 0.0 means completely different
            - Difference values (closer to 0.0 is more similar)
            """)
        
        # Visualization
        st.subheader("Graph Visualization")
        fig = plot_graphs(G1, G2)
        st.pyplot(fig)

if __name__ == "__main__":
    main() 