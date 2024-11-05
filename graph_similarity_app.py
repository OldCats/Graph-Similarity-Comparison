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
    
    # Check if graphs are connected
    is_G1_connected = nx.is_connected(G1)
    is_G2_connected = nx.is_connected(G2)
    
    # Get connected components count
    G1_components = nx.number_connected_components(G1)
    G2_components = nx.number_connected_components(G2)
    
    # Calculate average path length and diameter only if graphs are connected
    try:
        avg_path_G1 = nx.average_shortest_path_length(G1) if is_G1_connected else None
        avg_path_G2 = nx.average_shortest_path_length(G2) if is_G2_connected else None
        diameter_G1 = nx.diameter(G1) if is_G1_connected else None
        diameter_G2 = nx.diameter(G2) if is_G2_connected else None
    except:
        avg_path_G1 = avg_path_G2 = diameter_G1 = diameter_G2 = None
    
    similarity_metrics = {
        'Number of Nodes Match': len(G1) == len(G2),
        'Number of Edges Match': len(G1.edges()) == len(G2.edges()),
        'Degree Sequence Match': sorted([d for n, d in G1.degree()]) == 
                                sorted([d for n, d in G2.degree()]),
        'Density Match': abs(nx.density(G1) - nx.density(G2)) < 1e-9,
        'Connected Components Match': G1_components == G2_components,
        'Both Graphs Connected': is_G1_connected and is_G2_connected
    }
    
    # Add path-based metrics only if both graphs are connected
    if is_G1_connected and is_G2_connected:
        similarity_metrics.update({
            'Average Path Length Match': abs(avg_path_G1 - avg_path_G2) < 1e-9,
            'Diameter Match': diameter_G1 == diameter_G2
        })
    
    return similarity_metrics

def compare_node_similarity(G1, G2):
    """Compare node-level properties between graphs"""
    
    deg_dist1 = Counter([d for n, d in G1.degree()])
    deg_dist2 = Counter([d for n, d in G2.degree()])
    
    clustering1 = nx.average_clustering(G1)
    clustering2 = nx.average_clustering(G2)
    
    # Calculate centrality measures for each component if graphs are disconnected
    def safe_centrality_calculation(G, centrality_func):
        if nx.is_connected(G):
            return centrality_func(G)
        else:
            # Calculate for each component and combine
            centrality_dict = {}
            for component in nx.connected_components(G):
                subgraph = G.subgraph(component)
                centrality_dict.update(centrality_func(subgraph))
            return centrality_dict
    
    # Calculate centrality measures safely
    betweenness1 = safe_centrality_calculation(G1, nx.betweenness_centrality)
    betweenness2 = safe_centrality_calculation(G2, nx.betweenness_centrality)
    
    closeness1 = safe_centrality_calculation(G1, nx.closeness_centrality)
    closeness2 = safe_centrality_calculation(G2, nx.closeness_centrality)
    
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
    pos1 = nx.spring_layout(G1, k=1)  # Increased k for better spacing of components
    nx.draw(G1, pos1, ax=ax1, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=16)
    ax1.set_title(f"Graph 1 ({nx.number_connected_components(G1)} components)")
    
    # Plot second graph with spring layout
    pos2 = nx.spring_layout(G2, k=1)  # Increased k for better spacing of components
    nx.draw(G2, pos2, ax=ax2, with_labels=True, node_color='lightgreen', 
            node_size=500, font_size=16)
    ax2.set_title(f"Graph 2 ({nx.number_connected_components(G2)} components)")
    
    plt.tight_layout()
    return fig

def calculate_matrix_similarities(G1, G2):
    """Calculate similarities based on adjacency matrices"""
    # Get the maximum number of nodes
    max_nodes = max(len(G1), len(G2))
    
    # Create padded adjacency matrices
    A1 = nx.adjacency_matrix(G1).todense()
    A2 = nx.adjacency_matrix(G2).todense()
    
    # Pad matrices to the same size
    A1_padded = np.zeros((max_nodes, max_nodes))
    A2_padded = np.zeros((max_nodes, max_nodes))
    
    A1_padded[:A1.shape[0], :A1.shape[1]] = A1
    A2_padded[:A2.shape[0], :A2.shape[1]] = A2
    
    # Flatten padded matrices
    A1_flat = A1_padded.flatten()
    A2_flat = A2_padded.flatten()
    
    # Jaccard similarity
    intersection = np.sum(np.logical_and(A1_flat, A2_flat))
    union = np.sum(np.logical_or(A1_flat, A2_flat))
    jaccard = intersection / union if union != 0 else 0
    
    # Cosine similarity
    cosine_sim = 1 - cosine(A1_flat, A2_flat) if np.any(A1_flat) and np.any(A2_flat) else 0
    
    # Spectral similarity (using top k eigenvalues)
    k = min(len(G1), len(G2))
    eig1 = sorted(np.real(eigvals(A1_padded)))[-k:]
    eig2 = sorted(np.real(eigvals(A2_padded)))[-k:]
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
        'Connected Components Match': 'Checks if both graphs have the same number of disconnected parts.',
        'Both Graphs Connected': 'Indicates whether both graphs are fully connected (no isolated parts).',
        'Average Path Length Match': 'Compares the average shortest path length (only for connected graphs).',
        'Diameter Match': 'Compares the maximum shortest path length (only for connected graphs).',
        
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
    st.set_page_config(layout="wide", page_title="Graph Similarity Comparison")
    
    # Title and description in a container
    with st.container():
        st.title("Graph Similarity Comparison")
        st.write("Compare two graphs for structural and node-level similarity")
    
    # Help section in a container
    with st.container():
        col1, col2 = st.columns([2, 1])
        with col1:
            with st.expander("ℹ️ How to use this app"):
                st.write("""
                1. Enter the edges for each graph in the text areas below
                2. Use one edge per line in format: 'node1 node2' or 'node1,node2'
                3. Nodes should be numbered (e.g., 1, 2, 3...)
                4. The app will compare the graphs and show various similarity metrics
                """)
        with col2:
            with st.expander("📖 How to interpret results"):
                st.write("""
                - ✅ indicates exact matches between the two graphs
                - ❌ indicates differences between the graphs
                - Similarity values (0.0 to 1.0):
                    - 1.0 = perfectly similar
                    - 0.0 = completely different
                - For differences, closer to 0.0 is more similar
                """)
    
    # Graph input section
    st.markdown("---")
    st.subheader("📊 Graph Input")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Graph 1")
        graph1_input = st.text_area(
            "Enter edges for Graph 1",
            value="1 2\n2 3\n3 4\n4 1\n2 4",
            key="graph1",
            height=150
        )
    
    with col2:
        st.markdown("### Graph 2")
        graph2_input = st.text_area(
            "Enter edges for Graph 2",
            value="5 6\n6 7\n7 8\n8 5\n6 8",
            key="graph2",
            height=150
        )
    
    # Process graphs and show results
    edges1 = parse_edge_input(graph1_input)
    edges2 = parse_edge_input(graph2_input)
    
    if edges1 and edges2:
        G1 = nx.Graph()
        G2 = nx.Graph()
        G1.add_edges_from(edges1)
        G2.add_edges_from(edges2)
        
        # Get metric explanations
        explanations = get_metric_explanations()
        
        # Visualization section first
        st.markdown("---")
        st.subheader("🎨 Graph Visualization")
        fig = plot_graphs(G1, G2)
        st.pyplot(fig)
        
        # Results section second
        st.markdown("---")
        st.subheader("📈 Comparison Results")
        
        # Create tabs for different metric categories
        tab1, tab2, tab3 = st.tabs([
            "Matrix-based Similarities", 
            "Structural Similarities", 
            "Node-level Similarities"
        ])
        
        # Matrix-based similarities tab
        with tab1:
            matrix_sim = calculate_matrix_similarities(G1, G2)
            for metric, value in matrix_sim.items():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{metric}**")
                        st.caption(explanations[metric])
                    with col2:
                        st.metric(
                            label=metric,
                            value=f"{value:.4f}",
                            label_visibility="collapsed"
                        )
        
        # Structural similarities tab
        with tab2:
            struct_sim = compare_structural_similarity(G1, G2)
            for metric, value in struct_sim.items():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{metric}**")
                        st.caption(explanations[metric])
                    with col2:
                        st.write(f"{'✅' if value else '❌'}")
        
        # Node-level similarities tab
        with tab3:
            node_sim = compare_node_similarity(G1, G2)
            for metric, value in node_sim.items():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{metric}**")
                        st.caption(explanations[metric])
                    with col2:
                        if isinstance(value, bool):
                            st.write(f"{'✅' if value else '❌'}")
                        else:
                            st.metric(
                                label=metric,
                                value=f"{value:.4f}",
                                label_visibility="collapsed"
                            )

if __name__ == "__main__":
    main() 