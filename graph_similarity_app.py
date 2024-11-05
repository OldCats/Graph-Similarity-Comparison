import streamlit as st
import networkx as nx
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.linalg import eigvals
from node2vec import Node2Vec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

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
    """Calculate similarities based on adjacency matrices with detailed steps"""
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
    
    # Jaccard similarity with steps
    intersection = np.sum(np.logical_and(A1_flat, A2_flat))
    union = np.sum(np.logical_or(A1_flat, A2_flat))
    jaccard = intersection / union if union != 0 else 0
    jaccard_steps = {
        'intersection': int(intersection),
        'union': int(union),
        'value': jaccard
    }
    
    # Cosine similarity with steps
    dot_product = np.dot(A1_flat, A2_flat)
    norm1 = np.linalg.norm(A1_flat)
    norm2 = np.linalg.norm(A2_flat)
    cosine_sim = 1 - cosine(A1_flat, A2_flat) if np.any(A1_flat) and np.any(A2_flat) else 0
    cosine_steps = {
        'dot_product': float(dot_product),
        'norm1': float(norm1),
        'norm2': float(norm2),
        'value': cosine_sim
    }
    
    # Spectral similarity with steps
    k = min(len(G1), len(G2))
    eig1 = sorted(np.real(eigvals(A1_padded)))[-k:]
    eig2 = sorted(np.real(eigvals(A2_padded)))[-k:]
    spectral_diff = np.linalg.norm(np.array(eig1) - np.array(eig2))
    spectral_sim = 1 / (1 + spectral_diff)
    spectral_steps = {
        'eigenvalues1': eig1,
        'eigenvalues2': eig2,
        'spectral_diff': float(spectral_diff),
        'value': spectral_sim
    }
    
    # Graph Edit Distance similarity with steps
    try:
        ged = nx.graph_edit_distance(G1, G2)
        max_possible_ged = max(len(G1) + len(G2.edges()), len(G2) + len(G1.edges()))
        ged_sim = 1 - (ged / max_possible_ged if max_possible_ged > 0 else 0)
        ged_steps = {
            'ged': ged,
            'max_possible_ged': max_possible_ged,
            'value': ged_sim
        }
    except:
        ged_sim = 0
        ged_steps = {
            'error': 'GED calculation failed',
            'value': 0
        }
    
    return {
        'Jaccard Similarity': {'value': jaccard, 'steps': jaccard_steps},
        'Cosine Similarity': {'value': cosine_sim, 'steps': cosine_steps},
        'Spectral Similarity': {'value': spectral_sim, 'steps': spectral_steps},
        'Graph Edit Distance Similarity': {'value': ged_sim, 'steps': ged_steps}
    }

def get_graph_embedding(G, dimensions=8, walk_length=30, num_walks=200):
    """Generate graph embedding using node2vec"""
    # Initialize node2vec model
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=1  # Single worker for reproducibility
    )
    
    # Train the model
    model = node2vec.fit(window=10, min_count=1)
    
    # Get embeddings for all nodes
    node_embeddings = {}
    for node in G.nodes():
        node_embeddings[node] = model.wv[str(node)]
    
    # Calculate graph embedding as mean of node embeddings
    graph_embedding = np.mean(list(node_embeddings.values()), axis=0)
    
    return graph_embedding, node_embeddings

def compare_embeddings(G1, G2):
    """Compare graphs using embeddings"""
    # Generate embeddings
    graph1_emb, nodes1_emb = get_graph_embedding(G1)
    graph2_emb, nodes2_emb = get_graph_embedding(G2)
    
    # Calculate graph-level similarity
    graph_similarity = cosine_similarity(
        graph1_emb.reshape(1, -1),
        graph2_emb.reshape(1, -1)
    )[0][0]
    
    # Calculate average node embedding similarity
    node_similarities = []
    for node1, emb1 in nodes1_emb.items():
        for node2, emb2 in nodes2_emb.items():
            sim = cosine_similarity(
                emb1.reshape(1, -1),
                emb2.reshape(1, -1)
            )[0][0]
            node_similarities.append(sim)
    avg_node_similarity = np.mean(node_similarities)
    
    return {
        'Graph Embedding Similarity': {
            'value': graph_similarity,
            'steps': {
                'graph1_embedding': graph1_emb,
                'graph2_embedding': graph2_emb,
                'similarity': graph_similarity
            }
        },
        'Average Node Embedding Similarity': {
            'value': avg_node_similarity,
            'steps': {
                'individual_similarities': node_similarities,
                'average': avg_node_similarity
            }
        }
    }

def get_metric_explanations():
    """Return explanations and formulas for each metric"""
    base_explanations = {
        # Matrix Similarity Metrics
        'Jaccard Similarity': {
            'description': 'Measures similarity as ratio of common edges to total edges (1.0 = identical graphs)',
            'formula': r"""
            J(A_1, A_2) = \frac{|A_1 \cap A_2|}{|A_1 \cup A_2|}
            """
        },
        'Cosine Similarity': {
            'description': 'Measures similarity as cosine of angle between adjacency matrices (1.0 = identical graphs)',
            'formula': r"""
            \cos(A_1, A_2) = \frac{A_1 \cdot A_2}{||A_1|| \cdot ||A_2||}
            """
        },
        'Spectral Similarity': {
            'description': 'Compares graph structure using eigenvalues of adjacency matrices (1.0 = identical graphs)',
            'formula': r"""
            S(A_1, A_2) = \frac{1}{1 + ||\lambda_1 - \lambda_2||}
            """
        },
        'Graph Edit Distance Similarity': {
            'description': 'Measures similarity based on minimum operations to transform one graph to another (1.0 = identical graphs)',
            'formula': r"""
            GED_{sim} = 1 - \frac{GED}{\max(|V_1| + |E_2|, |V_2| + |E_1|)}
            """
        },
        
        # Node-level Metrics
        'Clustering Coefficient Difference': {
            'description': 'Measures the difference in how much nodes tend to cluster together',
            'formula': r"""
            |C_1 - C_2|, \text{ where } C_i = \frac{3 \times \text{triangles}}{\text{total possible triangles}}
            """
        },
        'Average Betweenness Difference': {
            'description': 'Compares the average betweenness centrality, which measures how often a node acts as a bridge',
            'formula': r"""
            |\text{avg}(B_1) - \text{avg}(B_2)|, \text{ where }
            B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}
            """
        },
        'Average Closeness Difference': {
            'description': 'Compares the average closeness centrality, measuring how close each node is to all other nodes',
            'formula': r"""
            |\text{avg}(C_1) - \text{avg}(C_2)|, \text{ where }
            C(v) = \frac{n-1}{\sum_{u \neq v} d(v,u)}
            """
        },

        # Structural Metrics (keeping original descriptions for boolean metrics)
        'Number of Nodes Match': 'Checks if both graphs have the same number of vertices/nodes.',
        'Number of Edges Match': 'Checks if both graphs have the same number of edges/connections.',
        'Degree Sequence Match': 'Compares the sorted sequence of node degrees (number of connections per node) between graphs.',
        'Density Match': 'Compares the density (ratio of actual edges to possible edges) between graphs.',
        'Connected Components Match': 'Checks if both graphs have the same number of disconnected parts.',
        'Both Graphs Connected': 'Indicates whether both graphs are fully connected (no isolated parts).',
        'Average Path Length Match': 'Compares the average shortest path length (only for connected graphs).',
        'Diameter Match': 'Compares the maximum shortest path length (only for connected graphs).',
        'Degree Distribution Match': 'Compares the frequency distribution of node degrees between graphs.',
        
        # Add embedding metrics
        'Graph Embedding Similarity': {
            'description': 'Measures similarity between graph-level embeddings using node2vec',
            'formula': r"""
            sim(G_1, G_2) = \cos(\text{mean}(N_1), \text{mean}(N_2))
            \text{ where } N_i \text{ are node embeddings}
            """
        },
        'Average Node Embedding Similarity': {
            'description': 'Average similarity between all pairs of node embeddings',
            'formula': r"""
            sim = \frac{1}{|V_1||V_2|}\sum_{i \in V_1}\sum_{j \in V_2} \cos(emb(i), emb(j))
            """
        }
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
        tab1, tab2, tab3, tab4 = st.tabs([
            "Matrix-based Similarities",
            "Structural Similarities",
            "Node-level Similarities",
            "Embedding-based Similarities"
        ])
        
        # Matrix-based similarities tab
        with tab1:
            matrix_sim = calculate_matrix_similarities(G1, G2)
            for metric, data in matrix_sim.items():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{metric}**")
                        explanation = explanations[metric]
                        if isinstance(explanation, dict):
                            st.caption(explanation['description'])
                            with st.expander("Show Formula and Calculation"):
                                # Show the formula
                                st.latex(explanation['formula'])
                                
                                # Show calculation steps
                                st.write("**Calculation Steps:**")
                                steps = data['steps']
                                
                                if metric == 'Jaccard Similarity':
                                    st.latex(f"""
                                    J = \\frac{{{steps['intersection']}}}{{{steps['union']}}} = {steps['value']:.4f}
                                    """)
                                    
                                elif metric == 'Cosine Similarity':
                                    st.latex(f"""
                                    \\cos = \\frac{{{steps['dot_product']:.2f}}}
                                    {{{steps['norm1']:.2f} \\cdot {steps['norm2']:.2f}}} = {steps['value']:.4f}
                                    """)
                                    
                                elif metric == 'Spectral Similarity':
                                    st.write(f"Eigenvalues Graph 1: {[f'{x:.2f}' for x in steps['eigenvalues1']]}")
                                    st.write(f"Eigenvalues Graph 2: {[f'{x:.2f}' for x in steps['eigenvalues2']]}")
                                    st.latex(f"""
                                    S = \\frac{{1}}{{1 + {steps['spectral_diff']:.4f}}} = {steps['value']:.4f}
                                    """)
                                    
                                elif metric == 'Graph Edit Distance Similarity':
                                    if 'error' not in steps:
                                        st.latex(f"""
                                        GED_{{sim}} = 1 - \\frac{{{steps['ged']}}}{{{steps['max_possible_ged']}}} = {steps['value']:.4f}
                                        """)
                                    else:
                                        st.write(steps['error'])
                        else:
                            st.caption(explanation)
                    with col2:
                        st.metric(
                            label=metric,
                            value=f"{data['value']:.4f}",
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
                        explanation = explanations[metric]
                        if isinstance(explanation, dict):
                            st.caption(explanation['description'])
                            with st.expander("Show Formula"):
                                st.latex(explanation['formula'])
                        else:
                            st.caption(explanation)
                    with col2:
                        if isinstance(value, bool):
                            st.write(f"{'✅' if value else '❌'}")
                        else:
                            st.metric(
                                label=metric,
                                value=f"{value:.4f}",
                                label_visibility="collapsed"
                            )
        
        # Embedding-based similarities tab
        with tab4:
            try:
                embedding_sim = compare_embeddings(G1, G2)
                for metric, data in embedding_sim.items():
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{metric}**")
                            explanation = explanations[metric]
                            if isinstance(explanation, dict):
                                st.caption(explanation['description'])
                                with st.expander("Show Formula and Details"):
                                    st.latex(explanation['formula'])
                                    st.write("**Calculation Details:**")
                                    steps = data['steps']
                                    
                                    if metric == 'Graph Embedding Similarity':
                                        st.write("Graph 1 embedding (first 3 dimensions):")
                                        st.write(steps['graph1_embedding'][:3])
                                        st.write("Graph 2 embedding (first 3 dimensions):")
                                        st.write(steps['graph2_embedding'][:3])
                                        st.write(f"Cosine similarity: {steps['similarity']:.4f}")
                                    
                                    elif metric == 'Average Node Embedding Similarity':
                                        st.write("Distribution of node-pair similarities:")
                                        fig, ax = plt.subplots()
                                        ax.hist(steps['individual_similarities'], bins=20)
                                        ax.set_xlabel("Similarity")
                                        ax.set_ylabel("Frequency")
                                        st.pyplot(fig)
                                        st.write(f"Average similarity: {steps['average']:.4f}")
                        
                        with col2:
                            st.metric(
                                label=metric,
                                value=f"{data['value']:.4f}",
                                label_visibility="collapsed"
                            )
            except Exception as e:
                st.error(f"Error calculating embedding similarities: {str(e)}")
                st.info("Note: Embedding calculation requires connected graphs with sufficient nodes.")

if __name__ == "__main__":
    main() 