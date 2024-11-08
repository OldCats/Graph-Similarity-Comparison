import streamlit as st
import networkx as nx
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.linalg import eigvals
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import rdflib
from rdflib import Graph as RDFGraph
from io import StringIO

def create_sample_graphs(directed=False):
    """Create sample graphs with option for directed graphs"""
    # Use DiGraph for directed graphs, regular Graph for undirected
    GraphType = nx.DiGraph if directed else nx.Graph
    
    # First graph
    G1 = GraphType()
    G1.add_edges_from([
        (1, 2), (2, 3), (3, 4), (4, 1), (2, 4)
    ])
    
    # Second graph - similar structure but different node labels
    G2 = GraphType()
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

def parse_triple_input(triple_text):
    """Parse triple (TTL) input text into a list of edges"""
    edges = []
    try:
        # Split the input into lines and process each line
        lines = triple_text.strip().split('\n')
        for line in lines:
            if line.strip():
                # Split on whitespace and extract subject and object
                parts = line.strip().split()
                if len(parts) >= 3:  # Must have subject, predicate, object
                    subject = parts[0].strip('<>')  # Remove < > if present
                    object_ = parts[2].strip('<>')  # Remove < > if present
                    # Convert URIs/strings to integers for graph processing
                    try:
                        # Try to extract numeric part from URIs like "node1" or "http://example.org/node1"
                        subject_num = int(''.join(filter(str.isdigit, subject)))
                        object_num = int(''.join(filter(str.isdigit, object_)))
                        edges.append((subject_num, object_num))
                    except ValueError:
                        # If no numbers found, use hash of string modulo 1000 as node ID
                        subject_num = hash(subject) % 1000
                        object_num = hash(object_) % 1000
                        edges.append((subject_num, object_num))
        return edges
    except ValueError as e:
        st.error(f"Invalid input format: {str(e)}\nPlease use TTL format: '<subject> <predicate> <object>'")
        return None

def parse_rdf_input(rdf_text, format='turtle'):
    """Parse RDF input text into a list of edges"""
    edges = []
    try:
        # Create RDF graph
        g = RDFGraph()
        
        # Parse RDF content
        g.parse(StringIO(rdf_text), format=format)
        
        # Extract edges from triples
        for s, p, o in g:
            # Convert URIs/literals to strings and extract numeric parts or hash
            try:
                # Try to extract numeric part from URIs
                subject = str(s).split('/')[-1].strip('<>')
                object_ = str(o).split('/')[-1].strip('<>')
                
                # Try to get numeric values
                subject_num = int(''.join(filter(str.isdigit, subject)))
                object_num = int(''.join(filter(str.isdigit, object_)))
            except ValueError:
                # If no numbers found, use hash of string modulo 1000
                subject_num = hash(str(s)) % 1000
                object_num = hash(str(o)) % 1000
            
            edges.append((subject_num, object_num))
        
        return edges
    except Exception as e:
        st.error(f"Error parsing RDF: {str(e)}")
        return None

def compare_structural_similarity(G1, G2):
    """Compare graphs based on structural properties with actual values, supporting directed graphs"""
    
    # Check if graphs are connected (use appropriate function for directed graphs)
    is_G1_connected = (nx.is_weakly_connected(G1) if isinstance(G1, nx.DiGraph) 
                      else nx.is_connected(G1))
    is_G2_connected = (nx.is_weakly_connected(G2) if isinstance(G2, nx.DiGraph)
                      else nx.is_connected(G2))
    
    # Get connected components count
    G1_components = (nx.number_weakly_connected_components(G1) if isinstance(G1, nx.DiGraph)
                    else nx.number_connected_components(G1))
    G2_components = (nx.number_weakly_connected_components(G2) if isinstance(G2, nx.DiGraph)
                    else nx.number_connected_components(G2))
    
    # Calculate average path length and diameter only if graphs are connected
    try:
        avg_path_G1 = nx.average_shortest_path_length(G1) if is_G1_connected else None
        avg_path_G2 = nx.average_shortest_path_length(G2) if is_G2_connected else None
        diameter_G1 = nx.diameter(G1) if is_G1_connected else None
        diameter_G2 = nx.diameter(G2) if is_G2_connected else None
    except:
        avg_path_G1 = avg_path_G2 = diameter_G1 = diameter_G2 = None
    
    # Get actual values for comparison
    nodes_G1 = len(G1)
    nodes_G2 = len(G2)
    edges_G1 = len(G1.edges())
    edges_G2 = len(G2.edges())
    degree_seq_G1 = sorted([d for n, d in G1.degree()])
    degree_seq_G2 = sorted([d for n, d in G2.degree()])
    density_G1 = nx.density(G1)
    density_G2 = nx.density(G2)
    
    similarity_metrics = {
        'Number of Nodes Match': {
            'match': nodes_G1 == nodes_G2,
            'values': (nodes_G1, nodes_G2)
        },
        'Number of Edges Match': {
            'match': edges_G1 == edges_G2,
            'values': (edges_G1, edges_G2)
        },
        'Degree Sequence Match': {
            'match': degree_seq_G1 == degree_seq_G2,
            'values': (degree_seq_G1, degree_seq_G2)
        },
        'Density Match': {
            'match': abs(density_G1 - density_G2) < 1e-9,
            'values': (f"{density_G1:.4f}", f"{density_G2:.4f}")
        },
        'Connected Components Match': {
            'match': G1_components == G2_components,
            'values': (G1_components, G2_components)
        },
        'Both Graphs Connected': {
            'match': is_G1_connected and is_G2_connected,
            'values': (is_G1_connected, is_G2_connected)
        }
    }
    
    # Add path-based metrics only if both graphs are connected and values are not None
    if is_G1_connected and is_G2_connected and avg_path_G1 is not None and avg_path_G2 is not None:
        similarity_metrics.update({
            'Average Path Length Match': {
                'match': abs(avg_path_G1 - avg_path_G2) < 1e-9,
                'values': (f"{avg_path_G1:.4f}", f"{avg_path_G2:.4f}")
            }
        })
        
        if diameter_G1 is not None and diameter_G2 is not None:
            similarity_metrics.update({
                'Diameter Match': {
                    'match': diameter_G1 == diameter_G2,
                    'values': (diameter_G1, diameter_G2)
                }
            })
    
    return similarity_metrics

def safe_centrality_calculation(G, centrality_func):
    """Calculate centrality with support for both directed and undirected graphs"""
    try:
        if isinstance(G, nx.DiGraph):
            # For directed graphs, try to use specific directed versions if available
            if centrality_func == nx.closeness_centrality:
                return nx.closeness_centrality(G)
            elif centrality_func == nx.betweenness_centrality:
                return nx.betweenness_centrality(G)
            else:
                return {}  # Return empty dict for unsupported metrics
        else:
            # Original logic for undirected graphs
            if nx.is_connected(G):
                return centrality_func(G)
            else:
                centrality_dict = {}
                for component in nx.connected_components(G):
                    subgraph = G.subgraph(component)
                    centrality_dict.update(centrality_func(subgraph))
                return centrality_dict
    except nx.NetworkXNotImplemented:
        return {}  # Return empty dict for unsupported metrics
    except Exception as e:
        st.warning(f"Error calculating centrality: {str(e)}")
        return {}

def compare_node_similarity(G1, G2):
    """Compare node-level properties between graphs"""
    metrics = {}
    
    try:
        # Degree distribution (works for both directed and undirected)
        deg_dist1 = Counter([d for n, d in G1.degree()])
        deg_dist2 = Counter([d for n, d in G2.degree()])
        metrics['Degree Distribution Match'] = deg_dist1 == deg_dist2
    except Exception as e:
        st.warning(f"Error calculating degree distribution: {str(e)}")
        metrics['Degree Distribution Match'] = False
    
    try:
        # Clustering coefficient (show unsupported for directed graphs)
        if isinstance(G1, nx.DiGraph) or isinstance(G2, nx.DiGraph):
            metrics['Clustering Coefficient Difference'] = "Unsupported for directed graphs"
        else:
            clustering1 = nx.average_clustering(G1)
            clustering2 = nx.average_clustering(G2)
            metrics['Clustering Coefficient Difference'] = abs(clustering1 - clustering2)
    except Exception as e:
        st.warning(f"Error calculating clustering coefficient: {str(e)}")
        metrics['Clustering Coefficient Difference'] = "Error in calculation"
    
    # Calculate centrality measures safely
    try:
        if isinstance(G1, nx.DiGraph) or isinstance(G2, nx.DiGraph):
            metrics['Average Betweenness Difference'] = "Unsupported for directed graphs"
            metrics['Average Closeness Difference'] = "Unsupported for directed graphs"
        else:
            betweenness1 = safe_centrality_calculation(G1, nx.betweenness_centrality)
            betweenness2 = safe_centrality_calculation(G2, nx.betweenness_centrality)
            closeness1 = safe_centrality_calculation(G1, nx.closeness_centrality)
            closeness2 = safe_centrality_calculation(G2, nx.closeness_centrality)
            
            if betweenness1 and betweenness2:
                metrics['Average Betweenness Difference'] = abs(
                    sum(betweenness1.values())/len(betweenness1) - 
                    sum(betweenness2.values())/len(betweenness2)
                )
            else:
                metrics['Average Betweenness Difference'] = "Unsupported for this graph type"
                
            if closeness1 and closeness2:
                metrics['Average Closeness Difference'] = abs(
                    sum(closeness1.values())/len(closeness1) - 
                    sum(closeness2.values())/len(closeness2)
                )
            else:
                metrics['Average Closeness Difference'] = "Unsupported for this graph type"
    except Exception as e:
        st.warning(f"Error calculating centrality metrics: {str(e)}")
        metrics['Average Betweenness Difference'] = "Error in calculation"
        metrics['Average Closeness Difference'] = "Error in calculation"
    
    return metrics

def plot_graphs(G1, G2):
    """Create a matplotlib figure with both graphs, supporting directed graphs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot first graph with spring layout
    pos1 = nx.spring_layout(G1, k=1)
    nx.draw(G1, pos1, ax=ax1, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=16,
            arrows=isinstance(G1, nx.DiGraph),  # Show arrows if directed
            arrowsize=20)
    ax1.set_title(f"Graph 1 ({nx.number_weakly_connected_components(G1) if isinstance(G1, nx.DiGraph) else nx.number_connected_components(G1)} components)")
    
    # Plot second graph with spring layout
    pos2 = nx.spring_layout(G2, k=1)
    nx.draw(G2, pos2, ax=ax2, with_labels=True, node_color='lightgreen', 
            node_size=500, font_size=16,
            arrows=isinstance(G2, nx.DiGraph),  # Show arrows if directed
            arrowsize=20)
    ax2.set_title(f"Graph 2 ({nx.number_weakly_connected_components(G2) if isinstance(G2, nx.DiGraph) else nx.number_connected_components(G2)} components)")
    
    plt.tight_layout()
    return fig

def calculate_matrix_similarities(G1, G2):
    """Calculate similarities based on adjacency matrices with detailed steps"""
    try:
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
            'Jaccard Similarity': {
                'value': jaccard,
                'steps': jaccard_steps
            },
            'Cosine Similarity': {
                'value': cosine_sim,
                'steps': cosine_steps
            },
            'Spectral Similarity': {
                'value': spectral_sim,
                'steps': spectral_steps
            },
            'Graph Edit Distance Similarity': {
                'value': ged_sim,
                'steps': ged_steps
            }
        }
    except Exception as e:
        st.error(f"Error calculating matrix similarities: {str(e)}")
        return {}  # Return empty dict instead of None

def get_graph_embedding(G, dimensions=8):
    """使用改進譜嵌入方法生成圖嵌入"""
    # 檢查圖的大小
    n_nodes = len(G)
    
    # 調整維度，確保所圖使用相同維度
    actual_dimensions = min(8, max(1, n_nodes - 1))
    
    try:
        # 計算拉普拉斯矩陣的特徵向量作為節點嵌入
        L = nx.normalized_laplacian_matrix(G).todense()
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # 使用最小的非零特徵值對應的特徵向量
        # 排除第一個特徵值（總是0）
        idx = np.argsort(eigenvalues)[1:actual_dimensions+1]
        embeddings = np.real(eigenvectors[:, idx])
        
        # 安全的標準化嵌入
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # 避免除以零，如果範數為零，則���1替代
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        # 檢查並處理任何可能的 NaN 值
        embeddings = np.nan_to_num(embeddings, nan=0.0)
        
        # 創建節點嵌入字典
        node_embeddings = {node: embeddings[i] for i, node in enumerate(G.nodes())}
        
        # 計算圖嵌入為節點嵌入的加權平均
        # 使用度作為權重，確保權重和為1
        degrees = np.array([G.degree(node) for node in G.nodes()])
        weights = degrees / (degrees.sum() + 1e-10)  # 添加小量以避免除以零
        
        # 使用安全的加權平均
        graph_embedding = np.average(embeddings, axis=0, weights=weights)
        
        # 最後檢查確保沒有 NaN 值
        if np.any(np.isnan(graph_embedding)):
            raise ValueError("Graph embedding contains NaN values")
            
        return graph_embedding, node_embeddings
        
    except Exception as e:
        st.warning(f"使用基本嵌入方法。原因：{str(e)}")
        # 使用更穩健的後備方法
        try:
            # 使用基本的圖特徵
            features = []
            
            # 1. 度中心性
            degree_cent = nx.degree_centrality(G)
            features.append(list(degree_cent.values()))
            
            # 2. 聚類係數
            clustering = nx.clustering(G)
            features.append(list(clustering.values()))
            
            # 3. 節點連接數
            degrees = [G.degree(node) for node in G.nodes()]
            max_degree = max(degrees) if degrees else 1
            normalized_degrees = [d/max_degree for d in degrees]
            features.append(normalized_degrees)
            
            # 組合特徵
            embeddings = np.array(features).T  # 轉置使每行代表一個節點
            
            # 如果要更多維度，用零填充
            if embeddings.shape[1] < actual_dimensions:
                padding = np.zeros((embeddings.shape[0], actual_dimensions - embeddings.shape[1]))
                embeddings = np.hstack([embeddings, padding])
            
            # 標準化
            embeddings = (embeddings - embeddings.mean(axis=0)) / (embeddings.std(axis=0) + 1e-10)
            
            # 創建節點嵌入字典
            node_embeddings = {node: embeddings[i] for i, node in enumerate(G.nodes())}
            
            # 計算圖嵌入
            graph_embedding = np.mean(embeddings, axis=0)
            
            return graph_embedding, node_embeddings
            
        except Exception as e:
            # 如果所有方法都失敗，返��隨機嵌入
            st.warning(f"使用隨機嵌入。原因：{str(e)}")
            random_state = np.random.RandomState(42)
            embeddings = random_state.rand(n_nodes, actual_dimensions)
            node_embeddings = {node: embeddings[i] for i, node in enumerate(G.nodes())}
            graph_embedding = np.mean(embeddings, axis=0)
            return graph_embedding, node_embeddings

def compare_embeddings(G1, G2):
    """Compare graphs using embeddings with improved isomorphic handling"""
    try:
        # Generate embeddings with fixed dimensions
        graph1_emb, nodes1_emb = get_graph_embedding(G1)
        graph2_emb, nodes2_emb = get_graph_embedding(G2)
        
        # Ensure both embeddings have the same dimensions
        min_dim = min(len(graph1_emb), len(graph2_emb))
        graph1_emb = graph1_emb[:min_dim]
        graph2_emb = graph2_emb[:min_dim]
        
        # Calculate graph-level similarity
        graph_similarity = max(
            abs(cosine_similarity(graph1_emb.reshape(1, -1), graph2_emb.reshape(1, -1))[0][0]),
            abs(cosine_similarity(graph1_emb.reshape(1, -1), -graph2_emb.reshape(1, -1))[0][0])
        )
        
        # Calculate node similarities
        node_similarities = []
        for emb1 in nodes1_emb.values():
            best_sim = -1
            for emb2 in nodes2_emb.values():
                sim = max(
                    abs(cosine_similarity(emb1[:min_dim].reshape(1, -1), 
                                       emb2[:min_dim].reshape(1, -1))[0][0]),
                    abs(cosine_similarity(emb1[:min_dim].reshape(1, -1), 
                                       -emb2[:min_dim].reshape(1, -1))[0][0])
                )
                best_sim = max(best_sim, sim)
            node_similarities.append(best_sim)
        
        avg_node_similarity = np.mean(node_similarities) if node_similarities else 0.0
        
        return {
            'Graph Embedding Similarity': {
                'value': float(graph_similarity),
                'steps': {
                    'graph1_embedding': graph1_emb,
                    'graph2_embedding': graph2_emb,
                    'similarity': float(graph_similarity)
                }
            },
            'Average Node Embedding Similarity': {
                'value': float(avg_node_similarity),
                'steps': {
                    'individual_similarities': node_similarities,
                    'average': float(avg_node_similarity),
                    'num_comparisons': len(node_similarities)
                }
            }
        }
    except Exception as e:
        st.error(f"Embedding calculation error: {str(e)}")
        return {}

def get_metric_explanations():
    """Return explanations and formulas for each metric"""
    base_explanations = {
        # Matrix-based Metrics
        'Jaccard Similarity': {
            'description': 'Measures similarity as ratio of common edges to total edges',
            'formula': r"""
            J(A_1, A_2) = \frac{|A_1 \cap A_2|}{|A_1 \cup A_2|}
            """,
            'interpretation': """
            - 1.0 indicates identical edge sets
            - 0.0 indicates no common edges
            - Works for both directed and undirected graphs
            """
        },
        'Cosine Similarity': {
            'description': 'Measures similarity as cosine of angle between adjacency matrices',
            'formula': r"""
            \cos(A_1, A_2) = \frac{A_1 \cdot A_2}{||A_1|| \cdot ||A_2||}
            """,
            'interpretation': """
            - 1.0 indicates identical structural patterns
            - 0.0 indicates completely different patterns
            - Works for both directed and undirected graphs
            """
        },
        'Spectral Similarity': {
            'description': 'Compares graph structure using eigenvalues of adjacency matrices',
            'formula': r"""
            S(A_1, A_2) = \frac{1}{1 + ||\lambda_1 - \lambda_2||}
            """,
            'interpretation': """
            - Higher values indicate similar spectral properties
            - Captures global structural similarity
            - Works for both directed and undirected graphs
            """
        },
        'Graph Edit Distance Similarity': {
            'description': 'Measures similarity based on minimum operations to transform one graph to another',
            'formula': r"""
            GED_{sim} = 1 - \frac{GED}{\max(|V_1| + |E_2|, |V_2| + |E_1|)}
            """,
            'interpretation': """
            - 1.0 indicates identical graphs
            - Lower values indicate more operations needed
            - Works for both directed and undirected graphs
            """
        },

        # Node-level Metrics
        'Degree Distribution Match': {
            'description': 'Compares the frequency distribution of node degrees between graphs',
            'formula': r"""
            \text{deg}(v) = \text{number of edges connected to node } v \\
            \text{distribution} = \text{frequency count of each degree value}
            """,
            'interpretation': """
            - Matches if both graphs have same frequency of node degrees
            - Example: if both graphs have same number of nodes with degree 1, 2, etc.
            """
        },
        'Clustering Coefficient Difference': {
            'description': 'Measures how much nodes tend to cluster together (form triangles)',
            'formula': r"""
            C(v) = \frac{2L_v}{k_v(k_v-1)} \\
            \text{where } L_v \text{ is number of edges between neighbors of } v \\
            k_v \text{ is degree of node } v \\
            \text{Graph clustering} = \frac{1}{n}\sum_{v \in V} C(v)
            """,
            'interpretation': """
            - Values range from 0 (no clustering) to 1 (complete clustering)
            - Smaller difference indicates more similar clustering patterns
            - Unsupported for directed graphs
            """
        },
        'Average Betweenness Difference': {
            'description': 'Compares how often nodes act as bridges in shortest paths',
            'formula': r"""
            B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}} \\
            \text{where } \sigma_{st} \text{ is total number of shortest paths from } s \text{ to } t \\
            \sigma_{st}(v) \text{ is number of those paths passing through } v
            """,
            'interpretation': """
            - Higher values indicate more important bridging nodes
            - Smaller difference means similar network centrality patterns
            - Unsupported for directed graphs
            """
        },
        'Average Closeness Difference': {
            'description': 'Compares how close nodes are to all other nodes',
            'formula': r"""
            C(v) = \frac{n-1}{\sum_{u \neq v} d(v,u)} \\
            \text{where } d(v,u) \text{ is shortest path length between } v \text{ and } u \\
            n \text{ is number of nodes}
            """,
            'interpretation': """
            - Higher values indicate more central nodes
            - Smaller difference means similar centrality patterns
            - Unsupported for directed graphs
            """
        },

        # Structural Metrics with formulas
        'Number of Nodes Match': {
            'description': 'Checks if both graphs have the same number of vertices/nodes',
            'formula': r"""
            |V_1| = |V_2|
            """
        },
        'Number of Edges Match': {
            'description': 'Checks if both graphs have the same number of edges/connections',
            'formula': r"""
            |E_1| = |E_2|
            """
        },
        'Density Match': {
            'description': 'Compares the density (ratio of actual edges to possible edges)',
            'formula': r"""
            D(G) = \frac{2|E|}{|V|(|V|-1)} \text{ for undirected} \\
            D(G) = \frac{|E|}{|V|(|V|-1)} \text{ for directed}
            """
        },
        'Average Path Length Match': {
            'description': 'Compares the average shortest path length',
            'formula': r"""
            L(G) = \frac{1}{n(n-1)} \sum_{i \neq j} d(v_i, v_j)
            """
        },
        'Diameter Match': {
            'description': 'Compares the maximum shortest path length',
            'formula': r"""
            diam(G) = \max_{i,j} d(v_i, v_j)
            """
        },
        
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
        },

        # Add Degree Sequence Match explanation
        'Degree Sequence Match': {
            'description': 'Compares the sorted sequence of node degrees between graphs',
            'formula': r"""
            deg\_seq(G) = \text{sorted}([d_1, d_2, ..., d_n]) \\
            \text{where } d_i \text{ is the degree of node } i
            """,
            'interpretation': """
            - Matches if both graphs have identical degree sequences when sorted
            - Indicates similar connectivity patterns between graphs
            - Works for both directed and undirected graphs
            """
        },

        'Connected Components Match': {
            'description': 'Compares the number of connected components in both graphs',
            'formula': r"""
            \text{For undirected: } CC(G) = |\{\text{connected components}\}| \\
            \text{For directed: } WCC(G) = |\{\text{weakly connected components}\}|
            """,
            'interpretation': """
            - Matches if both graphs have same number of components
            - Uses weak connectivity for directed graphs
            - Indicates similar graph fragmentation
            """
        },

        'Both Graphs Connected': {
            'description': 'Checks if both graphs are fully connected',
            'formula': r"""
            \text{For undirected: } \text{is\_connected}(G) \\
            \text{For directed: } \text{is\_weakly\_connected}(G)
            """,
            'interpretation': """
            - True if both graphs are connected
            - Uses weak connectivity for directed graphs
            - Required for certain path-based metrics
            """
        },
    }
    return base_explanations

def display_metric_with_details(metric_name, value, explanation, actual_values=None):
    """Unified display function for all metrics with details and actual values"""
    with st.container():
        # Header row with metric name and value only
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{metric_name}**")
            if isinstance(explanation, dict):
                st.caption(explanation['description'])
            else:
                st.caption(explanation)
        with col2:
            if isinstance(value, str) and ("Unsupported" in value or "Error" in value):
                st.info(value) if "Unsupported" in value else st.error(value)
            elif isinstance(value, bool):
                st.write(f"{'✅' if value else '❌'}")
            else:
                st.metric(
                    label=metric_name,
                    value=f"{value:.4f}" if isinstance(value, (float, int)) else value,
                    label_visibility="collapsed"
                )
        
        # All details in expander
        with st.expander("Show Details"):
            if isinstance(explanation, dict):
                # Always show formula first
                st.write("**Formula:**")
                st.latex(explanation['formula'])
                
                # Show actual values and calculation for all metrics
                if actual_values:
                    st.write("**Calculation with actual values:**")
                    if isinstance(actual_values, tuple):
                        # For basic structural metrics
                        st.latex(f"""
                        G_1: {actual_values[0]} \\\\
                        G_2: {actual_values[1]}
                        """)
                    elif isinstance(actual_values, dict):
                        if metric_name == 'Jaccard Similarity':
                            st.latex(f"""
                            J = \\frac{{{actual_values['intersection']}}}{{{actual_values['union']}}} = {actual_values['value']:.4f}
                            """)
                        elif metric_name == 'Cosine Similarity':
                            st.latex(f"""
                            \\cos = \\frac{{{actual_values['dot_product']:.2f}}}
                            {{{actual_values['norm1']:.2f} \\cdot {actual_values['norm2']:.2f}}} = {actual_values['value']:.4f}
                            """)
                        elif metric_name == 'Spectral Similarity':
                            st.latex(f"""
                            \\lambda_1 = [{', '.join([f'{x:.2f}' for x in actual_values['eigenvalues1'][:3]])}...] \\\\
                            \\lambda_2 = [{', '.join([f'{x:.2f}' for x in actual_values['eigenvalues2'][:3]])}...] \\\\
                            \\text{{spectral diff}} = {actual_values['spectral_diff']:.4f} \\\\
                            S = \\frac{{1}}{{1 + {actual_values['spectral_diff']:.4f}}} = {actual_values['value']:.4f}
                            """)
                        elif metric_name == 'Graph Edit Distance Similarity':
                            if 'error' not in actual_values:
                                st.latex(f"""
                                GED = {actual_values['ged']} \\\\
                                \\text{{max possible GED}} = {actual_values['max_possible_ged']} \\\\
                                GED_{{\text{{sim}}}} = 1 - \\frac{{{actual_values['ged']}}}{{{actual_values['max_possible_ged']}}} = {actual_values['value']:.4f}
                                """)
                            else:
                                st.write(actual_values['error'])
                        elif metric_name == 'Degree Distribution Match':
                            st.latex(f"""
                            G_1: {actual_values['Graph 1 Distribution']} \\\\
                            G_2: {actual_values['Graph 2 Distribution']}
                            """)
                        elif metric_name in ['Clustering Coefficient Difference', 'Average Betweenness Difference', 'Average Closeness Difference']:
                            if isinstance(actual_values, str):
                                st.write(actual_values)
                            else:
                                # Check if we have the correct keys for the metric
                                g1_key = f"Graph 1 {metric_name.split(' ')[1]}"  # e.g., "Graph 1 Clustering"
                                g2_key = f"Graph 2 {metric_name.split(' ')[1]}"  # e.g., "Graph 2 Clustering"
                                
                                if g1_key in actual_values and g2_key in actual_values:
                                    st.latex(f"""
                                    G_1: {actual_values[g1_key]} \\\\
                                    G_2: {actual_values[g2_key]} \\\\
                                    |G_1 - G_2| = {value:.4f}
                                    """)
                                else:
                                    # Fallback for old format
                                    st.write(f"Values: {actual_values}")
                        elif metric_name == 'Graph Embedding Similarity':
                            if 'graph1_embedding' in actual_values:
                                st.latex(f"""
                                G_1: [{', '.join([f'{x:.3f}' for x in actual_values['graph1_embedding'][:3]])}...] \\\\
                                G_2: [{', '.join([f'{x:.3f}' for x in actual_values['graph2_embedding'][:3]])}...] \\\\
                                sim = {actual_values['similarity']:.4f}
                                """)
                        elif metric_name == 'Average Node Embedding Similarity':
                            if 'individual_similarities' in actual_values:
                                similarities = actual_values['individual_similarities']
                                st.latex(f"""
                                \\min: {min(similarities):.3f} \\\\
                                \\max: {max(similarities):.3f} \\\\
                                \\text{{avg}}: {actual_values['average']:.3f}
                                """)
                
                # Always show interpretation last if available
                if 'interpretation' in explanation:
                    st.write("**Interpretation:**")
                    st.write(explanation['interpretation'])

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
                1. Select graph type:
                   - Undirected: Edges have no direction
                   - Directed: Edges have direction (from source to target)
                
                2. Select your preferred input format:
                   - Edge List: Simple space-separated node pairs (e.g., "1 2")
                   - TTL (Triple): Semantic triple format
                   - RDF/XML: RDF in XML format
                   - N-Triples: RDF in N-Triples format
                
                3. For Edge List format:
                   - Enter one edge per line as "node1 node2" or "node1,node2"
                   - For directed graphs, edge goes from node1 to node2
                   - Use numeric node IDs
                
                4. For RDF formats (TTL/XML/N-Triples):
                   - Use appropriate format syntax
                   - Nodes can be URIs or simple strings
                   - The app will convert node identifiers to numbers
                   - Direction is preserved from subject to object
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
    
    # Add graph type selection
    graph_type = st.radio(
        "Select graph type",
        ["Undirected", "Directed"],
        horizontal=True
    )
    
    # Add format selection
    input_format = st.radio(
        "Select input format",
        ["Edge List", "TTL (Triple)", "RDF/XML", "N-Triples"],
        horizontal=True
    )
    
    format_map = {
        "TTL (Triple)": "turtle",
        "RDF/XML": "xml",
        "N-Triples": "nt",
        "Edge List": "edge_list"
    }
    
    # Graph input section
    st.markdown("---")
    st.subheader("📊 Graph Input")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Graph 1")
        if input_format == "Edge List":
            default_value = """1 2
2 3
3 4
4 1
2 4"""
        elif input_format == "TTL (Triple)":
            default_value = """<node1> <connects> <node2> .
<node2> <connects> <node3> .
<node3> <connects> <node4> .
<node4> <connects> <node1> .
<node2> <connects> <node4> ."""
        elif input_format == "RDF/XML":
            default_value = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:ex="http://example.org/">
  <rdf:Description rdf:about="http://example.org/node1">
    <ex:connects rdf:resource="http://example.org/node2"/>
  </rdf:Description>
  <rdf:Description rdf:about="http://example.org/node2">
    <ex:connects rdf:resource="http://example.org/node3"/>
    <ex:connects rdf:resource="http://example.org/node4"/>
  </rdf:Description>
  <rdf:Description rdf:about="http://example.org/node3">
    <ex:connects rdf:resource="http://example.org/node4"/>
  </rdf:Description>
  <rdf:Description rdf:about="http://example.org/node4">
    <ex:connects rdf:resource="http://example.org/node1"/>
  </rdf:Description>
</rdf:RDF>"""
        else:  # N-Triples
            default_value = """<http://example.org/node1> <http://example.org/connects> <http://example.org/node2> .
<http://example.org/node2> <http://example.org/connects> <http://example.org/node3> .
<http://example.org/node3> <http://example.org/connects> <http://example.org/node4> .
<http://example.org/node4> <http://example.org/connects> <http://example.org/node1> .
<http://example.org/node2> <http://example.org/connects> <http://example.org/node4> ."""
        
        graph1_input = st.text_area(
            f"Enter Graph 1 in {input_format} format",
            value=default_value,
            key="graph1",
            height=200
        )
    
    with col2:
        st.markdown("### Graph 2")
        if input_format == "Edge List":
            default_value = """5 6
6 7
7 8
8 5
6 8"""
        elif input_format == "TTL (Triple)":
            default_value = """<node5> <connects> <node6> .
<node6> <connects> <node7> .
<node7> <connects> <node8> .
<node8> <connects> <node5> .
<node6> <connects> <node8> ."""
        elif input_format == "RDF/XML":
            default_value = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:ex="http://example.org/">
  <rdf:Description rdf:about="http://example.org/node5">
    <ex:connects rdf:resource="http://example.org/node6"/>
  </rdf:Description>
  <rdf:Description rdf:about="http://example.org/node6">
    <ex:connects rdf:resource="http://example.org/node7"/>
    <ex:connects rdf:resource="http://example.org/node8"/>
  </rdf:Description>
  <rdf:Description rdf:about="http://example.org/node7">
    <ex:connects rdf:resource="http://example.org/node8"/>
  </rdf:Description>
  <rdf:Description rdf:about="http://example.org/node8">
    <ex:connects rdf:resource="http://example.org/node5"/>
  </rdf:Description>
</rdf:RDF>"""
        else:  # N-Triples
            default_value = """<http://example.org/node5> <http://example.org/connects> <http://example.org/node6> .
<http://example.org/node6> <http://example.org/connects> <http://example.org/node7> .
<http://example.org/node7> <http://example.org/connects> <http://example.org/node8> .
<http://example.org/node8> <http://example.org/connects> <http://example.org/node5> .
<http://example.org/node6> <http://example.org/connects> <http://example.org/node8> ."""
        
        graph2_input = st.text_area(
            f"Enter Graph 2 in {input_format} format",
            value=default_value,
            key="graph2",
            height=200
        )
    
    # Process graphs based on selected format
    if input_format == "Edge List":
        edges1 = parse_edge_input(graph1_input)
        edges2 = parse_edge_input(graph2_input)
    elif input_format == "TTL (Triple)":
        edges1 = parse_triple_input(graph1_input)
        edges2 = parse_triple_input(graph2_input)
    else:
        edges1 = parse_rdf_input(graph1_input, format=format_map[input_format])
        edges2 = parse_rdf_input(graph2_input, format=format_map[input_format])
    
    if edges1 and edges2:
        # Create appropriate graph type based on selection
        GraphType = nx.DiGraph if graph_type == "Directed" else nx.Graph
        G1 = GraphType()
        G2 = GraphType()
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
            st.write("### Matrix-based Similarities")
            matrix_sim = calculate_matrix_similarities(G1, G2)
            if matrix_sim:
                for metric, data in matrix_sim.items():
                    display_metric_with_details(
                        metric_name=metric,
                        value=data['value'],
                        explanation=explanations[metric],
                        actual_values=data['steps']
                    )
            else:
                st.warning("Unable to calculate matrix similarities")
        
        # Structural similarities tab
        with tab2:
            st.write("### Structural Similarities")
            struct_sim = compare_structural_similarity(G1, G2)
            for metric, data in struct_sim.items():
                display_metric_with_details(
                    metric_name=metric,
                    value=data['match'],
                    explanation=explanations[metric],
                    actual_values=data['values']
                )
        
        # Node-level similarities tab
        with tab3:
            st.write("### Node-level Similarities")
            node_sim = compare_node_similarity(G1, G2)
            for metric, value in node_sim.items():
                if metric in explanations:
                    # Get actual values from graphs for the metric
                    actual_values = get_node_metric_values(G1, G2, metric)
                    display_metric_with_details(
                        metric_name=metric,
                        value=value,
                        explanation=explanations[metric],
                        actual_values=actual_values
                    )
        
        # Embedding-based similarities tab
        with tab4:
            st.write("### Embedding-based Similarities")
            try:
                embedding_sim = compare_embeddings(G1, G2)
                for metric, data in embedding_sim.items():
                    display_metric_with_details(
                        metric_name=metric,
                        value=data['value'],
                        explanation=explanations[metric],
                        actual_values=data['steps']
                    )
            except Exception as e:
                st.error(f"Error calculating embedding similarities: {str(e)}")

def get_node_metric_values(G1, G2, metric):
    """Get actual values for node-level metrics"""
    try:
        if metric == 'Degree Distribution Match':
            return {
                'Graph 1 Distribution': dict(Counter([d for n, d in G1.degree()])),
                'Graph 2 Distribution': dict(Counter([d for n, d in G2.degree()]))
            }
        elif metric == 'Clustering Coefficient Difference':
            if isinstance(G1, nx.DiGraph) or isinstance(G2, nx.DiGraph):
                return "Unsupported for directed graphs"
            return {
                'Graph 1 Clustering': f"{nx.average_clustering(G1):.4f}",
                'Graph 2 Clustering': f"{nx.average_clustering(G2):.4f}"
            }
        elif metric == 'Average Betweenness Difference':
            if isinstance(G1, nx.DiGraph) or isinstance(G2, nx.DiGraph):
                return "Unsupported for directed graphs"
            bet1 = nx.betweenness_centrality(G1)
            bet2 = nx.betweenness_centrality(G2)
            return {
                'Graph 1 Average': f"{sum(bet1.values())/len(bet1):.4f}",
                'Graph 2 Average': f"{sum(bet2.values())/len(bet2):.4f}"
            }
        elif metric == 'Average Closeness Difference':
            if isinstance(G1, nx.DiGraph) or isinstance(G2, nx.DiGraph):
                return "Unsupported for directed graphs"
            close1 = nx.closeness_centrality(G1)
            close2 = nx.closeness_centrality(G2)
            return {
                'Graph 1 Average': f"{sum(close1.values())/len(close1):.4f}",
                'Graph 2 Average': f"{sum(close2.values())/len(close2):.4f}"
            }
    except Exception as e:
        return f"Error calculating values: {str(e)}"
    return None

if __name__ == "__main__":
    main() 