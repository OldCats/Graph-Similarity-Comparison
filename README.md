# Graph Similarity Comparison Tool

![Graph Comparison Tool](graph_comparison.png)

A Streamlit web application for comparing two graphs using various similarity metrics. This tool provides visual and quantitative analysis of graph similarities through multiple comparison methods.

## Features

- Interactive web interface with:
  - Easy graph input through text areas (supports 'node1 node2' or 'node1,node2' format)
  - Side-by-side graph visualization with component counts
  - Tabbed results display with detailed metric explanations and formulas

- Comprehensive similarity metrics:
  1. Matrix-based Similarities:
     - Jaccard Similarity (ratio of common edges to total edges)
     - Cosine Similarity (adjacency matrix angle comparison)
     - Spectral Similarity (eigenvalue-based structural comparison)
     - Graph Edit Distance Similarity (minimum transformation cost)

  2. Structural Similarities:
     - Node and edge count matching
     - Degree sequence comparison
     - Graph density analysis
     - Connected components detection
     - Average path length comparison (for connected graphs)
     - Graph diameter analysis (for connected graphs)

  3. Node-level Properties:
     - Degree distribution patterns
     - Clustering coefficient measurement
     - Betweenness centrality analysis
     - Closeness centrality comparison

  4. Embedding-based Similarities:
     - Graph-level embedding similarity using node2vec
     - Average node embedding similarity
     - Node embedding distribution visualization

- Visualization Features:
  - Side-by-side graph plots
  - Color-coded nodes (lightblue for Graph 1, lightgreen for Graph 2)
  - Connected component count display
  - Clear node labels with size 16 font
  - Spring layout with k=1 spacing

## Installation

1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run graph_similarity_app.py
   ```

## Usage

1. Input Graph Data:
   - Enter edges for both graphs using either format:
     ```
     1 2
     2 3
     3 4
     4 1
     2 4
     ```
     or
     ```
     1,2
     2,3
     3,4
     4,1
     2,4
     ```

2. View Results:
   - Graph visualizations with component counts
   - Detailed similarity metrics across four categories:
     - Matrix-based Similarities
     - Structural Similarities  
     - Node-level Similarities
     - Embedding-based Similarities
   - Interactive formulas and calculation steps
   - Distribution visualizations for node embeddings

3. Interpret Results:
   - ✅ indicates exact matches between graphs
   - ❌ indicates differences between graphs
   - Similarity values range from 0.0 (different) to 1.0 (identical)
   - For differences, values closer to 0.0 indicate more similarity

## Notes

- Graphs should be undirected and unweighted
- Node labels must be integers
- Some metrics (like average path length) only available for connected graphs
- Embedding calculations require connected graphs with sufficient nodes
- Default sample graphs demonstrate basic usage

## Implementation Details

- Uses NetworkX for graph operations and metrics
- Implements node2vec for graph embeddings
- Calculates matrix similarities using numpy/scipy
- Handles disconnected graphs by computing metrics per component
- Provides detailed calculation steps and formulas
- Optimized visualization using matplotlib
