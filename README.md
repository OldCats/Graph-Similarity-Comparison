# Graph Similarity Comparison Tool

![Graph Comparison Tool](graph_comparison.png)

A Streamlit web application for comparing both directed and undirected graphs using various similarity metrics. This tool supports multiple input formats including Edge List, RDF, and Triple formats, providing visual and quantitative analysis of graph similarities through multiple comparison methods.

## Features

- Interactive web interface with:
  - Graph type selection:
    - Undirected graphs
    - Directed graphs (with arrow visualization)
  - Multiple input format support:
    - Edge List (simple node pairs)
    - TTL (Triple) format
    - RDF/XML format
    - N-Triples format
  - Side-by-side graph visualization with component counts
  - Tabbed results display with detailed metric explanations and formulas

- Comprehensive similarity metrics:
  1. Matrix-based Similarities (supports both directed and undirected):
     - Jaccard Similarity (ratio of common edges to total edges)
     - Cosine Similarity (adjacency matrix angle comparison)
     - Spectral Similarity (eigenvalue-based structural comparison)
     - Graph Edit Distance Similarity (minimum transformation cost)

  2. Structural Similarities:
     - Node and edge count matching
     - Degree sequence comparison
     - Graph density analysis
     - Connected components detection (uses weak connectivity for directed graphs)
     - Average path length comparison (for connected graphs)
     - Graph diameter analysis (for connected graphs)

  3. Node-level Properties:
     - Degree distribution patterns
     - Clustering coefficient measurement (undirected only)
     - Betweenness centrality analysis (undirected only)
     - Closeness centrality comparison (undirected only)

  4. Embedding-based Similarities:
     - Graph-level embedding similarity
     - Average node embedding similarity
     - Node embedding distribution visualization

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

1. Select Graph Type:
   - Choose between Undirected and Directed graphs
   - Note that some metrics are only available for undirected graphs

2. Select Input Format:
   Choose from four supported formats:

   a. Edge List format (direction: from first node to second):
   ```
   1 2
   2 3
   3 4
   4 1
   2 4
   ```

   b. TTL (Triple) format (direction: from subject to object):
   ```
   <node1> <connects> <node2> .
   <node2> <connects> <node3> .
   <node3> <connects> <node4> .
   ```

   c. RDF/XML format (direction: from subject to object):
   ```xml
   <?xml version="1.0"?>
   <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
            xmlns:ex="http://example.org/">
     <rdf:Description rdf:about="http://example.org/node1">
       <ex:connects rdf:resource="http://example.org/node2"/>
     </rdf:Description>
   </rdf:RDF>
   ```

   d. N-Triples format (direction: from subject to object):
   ```
   <http://example.org/node1> <http://example.org/connects> <http://example.org/node2> .
   <http://example.org/node2> <http://example.org/connects> <http://example.org/node3> .
   ```

3. Input Graph Data:
   - Enter graph data in the selected format for both graphs
   - For directed graphs, edge direction follows input order
   - For RDF formats, direction goes from subject to object
   - The tool automatically extracts or generates numeric node IDs

4. View Results:
   - Graph visualizations with arrows for directed graphs
   - Component counts (using weak connectivity for directed graphs)
   - Detailed similarity metrics with clear indication of unsupported metrics
   - Interactive formulas and calculation steps

5. Interpret Results:
   - ✅ indicates exact matches between graphs
   - ❌ indicates differences between graphs
   - "Unsupported for directed graphs" shown for inapplicable metrics
   - Similarity values range from 0.0 (different) to 1.0 (identical)

## Notes

- Supports both directed and undirected graph analysis
- Some metrics (clustering, betweenness, closeness) only available for undirected graphs
- Uses weak connectivity for directed graph components
- Automatically handles metric availability based on graph type
- Shows clear messages for unsupported metrics
- Default examples provided for each input format and graph type

## Implementation Details

- Uses NetworkX for graph operations and metrics
- Implements RDFLib for parsing RDF formats
- Calculates matrix similarities using numpy/scipy
- Handles both directed and undirected graph visualization
- Provides appropriate metrics based on graph type
- Shows arrows in visualization for directed graphs
- Optimized visualization using matplotlib

## Live Demo

Try the app at: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://oldcats-graph-similarity-comparison-graph-similarity-app-dfeblx.streamlit.app/)
