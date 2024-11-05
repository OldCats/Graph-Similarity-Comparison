# Graph Similarity Comparison Tool

A Streamlit web application for comparing two graphs using various similarity metrics. This tool provides visual and quantitative analysis of graph similarities through multiple comparison methods.

## Features

- Interactive web interface with:
  - Easy graph input through text areas (supports 'node1 node2' or 'node1,node2' format)
  - Side-by-side graph visualization with explanatory tooltips
  - Tabbed results display with detailed metric explanations

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
     - Average path length comparison
     - Graph diameter analysis

  3. Node-level Properties:
     - Degree distribution patterns
     - Clustering coefficient measurement
     - Betweenness centrality analysis
     - Closeness centrality comparison

- Visualization Features:
  - Interactive side-by-side graph plots
  - Color-coded nodes (lightblue for Graph 1, lightgreen for Graph 2)
  - Connected component count display
  - Clear node labels with size 16 font
  - Optimized spring layout with k=1 spacing

## Installation

1. Clone the repository
2. Install required dependencies using pip:
   ```bash
   pip install streamlit>=1.24.0
   pip install networkx>=3.1 
   pip install numpy>=1.24.0
   pip install matplotlib>=3.7.1
   pip install scipy>=1.10.1
   ```
   
   Or install all requirements at once:
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
     3 1
     ```
     or
     ```
     1,2
     2,3
     3,1
     ```

2. View Results:
   - Side-by-side graph visualizations
   - Detailed similarity metrics across multiple categories
   - Node-level property comparisons

3. Interpret Results:
   - Green checkmarks indicate matching properties
   - Numerical differences shown for continuous metrics
   - Connected component analysis for disconnected graphs

## Notes

- Graphs should be undirected and unweighted
- Node labels must be integers
- For best visualization, keep graphs under 20 nodes
- Some metrics only available for connected graphs
- Large graphs may take longer to process

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## Demo

Try out the live demo at: [Graph Similarity Comparison Demo](https://oldcats-graph-similarity-comparison-graph-similarity-app-dfeblx.streamlit.app/)

The demo site allows you to:
- Input sample graphs and see visualizations
- Compare structural properties in real-time 
- Explore different graph metrics and similarity measures
- Test with both connected and disconnected graphs



