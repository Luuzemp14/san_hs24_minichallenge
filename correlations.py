import utils
import networkx as nx
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import os
    
def get_spearman_and_pearson_correlations(graph):
    """
    Analyze the correlation between node 'value' attribute and centrality measures of a graph.

    Parameters:
        graph (networkx.Graph): A graph where nodes have a 'value' attribute.

    Returns:
        dict: A dictionary containing Pearson and Spearman correlation coefficients.
    """
    # Ensure all nodes have a 'value' attribute
    if not all('value' in graph.nodes[node] for node in graph.nodes()):
        raise ValueError("All nodes in the graph must have a 'value' attribute.")
    os.makedirs("plots/correlations/", exist_ok=True)
    # Step 1: Compute centrality measures
    degree_centrality = nx.degree_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    
    # Step 2: Create a DataFrame for analysis
    data = pd.DataFrame({
        "Value": [graph.nodes[node]["value"] for node in graph.nodes()],
        "Degree Centrality": [degree_centrality[node] for node in graph.nodes()],
        "Closeness Centrality": [closeness_centrality[node] for node in graph.nodes()],
        "Betweenness Centrality": [betweenness_centrality[node] for node in graph.nodes()]
    })

    # Step 3: Compute correlation coefficients
    pearson_corr = data.corr(method='pearson')
    spearman_corr = data.corr(method='spearman')
    
    paths = []
    for centrality in ["Degree Centrality", "Closeness Centrality", "Betweenness Centrality"]:
        path = f"plots/correlations/value_vs_{centrality}.png"
        paths.append(path)
        plt.figure(figsize=(6, 4))
        plt.scatter(data["Value"], data[centrality], alpha=0.7)
        plt.title(f"Appearance (Node Attribute) vs {centrality}")
        plt.xlabel("N")
        plt.ylabel(centrality)
        plt.grid(True)
        plt.savefig(path)
        plt.close()

    return {
        "Pearson Correlation": pearson_corr,
        "Spearman Correlation": spearman_corr,
        "paths": paths
    }

if __name__ == "__main__":  
    pass
