import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import utils
import networkx as nx
from typing import List


def print_actor_centrality(G: nx.Graph, centrality_type: str):
    print(f"\n{G.name} Centrality Analysis:")

    if centrality_type == "closeness":
        closeness_cent = nx.closeness_centrality(G)
        print("\nTop 5 characters by Closeness Centrality:")
        for char, cent in sorted(
            closeness_cent.items(), key=lambda x: x[1], reverse=True
        )[:5]:
            print(f"{char}: {cent:.3f}")

    if centrality_type == "degree":
        degree_cent = nx.degree_centrality(G)
        print("\nTop 5 characters by Degree Centrality:")
        for char, cent in sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]:
            print(f"{char}: {cent:.3f}")

    if centrality_type == "betweenness":
        between_cent = nx.betweenness_centrality(G)
        print("\nTop 5 characters by Betweenness Centrality:")
        for char, cent in sorted(
            between_cent.items(), key=lambda x: x[1], reverse=True
        )[:5]:
            print(f"{char}: {cent:.3f}")

    if centrality_type == "eigenvector":
        eigen_cent = nx.eigenvector_centrality(G)
        print("\nTop 5 characters by Eigenvector Centrality:")
        for char, cent in sorted(eigen_cent.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]:
            print(f"{char}: {cent:.3f}")

def plot_venn_diagram(G: nx.Graph, centrality_types: List[str]):
    centrality_types = centrality_types[:3]
    top_chars = {}
    for centrality_type in centrality_types:
        if centrality_type == "closeness":
            cent = nx.closeness_centrality(G)
        elif centrality_type == "degree":
            cent = nx.degree_centrality(G)
        elif centrality_type == "betweenness":
            cent = nx.betweenness_centrality(G)
        elif centrality_type == "eigenvector":
            cent = nx.eigenvector_centrality(G)
            
        top_chars[centrality_type] = set(
            [char for char, cent_value in sorted(cent.items(), key=lambda x: x[1], reverse=True)[:5]]
        )
    
    plt.figure(figsize=(10, 10))
    venn3([top_chars[ct] for ct in centrality_types], set_labels=centrality_types)
    plt.title(f"{G.name} - Top 5 Characters Overlap by Centrality Measure")
    plt.show()

if __name__ == "__main__":
    pass
