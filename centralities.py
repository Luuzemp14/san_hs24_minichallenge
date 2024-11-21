import networkx as nx
from typing import List


def get_actor_centralities(G: nx.Graph, types: List[str]):
    print(f"\n{G.name} Centrality Analysis:")

    if "degree" in types:
        degree_cent = nx.degree_centrality(G)
        print("\nTop 5 characters by Degree Centrality:")
        for char, cent in sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]:
            print(f"{char}: {cent:.3f}")

    if "betweenness" in types:
        between_cent = nx.betweenness_centrality(G)
        print("\nTop 5 characters by Betweenness Centrality:")
        for char, cent in sorted(
            between_cent.items(), key=lambda x: x[1], reverse=True
        )[:5]:
            print(f"{char}: {cent:.3f}")

    if "eigenvector" in types:
        eigen_cent = nx.eigenvector_centrality(G)
        print("\nTop 5 characters by Eigenvector Centrality:")
        for char, cent in sorted(eigen_cent.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]:
            print(f"{char}: {cent:.3f}")
