import os
import networkx as nx
from typing import List
import utils
from pprint import pprint


def get_top_nodes_by_files(
    files: List[str], centrality_type: str = "degree", top_n: int = 3
):
    top_nodes = []
    for i, file in enumerate(files):
        G = utils.get_graph_with_nodes_and_edges(
            nx.Graph(), utils.load_json(f"data/{file}")
        )
        episode = "Episode" + " " + file.split("-")[2]
        top_nodes.append(episode)
        if centrality_type == "degree":
            top_nodes.append(
                sorted(nx.degree_centrality(G).items(), key=lambda x: -x[1])[:top_n]
            )
        elif centrality_type == "closeness":
            top_nodes = sorted(nx.closeness_centrality(G).items(), key=lambda x: -x[1])[
                :top_n
            ]
        elif centrality_type == "betweenness":
            top_nodes.append(
                sorted(nx.betweenness_centrality(G).items(), key=lambda x: -x[1])[
                    :top_n
                ]
            )
        else:
            raise ValueError(f"Unknown centrality type: {centrality_type}")
    return top_nodes


def get_top_nodes_by_graph(
    G: nx.Graph, centrality_type: str = "degree", top_n: int = 3
):
    if centrality_type == "degree":
        return sorted(nx.degree_centrality(G).items(), key=lambda x: -x[1])[:top_n]
    elif centrality_type == "closeness":
        return sorted(nx.closeness_centrality(G).items(), key=lambda x: -x[1])[:top_n]
    elif centrality_type == "betweenness":
        return sorted(nx.betweenness_centrality(G).items(), key=lambda x: -x[1])[:top_n]
    else:
        raise ValueError(f"Unknown centrality type: {centrality_type}")


def get_top_nodes_by_community(
    G: nx.Graph, community: List[str], centrality_type: str = "degree", top_n: int = 3
):
    G_community = G.subgraph(community)
    return get_top_nodes_by_graph(G_community, centrality_type, top_n)


if __name__ == "__main__":
    pass
