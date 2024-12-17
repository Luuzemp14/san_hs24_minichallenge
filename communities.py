from pprint import pprint
import networkx as nx
import utils


def get_girvan_newman_communities(G: nx.Graph):
    G.remove_node("GOLD FIVE")
    return list(nx.community.girvan_newman(G))


def get_louvain_communities(file_path: str, seed: int = 42):
    G = utils.get_graph_from_file(file_path)
    G.remove_node("GOLD FIVE")
    communities = list(nx.community.louvain_communities(G, seed=seed))
    return communities


def get_label_propagation_communities(G: nx.Graph):
    G.remove_node("GOLD FIVE")
    return list(nx.community.label_propagation_communities(G))


if __name__ == "__main__":
    pass
