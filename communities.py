import networkx as nx


def get_girvan_newman_communities(G: nx.Graph):
    G.remove_node("GOLD FIVE")
    return list(nx.community.girvan_newman(G))


def get_louvain_communities(G: nx.Graph, seed: int = 42):
    G.remove_node("GOLD FIVE")
    communities = list(nx.community.louvain_communities(G, seed=seed))
    return communities


def get_label_propagation_communities(G: nx.Graph):
    G.remove_node("GOLD FIVE")
    return list(nx.community.label_propagation_communities(G))
