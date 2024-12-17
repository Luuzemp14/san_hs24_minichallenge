import networkx as nx


def get_girvan_newman_communities(G: nx.Graph):
    return list(nx.community.girvan_newman(G))

def get_louvain_communities(G: nx.Graph):
    return list(nx.community.louvain_communities(G))

def get_label_propagation_communities(G: nx.Graph):
    return list(nx.community.label_propagation_communities(G))
