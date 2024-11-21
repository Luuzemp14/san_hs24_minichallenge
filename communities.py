import networkx as nx


def get_girvan_newman_communities(G: nx.Graph):
    return list(nx.community.girvan_newman(G))
