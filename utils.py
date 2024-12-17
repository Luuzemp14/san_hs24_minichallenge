import json
import networkx as nx


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_graph_with_nodes_and_edges(G: nx.Graph, data: dict):
    for node in data["nodes"]:
        G.add_node(node["name"], value=node["value"], colour=node["colour"])

    for link in data["links"]:
        source = data["nodes"][link["source"]]["name"]
        target = data["nodes"][link["target"]]["name"]
        G.add_edge(source, target, weight=link["value"])

    return G


def get_network_statistics(G: nx.Graph):
    print("\nNetwork Statistics for {}:".format(G.name))
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Network density: {nx.density(G):.3f}")
    print(f"Average clustering coefficient: {nx.average_clustering(G):.3f}")


def get_graph_from_file(file: str):
    return get_graph_with_nodes_and_edges(nx.Graph(), load_json(f"data/{file}"))
