import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import numpy as np
import os


def plot_networks(G_interactions, G_mentions):
    plt.figure(figsize=(20, 10))

    k = 2 / np.sqrt(len(G_interactions.nodes()))
    iterations = 100

    plt.subplot(121)
    pos_interactions = nx.spring_layout(
        G_interactions,
        k=k,
        iterations=iterations,
        seed=42,
        weight="weight",
    )

    nx.draw_networkx_edges(
        G_interactions,
        pos_interactions,
        width=[G_interactions[u][v]["weight"] / 5 for u, v in G_interactions.edges()],
        alpha=0.3,
    )

    nodes = nx.draw_networkx_nodes(
        G_interactions,
        pos_interactions,
        node_size=[
            G_interactions.nodes[node]["value"] * 50 for node in G_interactions.nodes
        ],
        node_color=[
            G_interactions.nodes[node]["colour"] for node in G_interactions.nodes
        ],
        alpha=0.7,
    )

    labels = nx.draw_networkx_labels(
        G_interactions,
        pos_interactions,
        font_size=8,
        font_weight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
    )

    plt.title("Episode 1 Interactions Network", pad=20)

    plt.subplot(122)
    pos_mentions = nx.spring_layout(
        G_mentions, k=k, iterations=iterations, seed=42, weight="weight"
    )

    nx.draw_networkx_edges(
        G_mentions,
        pos_mentions,
        width=[G_mentions[u][v]["weight"] / 5 for u, v in G_mentions.edges()],
        alpha=0.3,
    )

    nodes = nx.draw_networkx_nodes(
        G_mentions,
        pos_mentions,
        node_size=[G_mentions.nodes[node]["value"] * 50 for node in G_mentions.nodes],
        node_color=[G_mentions.nodes[node]["colour"] for node in G_mentions.nodes],
        alpha=0.7,
    )

    labels = nx.draw_networkx_labels(
        G_mentions,
        pos_mentions,
        font_size=8,
        font_weight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
    )

    plt.title("Episode 1 Mentions Network", pad=20)

    plt.tight_layout(pad=3.0)

    for ax in plt.gcf().get_axes():
        ax.set_axis_off()

    plt.show()


def plot_strength_distribution(G, network_name):
    strengths = [
        sum(d["weight"] for _, _, d in G.edges(node, data=True)) for node in G.nodes()
    ]
    plt.figure(figsize=(8, 5))
    plt.hist(strengths, bins=20)
    plt.title(f"Node Strength Distribution - {network_name}")
    plt.xlabel("Node Strength")
    plt.ylabel("Frequency")
    plt.show()


def plot_communities(G, communities, network_name):
    plt.figure(figsize=(12, 8))

    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(communities) > len(colors):
        colors = plt.cm.tab20(np.linspace(0, 1, len(communities)))

    color_map = {}
    for i, community in enumerate(communities):
        for node in community:
            color_map[node] = colors[i % len(colors)]

    pos = nx.spring_layout(G, k=1 / np.sqrt(len(G.nodes())), iterations=50)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=[G.nodes[node]["value"] * 50 for node in G.nodes],
        node_color=[color_map[node] for node in G.nodes],
        alpha=0.7,
    )

    nx.draw_networkx_edges(
        G, pos, width=[G[u][v]["weight"] / 5 for u, v in G.edges()], alpha=0.5
    )

    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(f"Communities in {network_name}")

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[i % len(colors)],
            label=f"Community {i+1}",
            markersize=10,
        )
        for i in range(len(communities))
    ]
    plt.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()


def calculate_node_sizes(G, size_by: str, scaling_factor: float = 1.0):
    """Calculate node sizes based on different centrality measures"""
    if size_by == "value":
        return {node: G.nodes[node]["value"] * 50 * scaling_factor for node in G.nodes()}
    elif size_by == "degree":
        return {node: deg * 50 * scaling_factor for node, deg in dict(G.degree()).items()}
    elif size_by == "closeness":
        return {node: val * 5000 * scaling_factor for node, val in nx.closeness_centrality(G).items()}
    elif size_by == "betweenness":
        return {node: val * 10000 * scaling_factor for node, val in nx.betweenness_centrality(G).items()}
    elif size_by == "eigenvector":
        return {node: val * 1000 * scaling_factor for node, val in nx.eigenvector_centrality(G).items()}
    else:
        raise ValueError(f"Unknown size_by parameter: {size_by}")


def plot_node_size(G, layout_name: str, node_size: int = 1, size_by: str = "value"):
    """
    Plot network with node sizes based on different centrality measures
    size_by options: 'value', 'degree', 'closeness', 'betweenness', 'eigenvector'
    """
    os.makedirs(f"plots/node_size/{size_by}", exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.title(f"Star Wars Interactions Network - {layout_name}\nSize by: {size_by}")

    try:
        if layout_name in ["forceatlas2_layout", "rescale_layout", "rescale_layout_dict", "arf_layout"]:
            return
            
        node_sizes = calculate_node_sizes(G, size_by, node_size)

        layout_func = getattr(nx, layout_name)
        pos = layout_func(G)

        nx.draw(
            G,
            pos=pos,
            node_color="lightblue",
            node_size=[node_sizes[node] for node in G.nodes()],
            with_labels=True,
            font_size=8,
            edge_color="gray",
            alpha=0.7,
        )

        plt.tight_layout()
        plt.savefig(f"plots/node_size/{size_by}/{layout_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"Error plotting {layout_name}: {str(e)}")
        plt.close()


def plot_edge_weight(G, layout_name: str, node_size: int = 0, edge_weight: bool = False):
    os.makedirs("plots/edge_weight", exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.title(f"Star Wars Interactions Network - {layout_name}")

    try:
        if layout_name == "forceatlas2_layout":
            return
        elif layout_name == "rescale_layout" or layout_name == "rescale_layout_dict":
            return
        elif layout_name == "arf_layout":
            return
        else:
            layout_func = getattr(nx, layout_name)
            pos = layout_func(G)

        if edge_weight:
            edge_weights = [G[u][v]["weight"] / 5 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color="gray", alpha=0.5)

            nx.draw_networkx_nodes(
                G,
                pos=pos,
                node_color="lightblue",
                node_size=[G.nodes[node]["value"] * node_size for node in G.nodes()],
                alpha=0.7,
            )
            nx.draw_networkx_labels(G, pos=pos, font_size=8)
        else:
            if node_size > 0:
                nx.draw(
                    G,
                    pos=pos,
                    node_color="lightblue",
                    node_size=[G.nodes[node]["value"] * node_size for node in G.nodes()],
                    with_labels=True,
                    font_size=8,
                    edge_color="gray",
                    alpha=0.7,
                )
            else:
                nx.draw(
                    G,
                    pos=pos,
                    node_color="lightblue",
                    with_labels=True,
                    font_size=8,
                    edge_color="gray",
                    alpha=0.7,
                )

        plt.tight_layout()
        plt.savefig(f"plots/edge_weight/{layout_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"Error plotting {layout_name}: {str(e)}")


def plot_community_with_layout(G, layout_name: str, communities, community_type: str, size_by: str = "value"):
    """
    Plot communities with node sizes based on different centrality measures
    size_by options: 'value', 'degree', 'closeness', 'betweenness', 'eigenvector'
    """
    os.makedirs(f"plots/communities/{size_by}", exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    plt.title(f"Communities using {community_type}\nLayout: {layout_name}\nSize by: {size_by}")

    try:
        if layout_name in ["forceatlas2_layout", "rescale_layout", "rescale_layout_dict", "arf_layout",
                          "bipartite_layout", "bfs_layout", "multipartite_layout", "planar_layout"]:
            return
        
        # Calculate node sizes based on selected centrality measure
        if size_by == "value":
            node_sizes = {node: G.nodes[node]["value"] * 50 for node in G.nodes()}
        elif size_by == "degree":
            node_sizes = {node: deg * 50 for node, deg in dict(G.degree()).items()}
        elif size_by == "closeness":
            node_sizes = {node: val * 5000 for node, val in nx.closeness_centrality(G).items()}
        elif size_by == "betweenness":
            node_sizes = {node: val * 100 for node, val in nx.betweenness_centrality(G).items()}
        elif size_by == "eigenvector":
            node_sizes = {node: val * 1000 for node, val in nx.eigenvector_centrality(G).items()}
        else:
            raise ValueError(f"Unknown size_by parameter: {size_by}")

        layout_func = getattr(nx, layout_name)
        try:
            pos = layout_func(G)
        except Exception as e:
            print(f"Error computing {layout_name} layout: {str(e)}")
            return

        colors = list(mcolors.TABLEAU_COLORS.values())
        if len(communities) > len(colors):
            colors = plt.cm.tab20(np.linspace(0, 1, len(communities)))

        color_map = {}
        for i, community in enumerate(communities):
            for node in community:
                color_map[node] = colors[i % len(colors)]

        nx.draw_networkx_edges(
            G,
            pos,
            width=[G[u][v]["weight"] / 5 for u, v in G.edges()],
            edge_color="gray",
            alpha=0.3
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=[node_sizes[node] for node in G.nodes()],
            node_color=[color_map[node] for node in G.nodes()],
            alpha=0.7
        )

        nx.draw_networkx_labels(
            G,
            pos,
            font_size=8,
            font_weight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5)
        )

        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[i % len(colors)],
                label=f"Community {i+1}",
                markersize=10,
            )
            for i in range(len(communities))
        ]
        plt.legend(
            handles=legend_elements,
            loc="center left",
            bbox_to_anchor=(1, 0.5)
        )

        plt.tight_layout()
        plt.savefig(
            f"plots/communities/{size_by}/{community_type}_{layout_name}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    except Exception as e:
        print(f"Error plotting {layout_name} for {community_type}: {str(e)}")
        plt.close()
