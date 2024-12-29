import communities
import importance
import os
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
from typing import List, Tuple
from collections import defaultdict
from networkx.algorithms.community import girvan_newman
from community import community_louvain
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import numpy as np
import os
from typing import List
import utils
import json
import seaborn as sns
from collections import defaultdict


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
        return {
            node: G.nodes[node]["value"] * 50 * scaling_factor for node in G.nodes()
        }
    elif size_by == "degree":
        return {
            node: deg * 50 * scaling_factor for node, deg in dict(G.degree()).items()
        }
    elif size_by == "closeness":
        return {
            node: val * 5000 * scaling_factor
            for node, val in nx.closeness_centrality(G).items()
        }
    elif size_by == "betweenness":
        return {
            node: val * 10000 * scaling_factor
            for node, val in nx.betweenness_centrality(G).items()
        }
    elif size_by == "eigenvector":
        return {
            node: val * 1000 * scaling_factor
            for node, val in nx.eigenvector_centrality(G).items()
        }
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
        if layout_name in [
            "forceatlas2_layout",
            "rescale_layout",
            "rescale_layout_dict",
            "arf_layout",
        ]:
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
        plt.savefig(
            f"plots/node_size/{size_by}/{layout_name}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    except Exception as e:
        print(f"Error plotting {layout_name}: {str(e)}")
        plt.close()


def plot_edge_weight(
    G, layout_name: str, node_size: int = 0, edge_weight: bool = False
):
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
            nx.draw_networkx_edges(
                G, pos, width=edge_weights, edge_color="gray", alpha=0.5
            )

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
                    node_size=[
                        G.nodes[node]["value"] * node_size for node in G.nodes()
                    ],
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
        plt.savefig(
            f"plots/edge_weight/{layout_name}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    except Exception as e:
        print(f"Error plotting {layout_name}: {str(e)}")


def create_community_node_colors(graph, communities):
    number_of_colors = len(communities)
    colors = ["#D4FCB1", "#CDC5FC", "#FFC2C4", "#F2D140", "#BCC6C8"][:number_of_colors]
    node_colors = []
    for node in graph:
        current_community_index = 0
        for community in communities:
            if node in community:
                node_colors.append(colors[current_community_index])
                break
            current_community_index += 1
    return node_colors


def plot_girvan_newman_communities(graph, layout="spring_layout", level=1):
    """
    Detects and plots Girvan-Newman communities for a given graph using the specified layout.

    Parameters:
        graph (networkx.Graph): The graph to analyze and plot.
        layout (str): The layout to use for positioning nodes. Options: "circular_layout",
                      "kamada_kawai_layout", "random_layout", "shell_layout",
                      "spring_layout", "fruchterman_reingold_layout".
        level (int): The level of the hierarchy to extract communities (higher levels = fewer communities).
    """
    LAYOUTS = [
        "circular_layout",
        "kamada_kawai_layout",
        "random_layout",
        "shell_layout",
        "spring_layout",
        "fruchterman_reingold_layout",
    ]

    if layout not in LAYOUTS:
        raise ValueError(f"Invalid layout. Choose from: {', '.join(LAYOUTS)}")

    # Detect Girvan-Newman communities
    communities_generator = girvan_newman(graph)
    for _ in range(level - 1):  # Skip levels until the desired one
        next(communities_generator)
    communities = next(communities_generator)
    community_mapping = {
        node: i for i, community in enumerate(communities) for node in community
    }

    # Assign colors based on communities
    colors = [community_mapping[node] for node in graph.nodes()]

    # Get layout positions
    layout_function = getattr(nx, layout)
    pos = layout_function(graph)

    # Plot the graph
    plt.figure(figsize=(10, 8))
    nx.draw(
        graph,
        pos,
        node_color=colors,
        with_labels=True,
        node_size=500,
        cmap=plt.cm.tab10,
    )
    plt.title(f"Girvan-Newman Communities ({layout}, Level {level})")
    plt.show()


def plot_louvain_communities(graph, layout="spring_layout"):
    """
    Detects and plots Louvain communities for a given graph using the specified layout.

    Parameters:
        graph (networkx.Graph): The graph to analyze and plot.
        layout (str): The layout to use for positioning nodes. Options: "circular_layout",
                      "kamada_kawai_layout", "random_layout", "shell_layout",
                      "spring_layout", "fruchterman_reingold_layout".
    """
    LAYOUTS = [
        "circular_layout",
        "kamada_kawai_layout",
        "random_layout",
        "shell_layout",
        "spring_layout",
        "fruchterman_reingold_layout",
    ]

    if layout not in LAYOUTS:
        raise ValueError(f"Invalid layout. Choose from: {', '.join(LAYOUTS)}")

    # Detect Louvain communities
    partition = community_louvain.best_partition(graph)

    # Assign colors based on communities
    colors = [partition[node] for node in graph.nodes()]

    # Get layout positions
    layout_function = getattr(nx, layout)
    pos = layout_function(graph)

    # Plot the graph
    plt.figure(figsize=(10, 8))
    nx.draw(
        graph,
        pos,
        node_color=colors,
        with_labels=True,
        node_size=500,
        cmap=plt.cm.viridis,
    )
    plt.title(f"Louvain Communities ({layout})")
    plt.show()


def plot_centrality_over_time(files: List[str], centrality_type: str = "degree"):
    rows, cols = 3, 3  # 4x3 grid
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten()

    for i, file in enumerate(files):
        if i >= len(axes):
            break  # Avoid plotting more than the grid allows

        G = utils.get_graph_with_nodes_and_edges(
            nx.Graph(), utils.load_json(f"data/{file}")
        )
        episode = file.split("-")[2]
        if centrality_type == "degree":
            top_nodes = sorted(nx.degree_centrality(G).items(), key=lambda x: -x[1])[:3]
        elif centrality_type == "closeness":
            top_nodes = sorted(nx.closeness_centrality(G).items(), key=lambda x: -x[1])[
                :3
            ]
        elif centrality_type == "betweenness":
            top_nodes = sorted(
                nx.betweenness_centrality(G).items(), key=lambda x: -x[1]
            )[:3]
        else:
            raise ValueError(f"Unknown centrality type: {centrality_type}")
        nodes, values = zip(*top_nodes)
        axes[i].bar(nodes, values)
        axes[i].set_title(f"Episode {episode}")
        axes[i].set_ylabel(centrality_type.capitalize())
        axes[i].set_xlabel("Node")
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_centrality(file: str, centrality_type: str = "degree", top_n: int = 3):
    G = utils.get_graph_with_nodes_and_edges(
        nx.Graph(), utils.load_json(f"data/{file}")
    )
    if centrality_type == "degree":
        centrality_dict = nx.degree_centrality(G)
    elif centrality_type == "closeness":
        centrality_dict = nx.closeness_centrality(G)
    elif centrality_type == "betweenness":
        centrality_dict = nx.betweenness_centrality(G)
    else:
        raise ValueError(f"Unknown centrality type: {centrality_type}")

    top_nodes = sorted(centrality_dict.items(), key=lambda x: -x[1])[:top_n]
    nodes, values = zip(*top_nodes)
    plt.figure(figsize=(12, 6))
    plt.bar(nodes, values)
    episode = file.split("-")[2]
    plt.title(
        f"Top {top_n} Characters by {centrality_type.capitalize()} Centrality - {file}"
    )
    plt.ylabel(f"{centrality_type.capitalize()} Centrality")
    plt.xlabel("Character")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


def plot_centralities(file: str, top_n: int = 3):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    G = utils.get_graph_with_nodes_and_edges(
        nx.Graph(), utils.load_json(f"data/{file}")
    )

    centrality_types = ["degree", "closeness", "betweenness"]

    for i, centrality_type in enumerate(centrality_types):
        if centrality_type == "degree":
            centrality_dict = nx.degree_centrality(G)
        elif centrality_type == "closeness":
            centrality_dict = nx.closeness_centrality(G)
        else:
            centrality_dict = nx.betweenness_centrality(G)

        top_nodes = sorted(centrality_dict.items(), key=lambda x: -x[1])[:top_n]
        nodes, values = zip(*top_nodes)
        axes[i].bar(nodes, values)
        axes[i].set_title(f"{centrality_type.capitalize()} Centrality")
        axes[i].tick_params(axis="x", rotation=45)

    plt.suptitle(f"Top {top_n} Characters by Different Centrality Measures - {file}")
    plt.tight_layout()
    plt.show()


def plot_actor_appearances():
    # Read the JSON file
    with open("main_actors.json", "r") as f:
        data = json.load(f)

    # Create a dictionary to store actor appearances
    actor_appearances = defaultdict(list)

    # Process the data
    for episode_num, episode_data in data.items():
        for actor_info in episode_data["main_actors"]:
            actor_appearances[actor_info["actor"]].append(
                {"episode": episode_data["title"], "role": actor_info["role"]}
            )

    # Create figure
    plt.figure(figsize=(12, 6))

    # Create a list of unique episodes and actors
    episodes = [f"Episode {i}: {data[str(i)]['title']}" for i in range(1, 8)]
    # Modified to include role in the actor labels
    actors = [
        f"{actor} ({actor_appearances[actor][0]['role']})"
        for actor in actor_appearances.keys()
    ]

    # Create a matrix of appearances
    appearance_matrix = np.zeros((len(actors), len(episodes)))

    # Fill the matrix
    for i, actor in enumerate(actors):
        actor_name = actor.split(" (")[0]  # Extract just the actor name without role
        for appearance in actor_appearances[actor_name]:
            j = next(
                idx for idx, ep in enumerate(episodes) if appearance["episode"] in ep
            )
            appearance_matrix[i, j] = 1

    # Create heatmap
    sns.heatmap(
        appearance_matrix,
        xticklabels=episodes,
        yticklabels=actors,
        cmap="YlOrRd",
        cbar=False,
    )

    plt.title("Star Wars Main Actors Appearances Across Episodes")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return plt


def plot_top_nodes(top_nodes, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    nodes, values = zip(*top_nodes)
    ax.bar(nodes, values)
    ax.set_title(title)
    ax.set_ylabel("Centrality")
    ax.set_xlabel("Character")
    ax.tick_params(axis="x", rotation=45)

    return ax


def plot_top_nodes_by_community(file_path, centrality_type="betweenness", top_n=3):
    G_interactions = utils.get_graph_from_file(file_path)
    communities_list = communities.get_louvain_communities(file_path)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    plt.suptitle(
        f"Top {top_n} Nodes by {centrality_type.capitalize()} Centrality for Each Community"
    )

    for idx, community in enumerate(communities_list):
        G_community = G_interactions.subgraph(community)
        top_nodes = importance.get_top_nodes_by_graph(
            G_community, centrality_type=centrality_type, top_n=top_n
        )
        axes[idx] = plot_top_nodes(top_nodes, f"Community {idx+1}", ax=axes[idx])

    for idx in range(len(communities_list), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_louvain_communities(file_path):
    G = utils.get_graph_from_file(file_path)
    louvain_communities_result = communities.get_louvain_communities(file_path)
    num_louvain_communities = len(louvain_communities_result)
    distinct_colors = [
        plt.cm.tab20(i / num_louvain_communities)
        for i in range(num_louvain_communities)
    ]
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    axes = axes.flatten()
    for i, community in enumerate(louvain_communities_result):
        if i >= rows * cols:
            break
        subgraph = G.subgraph(community)
        pos_subgraph = nx.spring_layout(subgraph, seed=42, k=3)
        ax = axes[i]
        nx.draw_networkx_nodes(
            subgraph,
            pos_subgraph,
            node_color=[distinct_colors[i]] * len(subgraph),
            node_size=100,
            alpha=0.9,
            ax=ax,
        )
        nx.draw_networkx_edges(subgraph, pos_subgraph, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(
            subgraph, pos_subgraph, font_size=8, font_color="black", ax=ax
        )
        ax.set_title(f"Community {i + 1}", fontsize=12)
        ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pass
