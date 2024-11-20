import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import numpy as np

def plot_networks(G_interactions, G_mentions):
    """Plot interactions and mentions networks side by side with improved layout"""
    plt.figure(figsize=(20, 10))

    # Common layout parameters
    k = 2/np.sqrt(len(G_interactions.nodes()))  # Optimal distance between nodes
    iterations = 100  # More iterations for better convergence

    # Plot interactions network
    plt.subplot(121)
    # Use spring_layout with optimized parameters
    pos_interactions = nx.spring_layout(
        G_interactions,
        k=k,
        iterations=iterations,
        seed=42,
        weight='weight'  # Consider edge weights in layout
    )
    
    # Draw edges first (so they appear behind nodes)
    nx.draw_networkx_edges(
        G_interactions,
        pos_interactions,
        width=[G_interactions[u][v]["weight"] / 5 for u, v in G_interactions.edges()],
        alpha=0.3  # Make edges more transparent
    )
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G_interactions,
        pos_interactions,
        node_size=[G_interactions.nodes[node]["value"] * 50 for node in G_interactions.nodes],
        node_color=[G_interactions.nodes[node]["colour"] for node in G_interactions.nodes],
        alpha=0.7
    )
    
    # Add labels with better positioning
    labels = nx.draw_networkx_labels(
        G_interactions,
        pos_interactions,
        font_size=8,
        font_weight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
    )
    
    plt.title("Episode 1 Interactions Network", pad=20)

    # Plot mentions network
    plt.subplot(122)
    pos_mentions = nx.spring_layout(
        G_mentions,
        k=k,
        iterations=iterations,
        seed=42,
        weight='weight'
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G_mentions,
        pos_mentions,
        width=[G_mentions[u][v]["weight"] / 5 for u, v in G_mentions.edges()],
        alpha=0.3
    )
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G_mentions,
        pos_mentions,
        node_size=[G_mentions.nodes[node]["value"] * 50 for node in G_mentions.nodes],
        node_color=[G_mentions.nodes[node]["colour"] for node in G_mentions.nodes],
        alpha=0.7
    )
    
    # Add labels
    labels = nx.draw_networkx_labels(
        G_mentions,
        pos_mentions,
        font_size=8,
        font_weight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
    )
    
    plt.title("Episode 1 Mentions Network", pad=20)

    # Adjust layout
    plt.tight_layout(pad=3.0)
    
    # Remove axes for both subplots
    for ax in plt.gcf().get_axes():
        ax.set_axis_off()
    
    plt.show()

def plot_strength_distribution(G, network_name):
    """Plot node strength distribution histogram"""
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
    """Plot network with nodes colored by community membership"""
    plt.figure(figsize=(12, 8))
    
    # Create a color map for communities
    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(communities) > len(colors):
        colors = plt.cm.tab20(np.linspace(0, 1, len(communities)))
        
    # Create a mapping of nodes to colors based on community membership
    color_map = {}
    for i, community in enumerate(communities):
        for node in community:
            color_map[node] = colors[i % len(colors)]
    
    # Create layout
    pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos,
                          node_size=[G.nodes[node]["value"] * 50 for node in G.nodes],
                          node_color=[color_map[node] for node in G.nodes],
                          alpha=0.7)
    
    nx.draw_networkx_edges(G, pos,
                          width=[G[u][v]["weight"]/5 for u,v in G.edges()],
                          alpha=0.5)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f"Communities in {network_name}")
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=colors[i % len(colors)],
                                 label=f'Community {i+1}', markersize=10)
                      for i in range(len(communities))]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.show()
  