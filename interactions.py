import utils
import networkx as nx
from constants import LAYOUTS, COMMUNITY_DETECTIONS
from plots import plot_node_size, plot_edge_weight, plot_community_with_layout
import matplotlib.pyplot as plt
import os

interactions: dict = utils.load_json("data/starwars-full-interactions.json")
G_interactions = nx.Graph(name="Interactions")
G_interactions = utils.get_graph_with_nodes_and_edges(G_interactions, interactions)

# List of centrality measures to use for node sizes
centrality_measures = ["value", "degree", "closeness", "betweenness", "eigenvector"]

for layout in LAYOUTS:
    for size_by in centrality_measures:
        if size_by == "betweenness":
            print(f"Plotting {layout} with node size by {size_by}...")
            plot_node_size(G_interactions, layout, size_by=size_by)

# for community_type in COMMUNITY_DETECTIONS:
#     print(f"Detecting communities using {community_type}...")
#     community_func = getattr(nx.community, community_type)

#     if community_type in ['edge_betweenness_partition', 'edge_current_flow_betweenness_partition']:
#         communities = list(community_func(G_interactions, number_of_sets=2))
#     elif community_type == 'k_clique_communities':
#         communities = list(community_func(G_interactions, k=3))
#     elif community_type == 'lukes_partitioning':
#         communities = list(community_func(G_interactions, max_size=10))
#     else:
#         communities = list(community_func(G_interactions))

#     for layout in LAYOUTS:
#         for size_by in centrality_measures:
#             print(f"Plotting {community_type} with {layout} (size by {size_by})...")
#             plot_community_with_layout(G_interactions, layout, communities, community_type, size_by=size_by)

