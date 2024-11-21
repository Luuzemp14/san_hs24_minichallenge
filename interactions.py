import plots
import utils
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os

interactions: dict = utils.load_json("data/starwars-full-interactions.json")
G_interactions = nx.Graph(name="Interactions")
G_interactions = utils.get_graph_with_nodes_and_edges(G_interactions, interactions)
communities = list(nx.community.girvan_newman(G_interactions))
modularity_df = pd.DataFrame(
    [
        [k + 1, nx.community.modularity(G_interactions, communities[k])]
        for k in range(len(communities))
    ],
    columns=["k", "modularity"],
)

fig, ax = plt.subplots(3, figsize=(15, 20))
plots.visualize_communities(G_interactions, communities[0], 1)
modularity_df.plot.bar(
    x="k",
    ax=ax[2],
    color="#F2D140",
    title="Modularity Trend for Girvan-Newman Community Detection",
)
plt.show()