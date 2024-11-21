#!/usr/bin/env python
# coding: utf-8

# The data consists of:
# 
# **For each episode**
# - Interactions
#   - times the characters speak within the same scene
# - Mentions
#   - times the characters are mentioned within the same scene
# - All
#   - same as interactions but added with r2d2 and chewbacca
# 
# **Full**
# - All data from 6 episodes

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import json
import networkx as nx
import nxviz as nv
from nxviz import annotate
from nxviz import nodes
from nxviz import edges
from nxviz.plots import aspect_equal, despine

import matplotlib.pyplot as plt

from pyvis.network import Network

import importlib
import utils
import centralities
import communities
importlib.reload(utils)

interactions: dict = utils.load_json("data/starwars-full-interactions.json")
mentions: dict = utils.load_json("data/starwars-full-mentions.json")
all_interactions: dict = utils.load_json(
    "data/starwars-full-interactions-allCharacters.json"
)

G_interactions = nx.Graph(name="Interactions")
G_mentions = nx.Graph(name="Mentions")

G_interactions = utils.get_graph_with_nodes_and_edges(G_interactions, interactions)
G_mentions = utils.get_graph_with_nodes_and_edges(G_mentions, mentions)

utils.get_network_statistics(G_interactions)
utils.get_network_statistics(G_mentions)

centralities.get_actor_centralities(
    G_interactions, ["degree", "betweenness", "eigenvector"]
)
centralities.get_actor_centralities(
    G_mentions, ["degree", "betweenness", "eigenvector"]
)

try:
    communities_interactions = communities.detect_communities(
        G_interactions, "Interactions Network"
    )
    communities_mentions = communities.detect_communities(G_mentions, "Mentions Network")
except Exception as e:
    print(f"Error in community detection: {str(e)}")

import plots

plots.plot_networks(G_interactions, G_mentions)
plots.plot_strength_distribution(G_interactions, "Interactions Network")
plots.plot_strength_distribution(G_mentions, "Mentions Network")
plots.plot_communities(G_interactions, communities_interactions, "Interactions Network")
plots.plot_communities(G_mentions, communities_mentions, "Mentions Network")




