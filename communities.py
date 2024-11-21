import networkx as nx


def detect_communities(G: nx.Graph, network_name: str):
    try:
        communities = nx.community.louvain_communities(G, seed=42)
        print(f"\n{network_name} Communities:")
        for i, community in enumerate(communities, 1):
            if community:
                print(f"\nCommunity {i}:")
                print(", ".join(sorted(community)))
        return communities
    except Exception as e:
        print(f"Error detecting communities for {network_name}: {str(e)}")
        return []
