import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import time
from networkx.algorithms.community import louvain_communities, greedy_modularity_communities, label_propagation_communities
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score # type: ignore

# === PART 2 ===


# === LOAD DATA ===
edges = pd.read_csv('fb-pages-politician-edges.csv', names=['source', 'target'])
nodes = pd.read_csv('fb-pages-politician-nodes.csv')

# === CREATE NETWORKX GRAPH ===
G_nx = nx.Graph()

for _, row in nodes.iterrows():
    G_nx.add_node(row['new_id'], name=row['name'], original_id=row['id'])

for _, row in edges.iterrows():
    G_nx.add_edge(row['source'], row['target'])

# === RUN LOUVAIN ALGORITHM ===
print("\n--- LOUVAIN ---\n")
start_time = time.time()
louvain_partition = louvain_communities(G_nx, seed=42)
end_time = time.time()

print(f"Number of communities found (Louvain): {len(louvain_partition)} \n")

# === PRINT LOUVAIN COMMUNITIES (FIRST 10 NODES) ===
for i, community in enumerate(louvain_partition):
    names = [G_nx.nodes[node]['name'] for node in sorted(community)]
    subgraph = G_nx.subgraph(community)
    print(f"Community {i} ({len(names)} nodes): {names[:10]} - Connected: {nx.is_connected(subgraph)}\n")

print(f"Louvain execution time: {end_time - start_time:.4f} seconds\n")

# === RUN GREEDY MODULARITY ALGORITHM ===
print("\n--- GREEDY MODULARITY ---\n")
start_time = time.time()
greedy_partition = greedy_modularity_communities(G_nx)
end_time = time.time()

print(f"Number of communities found (Greedy Modularity): {len(greedy_partition)} \n")

# === PRINT GREEDY COMMUNITIES (FIRST 10 NODES) ===
for i, community in enumerate(greedy_partition):
    names = [G_nx.nodes[node]['name'] for node in sorted(community)]
    subgraph = G_nx.subgraph(community)
    print(f"Community {i} ({len(names)} nodes): {names[:10]} - Connected: {nx.is_connected(subgraph)}\n")

print(f"Greedy Modularity execution time: {end_time - start_time:.4f} seconds\n")

# === RUN LABEL PROPAGATION ALGORITHM ===
print("\n--- LABEL PROPAGATION ---\n")
start_time = time.time()
label_partition = list(label_propagation_communities(G_nx))
end_time = time.time()

print(f"Number of communities found (Label Propagation): {len(label_partition)} \n")

# === PRINT LABEL PROPAGATION COMMUNITIES (FIRST 10 NODES) ===
for i, community in enumerate(label_partition):
    names = [G_nx.nodes[node]['name'] for node in sorted(community)]
    subgraph = G_nx.subgraph(community)
    print(f"Community {i} ({len(names)} nodes): {names[:10]} - Connected: {nx.is_connected(subgraph)}\n")

print(f"Label Propagation execution time: {end_time - start_time:.4f} seconds")


# === PART 3 ===

def partition_to_labels(partition, nodes_list):
    """
    Convert a partition (list of sets of nodes) into a label list ordered according to nodes_list.
    """
    label_dict = {}
    for label, community in enumerate(partition):
        for node in community:
            label_dict[node] = label
    return [label_dict[node] for node in nodes_list]



# === LIST OF LABELS ===
nodes_list = sorted(G_nx.nodes())
labels_louvain = partition_to_labels(louvain_partition, nodes_list)
labels_greedy = partition_to_labels(greedy_partition, nodes_list)
labels_labelprop = partition_to_labels(label_partition, nodes_list)

# === ALL PAIRWISE ===
comparisons = [
    ("Louvain", "Greedy", labels_louvain, labels_greedy),
    ("Louvain", "Label Propagation", labels_louvain, labels_labelprop),
    ("Greedy", "Label Propagation", labels_greedy, labels_labelprop),
]

# === NMI and ARI  ===
for name1, name2, labels1, labels2 in comparisons:
    nmi = normalized_mutual_info_score(labels1, labels2)
    ari = adjusted_rand_score(labels1, labels2)
    print(f"{name1} vs {name2}")
    print(f"  NMI: {nmi:.4f}")
    print(f"  ARI: {ari:.4f}\n")


# === PART 4 ===

def draw_community_graph(G, partition, title):
    """
    Draws the graph G with node colors based on the given partition.
    Each community will be assigned a unique color.
    """
    pos = nx.spring_layout(G, seed=42, k = 0.1)  
    cmap = plt.get_cmap("tab20")        

    for i, community in enumerate(partition):
        color = cmap(i % 20)  
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(community),
            node_color=[color],
            label=f"Community {i}",
            node_size=30,
            alpha=0.8
        )

    nx.draw_networkx_edges(G, pos, alpha=0.2)
    plt.title(title)
    plt.axis('off')

# === Draw Louvain ===
plt.figure(figsize=(8, 6))
draw_community_graph(G_nx, louvain_partition, "Louvain Community Detection")
plt.show()

# === Draw Greedy Modularity ===
plt.figure(figsize=(8, 6))
draw_community_graph(G_nx, greedy_partition, "Greedy Modularity Community Detection")
plt.show()

# === Draw Label Propagation ===
plt.figure(figsize=(8, 6))
draw_community_graph(G_nx, label_partition, "Label Propagation Community Detection")
plt.show()
