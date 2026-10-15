import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import powerlaw # type: ignore

# === LOAD DATA ===
edges = pd.read_csv('fb-pages-politician-edges.csv', names=['source', 'target'])
nodes = pd.read_csv('fb-pages-politician-nodes.csv')

# === CREATE GRAPH ===
G = nx.Graph()

for _, row in nodes.iterrows():
    G.add_node(row['new_id'], name=row['name'], original_id=row['id'])

for _, row in edges.iterrows():
    G.add_edge(row['source'], row['target'])

# === BASIC NETWORK INFO ===
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")

# === DIAMETER OF LARGEST CONNECTED COMPONENT ===
if nx.is_connected(G):
    diameter = nx.diameter(G)
else:
    largest_component = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_component)
    diameter = nx.diameter(subgraph)
print(f"Diameter (largest connected component): {diameter}")

# === AVERAGE CLUSTERING COEFFICIENT ===
avg_clustering_local = nx.average_clustering(G)
print(f"Average clustering coefficient: {avg_clustering_local:.4f}")

# === AVERAGE SHORTEST PATH LENGTH ===
if nx.is_connected(G):
    avg_path_length = nx.average_shortest_path_length(G)
else:
    subgraph = G.subgraph(max(nx.connected_components(G), key=len))
    avg_path_length = nx.average_shortest_path_length(subgraph)
print(f"Average shortest path length (LCC): {avg_path_length:.4f}")

# === DEGREE DISTRIBUTION ===
degree_histogram = nx.degree_histogram(G)
plt.figure(figsize=(10, 6))
plt.bar(range(len(degree_histogram)), degree_histogram, alpha=0.75, color='skyblue', edgecolor='black')
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# === SCALE-FREE ANALYSIS ===
degrees = [degree for node, degree in G.degree()]
fit = powerlaw.Fit(degrees, discrete=True, verbose=False)

alpha = fit.power_law.alpha
xmin = fit.power_law.xmin
R, p = fit.distribution_compare('power_law', 'exponential')

print(f"\n=== SCALE-FREE ANALYSIS ===")
print(f"Power-law alpha (γ): {alpha:.4f}")
print(f"Power-law xmin: {xmin}")
print(f"Comparison with exponential: R = {R:.4f}, p = {p:.4f}")

if R > 0 and p < 0.05:
    print("The network is likely scale-free (power-law (Our network) is significantly better than exponential).")
else:
    print("The network is not clearly scale-free (power-law (Our network) is not significantly better).")

# === NETWORK DENSITY ===
density = nx.density(G)
print(f"Network density: {density:.6f}")

# === DEGREE ASSORTATIVITY ===
assortativity = nx.degree_assortativity_coefficient(G)
print(f"Degree assortativity coefficient: {assortativity:.4f}")

# === TOP 10 NODES BY DEGREE ===
node_degrees = dict(G.degree())
sorted_by_degree = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
top_n = 30
print(f"\nTop {top_n} nodes by degree:")
for node, degree in sorted_by_degree[:top_n]:
    name = G.nodes[node].get('name', 'Unknown')
    print(f"Node {node} ({name}): Degree = {degree}")

# === TOP 10 NODES BY BETWEENNESS CENTRALITY ===
betweenness = nx.betweenness_centrality(G)
top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:30]
print("\nTop 10 nodes by betweenness centrality:")
for node, score in top_betweenness:
    name = G.nodes[node].get('name', 'Unknown')
    print(f"Node {node} ({name}): Betweenness = {score:.4f}")
