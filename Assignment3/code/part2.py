import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import time

# === Load graph from CSV files ===
edges = pd.read_csv('fb-pages-politician-edges.csv', names=['source', 'target'])
nodes = pd.read_csv('fb-pages-politician-nodes.csv')

G = nx.Graph()
for _, row in nodes.iterrows():
    G.add_node(row['new_id'], name=row['name'], original_id=row['id'])
for _, row in edges.iterrows():
    G.add_edge(row['source'], row['target'])

# === FUNCTION: Compute the size of the giant component ===
def giant_component_size(G):
    if len(G) == 0:
        return 0
    giant = max(nx.connected_components(G), key=len)
    return len(giant)

# === FUNCTION: Simulate attacks using a given strategy and removal percentage per iteration ===
def simulate_attack(graph, strategy='random', percentage=0.05):
    G = graph.copy()
    N = G.number_of_nodes()
    sizes = []             # G/N at each step
    removed_ratios = []    # cumulative nodes removed / N
    removed_total = 0
    threshold = 0.05       # Stop when G/N < 0.05

    while True:
        components = list(nx.connected_components(G))
        if not components:
            break
        largest_cc_nodes = max(components, key=len)
        size = len(largest_cc_nodes)

        sizes.append(size / N)
        removed_ratios.append(removed_total / N)

        if size / N < threshold:
            break

        remove_count = max(1, int(size * percentage))
        largest_cc = G.subgraph(largest_cc_nodes).copy()

        if strategy == 'random':
            to_remove = random.sample(list(largest_cc.nodes()), remove_count)
        elif strategy == 'degree':
            degree_list = sorted(largest_cc.degree, key=lambda x: x[1], reverse=True)
            to_remove = [node for node, _ in degree_list[:remove_count]]
        elif strategy == 'pagerank':
            pr = nx.pagerank(largest_cc)
            pr_sorted = sorted(pr.items(), key=lambda x: x[1], reverse=True)
            to_remove = [node for node, _ in pr_sorted[:remove_count]]
        elif strategy == 'betweenness':
            bw = nx.betweenness_centrality(largest_cc)
            bw_sorted = sorted(bw.items(), key=lambda x: x[1], reverse=True)
            to_remove = [node for node, _ in bw_sorted[:remove_count]]
        else:
            raise ValueError("Invalid strategy")

        G.remove_nodes_from(to_remove)
        removed_total += remove_count

    return sizes, removed_ratios, removed_total

# === FUNCTION: Plot results (x = cumulative % nodes removed from giant component) ===
def plot_results(results, title, filename=None):
    plt.figure(figsize=(10, 6))
    for strategy, (g_over_n, removed_ratios, _) in results.items():
        plt.plot(np.array(removed_ratios) * 100, g_over_n, label=strategy)
    plt.title(title)
    plt.xlabel("Percentage of nodes removed from giant component until G/N < 0.05")
    plt.ylabel("Giant component size / Total nodes (G/N)")
    plt.legend()
    plt.grid(True)
    if filename:
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved plot as: {filename}")
    else:
        plt.show()

# === RUN EXPERIMENTS ===
strategies = ['random', 'degree', 'pagerank', 'betweenness']
percentage = 0.05  # 5% of giant component per iteration
results = {}
threshold_data = []

for strategy in strategies:
    print(f"Running strategy: {strategy}")
    start = time.time()
    sizes, removed_ratios, removed_total = simulate_attack(G, strategy, percentage=percentage)
    end = time.time()
    results[strategy] = (sizes, removed_ratios, removed_total)

    step = next((i for i, v in enumerate(sizes) if v < 0.05), -1)
    threshold_data.append({
        'strategy': strategy,
        'step_to_below_0.05': step,
        'percentage_removed_from_gc': removed_ratios[step] * 100 if step != -1 else None,
        'execution_time_seconds': round(end - start, 2)
    })

# === PLOT RESULTS ===
plot_results(results, f"Attacks on Facebook Politician Graph (n={G.number_of_nodes()}, 5% removal)", filename="fb_gc_removal.png")

# === SAVE THRESHOLD DATA ===
df = pd.DataFrame(threshold_data)
df.to_csv("fb_gc_threshold_steps.csv", index=False)
print("Saved threshold steps to fb_gc_threshold_steps.csv")
