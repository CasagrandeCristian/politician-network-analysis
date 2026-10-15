import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import time

start_time = time.time()

# === Load data ===
edges = pd.read_csv('fb-pages-politician-edges.csv', names=['source', 'target'])
nodes = pd.read_csv('fb-pages-politician-nodes.csv')

G_base = nx.Graph()
for _, row in nodes.iterrows():
    G_base.add_node(row['new_id'], name=row['name'], original_id=row['id'])
for _, row in edges.iterrows():
    G_base.add_edge(row['source'], row['target'])

# === Initial stats ===
initial_edges = G_base.number_of_edges()
initial_avg_degree = sum(dict(G_base.degree()).values()) / G_base.number_of_nodes()

# === Reinforce: add edges to low-degree nodes in giant component ===
def reinforce_low_deg_giant(G, bottom_percent=0.1, reinforcement_factor=0.03, seed=42):
    random.seed(seed)
    gc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    sorted_deg = sorted(gc.degree(), key=lambda x: x[1])
    cutoff = int(len(sorted_deg) * bottom_percent)
    low_deg = [n for n, _ in sorted_deg[:cutoff]]
    all_gc_nodes = list(gc.nodes())
    G_reinf = G.copy()

    for node in low_deg:
        add_count = max(1, int(len(all_gc_nodes) * reinforcement_factor))
        potentials = [n for n in all_gc_nodes if n != node and not G_reinf.has_edge(node, n)]
        new_neighbors = random.sample(potentials, min(add_count, len(potentials)))
        for v in new_neighbors:
            G_reinf.add_edge(node, v)

    return G_reinf

# === Compute f_c ===
def compute_fc(G):
    degrees = [d for _, d in G.degree()]
    k_avg = np.mean(degrees)
    k2_avg = np.mean([d * d for d in degrees])
    if k2_avg == k_avg:
        return 0.0
    return 1 - (1 / ((k2_avg / k_avg) - 1))

# === Simulate attack ===
def simulate_attack(G, strategy='random', percentage=0.05):
    H = G.copy()
    N = H.number_of_nodes()
    sizes = []
    removed_ratios = []
    removed_total = 0
    threshold = 0.05

    while True:
        components = list(nx.connected_components(H))
        if not components:
            break
        largest_cc_nodes = max(components, key=len)
        size = len(largest_cc_nodes)

        sizes.append(size / N)
        removed_ratios.append(removed_total / N)

        if size / N < threshold:
            break

        remove_count = max(1, int(size * percentage))
        largest_cc = H.subgraph(largest_cc_nodes).copy()

        if strategy == 'random':
            to_remove = random.sample(list(largest_cc.nodes()), remove_count)
        elif strategy == 'degree':
            degree_list = sorted(largest_cc.degree(), key=lambda x: x[1], reverse=True)
            to_remove = [n for n, _ in degree_list[:remove_count]]
        elif strategy == 'pagerank':
            pr = nx.pagerank(largest_cc)
            pr_sorted = sorted(pr.items(), key=lambda x: x[1], reverse=True)
            to_remove = [n for n, _ in pr_sorted[:remove_count]]
        elif strategy == 'betweenness':
            bw = nx.betweenness_centrality(largest_cc)
            bw_sorted = sorted(bw.items(), key=lambda x: x[1], reverse=True)
            to_remove = [n for n, _ in bw_sorted[:remove_count]]
        else:
            raise ValueError("Invalid strategy")

        H.remove_nodes_from(to_remove)
        removed_total += remove_count

    return sizes, removed_ratios, removed_total

# === Plot results ===
def plot_results(results, title, filename=None):
    plt.figure(figsize=(10, 6))
    for strat, (g_over_n, removed_ratios, _) in results.items():
        plt.plot(np.array(removed_ratios)*100, g_over_n, label=strat)
    plt.xlabel("Percentage of nodes removed from giant component (%)")
    plt.ylabel("Giant component size / Total nodes (G/N)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if filename:
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Plot saved as {filename}")
    else:
        plt.show()

# === MAIN ===
percentage = 0.05
strategies = ['random', 'degree', 'pagerank', 'betweenness']

fc_before = compute_fc(G_base)
G_reinf = reinforce_low_deg_giant(G_base, bottom_percent=0.1, reinforcement_factor=0.03)
fc_after = compute_fc(G_reinf)

final_edges = G_reinf.number_of_edges()
final_avg_degree = sum(dict(G_reinf.degree()).values()) / G_reinf.number_of_nodes()

results = {}
csv_rows = []

for strat in strategies:
    print(f"Running strategy: {strat}")
    start = time.time()
    sizes, removed_ratios, removed_total = simulate_attack(G_reinf, strat, percentage)
    end = time.time()

    results[strat] = (sizes, removed_ratios, removed_total)

    step = next((i for i, v in enumerate(sizes) if v < 0.05), -1)
    csv_rows.append({
        'graph': 'reinforced',
        'strategy': strat,
        'step_to_below_0.05': step,
        'percent_removed': removed_ratios[step] * 100 if step != -1 else None,
        'fc_before': fc_before,
        'fc_after': fc_after,
        'initial_edges': initial_edges,
        'final_edges': final_edges,
        'initial_avg_degree': initial_avg_degree,
        'final_avg_degree': final_avg_degree,
        'execution_time_seconds': round(end - start, 2)
    })

execution_time = round(time.time() - start_time, 2)

summary_rows = [
    {
        'graph': 'summary',
        'strategy': 'fc_before',
        'step_to_below_0.05': None,
        'percent_removed': fc_before * 100,
        'fc_before': fc_before,
        'fc_after': None,
        'initial_edges': None,
        'final_edges': None,
        'initial_avg_degree': None,
        'final_avg_degree': None,
        'execution_time_seconds': None
    },
    {
        'graph': 'summary',
        'strategy': 'fc_after',
        'step_to_below_0.05': None,
        'percent_removed': fc_after * 100,
        'fc_before': None,
        'fc_after': fc_after,
        'initial_edges': None,
        'final_edges': None,
        'initial_avg_degree': None,
        'final_avg_degree': None,
        'execution_time_seconds': None
    },
    {
        'graph': 'summary',
        'strategy': 'graph_stats',
        'step_to_below_0.05': None,
        'percent_removed': None,
        'fc_before': fc_before,
        'fc_after': fc_after,
        'initial_edges': initial_edges,
        'final_edges': final_edges,
        'initial_avg_degree': initial_avg_degree,
        'final_avg_degree': final_avg_degree,
        'execution_time_seconds': execution_time
    }
]

pd.DataFrame(csv_rows + summary_rows).to_csv("fb_low_degree_reinforced_summary.csv", index=False)
print("CSV saved to fb_low_degree_reinforced_summary.csv")

plot_results(results, "Attacks on Low-Degree Reinforced Facebook Graph", filename="fb_low_degree_attack_plot.png")

print(f"Total execution time: {execution_time} seconds")
