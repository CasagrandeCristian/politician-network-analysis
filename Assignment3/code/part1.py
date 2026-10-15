import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd

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

# === CREATE GRAPHS ===
G1 = nx.erdos_renyi_graph(n=500, p=0.05, seed=42)
G2 = nx.barabasi_albert_graph(n=500, m=3, seed=42)

# === DEFINE ATTACK STRATEGIES ===
strategies = ['random', 'degree', 'pagerank', 'betweenness']

# === RUN EXPERIMENTS ===
percentage = 0.05  # 5% of the giant component per iteration
results_G1 = {}
results_G2 = {}
threshold_data = []

for strategy in strategies:
    sizes1, x1, rem1 = simulate_attack(G1, strategy, percentage=percentage)
    sizes2, x2, rem2 = simulate_attack(G2, strategy, percentage=percentage)
    results_G1[strategy] = (sizes1, x1, rem1)
    results_G2[strategy] = (sizes2, x2, rem2)

    step1 = next((i for i, v in enumerate(sizes1) if v < 0.05), -1)
    step2 = next((i for i, v in enumerate(sizes2) if v < 0.05), -1)

    threshold_data.append({
        'strategy': strategy,
        'graph': 'erdos_renyi',
        'step_to_below_0.05': step1,
        'percentage_removed_from_gc': x1[step1] * 100 if step1 != -1 else None
    })
    threshold_data.append({
        'strategy': strategy,
        'graph': 'barabasi_albert',
        'step_to_below_0.05': step2,
        'percentage_removed_from_gc': x2[step2] * 100 if step2 != -1 else None
    })

# === SAVE PLOTS ===
plot_results(results_G1, "Attacks on Erdős‑Rényi Graph (n=500, 5% removal)", filename="erdos_gc_removal.png")
plot_results(results_G2, "Attacks on Barabási‑Albert Graph (n=500, 5% removal)", filename="barabasi_gc_removal.png")

# === SAVE THRESHOLD DATA ===
df = pd.DataFrame(threshold_data)
df.to_csv("gc_threshold_steps.csv", index=False)
print("Saved threshold steps to gc_threshold_steps.csv")
