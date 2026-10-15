from matplotlib import pyplot as plt
import networkx as nx
import pandas as pd
import random

# === LOAD DATA ===
edges = pd.read_csv('fb-pages-politician-edges.csv', names=['source', 'target'])
nodes = pd.read_csv('fb-pages-politician-nodes.csv')

# === CREATE GRAPH ===
G = nx.Graph()
for _, row in nodes.iterrows():
    G.add_node(row['new_id'], name=row['name'], original_id=row['id'])
for _, row in edges.iterrows():
    G.add_edge(row['source'], row['target'])

# === EXPERIMENT SETTINGS ===
base_theta = 0.15
base_gossip_starters = 30
base_malicious = 50

experiments = []

# 1. Vary gossip starters
for g in [5, 10, 50, 100, 250, 500, 1000, 3000, 5000, 5800]:
    experiments.append({
        "label": f"vary_gossip_{g}",
        "theta": base_theta,
        "num_gossip_starters": g,
        "num_malicious": base_malicious
    })

# 2. Vary malicious nodes
for m in [5, 10, 50, 100, 250, 500, 1000, 3000, 5000, 5800]:
    experiments.append({
        "label": f"vary_malicious_{m}",
        "theta": base_theta,
        "num_gossip_starters": base_gossip_starters,
        "num_malicious": m
    })

# 3. Vary theta
for t in [0.01, 0.03, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
    experiments.append({
        "label": f"vary_theta_{t:.2f}",
        "theta": t,
        "num_gossip_starters": base_gossip_starters,
        "num_malicious": base_malicious
    })

# === RUN EXPERIMENTS ===
all_results = []

for exp in experiments:
    theta = exp["theta"]
    num_gossip_starters = exp["num_gossip_starters"]
    num_malicious = exp["num_malicious"]
    label = exp["label"]

    print(f"Running: {label}")

    node_status = {node: 0 for node in G.nodes}  # 0 = uninformed, 1 = correct, 2 = infected
    message_origin = {}

    gossip_starters = random.sample(list(G.nodes), min(num_gossip_starters, len(G.nodes)))
    remaining_nodes = list(set(G.nodes) - set(gossip_starters))
    malicious_nodes = set(random.sample(remaining_nodes, min(num_malicious, len(remaining_nodes))))

    for node in gossip_starters:
        node_status[node] = 1
        message_origin[node] = 'correct'
    
    for node in malicious_nodes:
        node_status[node] = 2
        message_origin[node] = 'altered'

    status_log = []

    def count_statuses(step):
        counts = {
            "config": label,
            "step": step,
            "uninformed": sum(1 for s in node_status.values() if s == 0),
            "correct": sum(1 for s in message_origin.values() if s == 'correct'),
            "infected": sum(1 for s in message_origin.values() if s == 'altered')
        }
        status_log.append(counts)

    step = 0
    count_statuses(step)
    step += 1

    while True:
        newly_infected = {}
        for node in G.nodes:
            if node_status[node] != 0:
                continue

            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue

            informed_neighbors = [n for n in neighbors if node_status[n] in [1, 2]]
            if len(informed_neighbors) / len(neighbors) >= theta:
                # Count how many have each type
                correct_count = sum(1 for n in informed_neighbors if message_origin[n] == 'correct')
                altered_count = sum(1 for n in informed_neighbors if message_origin[n] == 'altered')

                if correct_count > altered_count:
                    chosen = 'correct' if random.random() > 0.999999 else 'altered'   #(Extreme 99, high 90, medium 35, low 10)
                elif altered_count > correct_count:
                    chosen = 'altered' if random.random() > 0.999999 else 'correct'
                else:
                    chosen = random.choice(['correct', 'altered'])

                node_status[node] = 1 if chosen == 'correct' else 2

                if node in malicious_nodes:
                    message_origin[node] = 'altered'
                else:
                    message_origin[node] = chosen

                newly_infected[node] = node_status[node]

        if not newly_infected:
            break

        count_statuses(step)
        step += 1

    all_results.extend(status_log)

# === SAVE TO FILE ===
df = pd.DataFrame(all_results)
df.to_csv("optional_part_experiments.csv", index=False)

# === SUMMARY ===
summary = df.groupby("config").agg({
    "step": "max",
    "correct": "max",
    "infected": "max",
    "uninformed": "min"
})
print("\nExperiment Summary:")
print(summary)

# === PLOTTING ===
df_final = df.sort_values("step").groupby("config").last().reset_index()

def parse_variable(row):
    if row["config"].startswith("vary_theta"):
        return "theta", float(row["config"].split("_")[-1])
    elif row["config"].startswith("vary_gossip"):
        return "gossip", int(row["config"].split("_")[-1])
    elif row["config"].startswith("vary_malicious"):
        return "malicious", int(row["config"].split("_")[-1])
    else:
        return "unknown", None

df_final[["variable", "value"]] = df_final.apply(lambda row: pd.Series(parse_variable(row)), axis=1)

def plot_variable(df_final, variable_name):
    subset = df_final[df_final["variable"] == variable_name].sort_values("value")
    
    x_values = subset["value"].values
    correct_values = subset["correct"].values
    infected_values = subset["infected"].values
    uninformed_values = subset["uninformed"].values

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, correct_values, label="Correct", marker="o")
    plt.plot(x_values, infected_values, label="Infected", marker="s")
    plt.plot(x_values, uninformed_values, label="Uninformed", marker="^")

    plt.xlabel(variable_name.capitalize())
    plt.ylabel("Number of Nodes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = f"optional_part_{variable_name}_extreme_trashold_plot.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")

for var in ["theta", "gossip", "malicious"]:
    plot_variable(df_final, var)