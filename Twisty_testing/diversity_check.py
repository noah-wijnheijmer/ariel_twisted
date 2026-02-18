from morphological_descriptor import compute_6d_descriptor
from networkx.readwrite import json_graph
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import mannwhitneyu
generations = [i+1 for i in range(20)]
all_reps_mixed = []
all_reps_non_twisty = []
all_reps_twisty = []
for j in range(30):
    diversity_mixed = []
    diversity_non_twisty = []
    diversity_twisty = []
    for i in range(20):
        with open(fr"evo_diversity\population_graphs\generation_{i}\rep{j}_mixed", 'r', encoding="utf-8") as file:
            mixed_pop = json.load(file)
        with open(fr"evo_diversity\population_graphs\generation_{i}\rep{j}_non-twisty", 'r', encoding="utf-8") as file:
            non_twisty_pop = json.load(file)
        with open(fr"evo_diversity\population_graphs\generation_{i}\rep{j}_twisted", 'r', encoding="utf-8") as file:
            twisted_pop = json.load(file)
        mixed = []
        for data in mixed_pop:
            i = json_graph.node_link_graph(
            data,
            directed=True,
            multigraph=False,
            edges="edges",
            )
            mixed.append(compute_6d_descriptor(i))
        non_twisty = []
        for data in non_twisty_pop:
            i = json_graph.node_link_graph(
            data,
            directed=True,
            multigraph=False,
            edges="edges",
            )
            non_twisty.append(compute_6d_descriptor(i))
        twisty = []
        for data in twisted_pop:
            i = json_graph.node_link_graph(
            data,
            directed=True,
            multigraph=False,
            edges="edges",
            )
            twisty.append(compute_6d_descriptor(i))
        # mixed
        descriptors_matrix = np.array(mixed)
        centroid = np.mean(descriptors_matrix, axis=0)
        distances = np.linalg.norm(descriptors_matrix - centroid, axis=1)
        diversity_score = np.mean(distances)
        diversity_mixed.append(diversity_score)
        # non_twisty
        descriptors_matrix = np.array(non_twisty)
        centroid = np.mean(descriptors_matrix, axis=0)
        distances = np.linalg.norm(descriptors_matrix - centroid, axis=1)
        diversity_score = np.mean(distances)
        diversity_non_twisty.append(diversity_score)
        # twisty
        descriptors_matrix = np.array(twisty)
        centroid = np.mean(descriptors_matrix, axis=0)
        distances = np.linalg.norm(descriptors_matrix - centroid, axis=1)
        diversity_score = np.mean(distances)
        diversity_twisty.append(diversity_score)
    all_reps_mixed.append(diversity_mixed)
    all_reps_non_twisty.append(diversity_non_twisty)
    all_reps_twisty.append(diversity_twisty)
# mixed
data_matrix_m = np.array(all_reps_mixed)
mean_diversity_mixed = np.mean(data_matrix_m, axis=0)
std_diversity = np.std(data_matrix_m, axis=0)
sems_m = std_diversity / np.sqrt(30)
confidence_margin_m = 1.96 * sems_m
lower_bound_mixed = mean_diversity_mixed - confidence_margin_m
upper_bound_mixed = mean_diversity_mixed + confidence_margin_m
# non_twisty
data_matrix_n = np.array(all_reps_non_twisty)
mean_diversity_non_twisty = np.mean(data_matrix_n, axis=0)
std_diversity = np.std(data_matrix_n, axis=0)
sems_n = std_diversity / np.sqrt(30)
confidence_margin_n = 1.96 * sems_n
lower_bound_non_twisty = mean_diversity_non_twisty - confidence_margin_n
upper_bound_non_twisty = mean_diversity_non_twisty + confidence_margin_n
# twisty
data_matrix_t = np.array(all_reps_twisty)
mean_diversity_twisty = np.mean(data_matrix_t, axis=0)
std_diversity = np.std(data_matrix_t, axis=0)
sems_t = std_diversity / np.sqrt(30)
confidence_margin_t = 1.96 * sems_t
lower_bound_twisty = mean_diversity_twisty - confidence_margin_t
upper_bound_twisty = mean_diversity_twisty + confidence_margin_t

plt.figure(figsize=(10, 6))
plt.plot(generations, mean_diversity_mixed, marker="o", markersize=4, label="Mixed", linewidth=2, color='#1f77b4')
plt.plot(generations, mean_diversity_non_twisty, marker="o", markersize=4, label="Non_twisty", linewidth=2, color="#1fb43f")
plt.fill_between(generations, lower_bound_mixed, upper_bound_mixed, color='#1f77b4', alpha=0.2, linewidth=0)
plt.fill_between(generations, lower_bound_non_twisty, upper_bound_non_twisty, color="#1fb43f", alpha=0.2, linewidth=0)
for i in range(len(generations)):
    gen_untwisted = data_matrix_n[:, i]
    gen_mixed = data_matrix_m[:, i]
    p_val = mannwhitneyu(gen_untwisted, gen_mixed)
    marker = None
    if p_val.pvalue < 0.001: marker = "***"
    elif p_val.pvalue < 0.01: marker = "**"
    elif p_val.pvalue < 0.05: marker = "*"
    if marker:
        high_point = max(mean_diversity_non_twisty[i] + confidence_margin_n[i], mean_diversity_mixed[i] + confidence_margin_m[i])
        plt.text(generations[i], high_point + 0.001, marker, 
                ha='center', va='bottom', fontsize=12)
plt.xlabel('Generation')
plt.ylabel('Diversity score')
plt.title('Diversity over Generations - Mixed vs Non_twisty')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(0.0, 1.0)
plt.savefig(f"MN_diversity.png")

plt.figure(figsize=(10, 6))
plt.plot(generations, mean_diversity_mixed, marker="o", markersize=4, label="Mixed", linewidth=2, color='#1f77b4')
plt.plot(generations, mean_diversity_twisty, marker="o", markersize=4, label="Twisty", linewidth=2, color="#b41f1f")
plt.fill_between(generations, lower_bound_mixed, upper_bound_mixed, color='#1f77b4', alpha=0.2, linewidth=0)
plt.fill_between(generations, lower_bound_twisty, upper_bound_twisty, color="#b41f1f", alpha=0.2, linewidth=0)
for i in range(len(generations)):
    gen_twisted = data_matrix_t[:, i]
    gen_mixed = data_matrix_m[:, i]
    p_val = mannwhitneyu(gen_twisted, gen_mixed)
    marker = None
    if p_val.pvalue < 0.001: marker = "***"
    elif p_val.pvalue < 0.01: marker = "**"
    elif p_val.pvalue < 0.05: marker = "*"
    if marker:
        high_point = max(mean_diversity_twisty[i] + confidence_margin_t[i], mean_diversity_mixed[i] + confidence_margin_m[i])
        plt.text(generations[i], high_point + 0.001, marker, 
                ha='center', va='bottom', fontsize=12)
plt.xlabel('Generation')
plt.ylabel('Diversity score')
plt.title('Diversity over Generations - Mixed vs Twisty')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(0.0, 1.0)
plt.savefig(f"MT_diversity.png")

plt.figure(figsize=(10, 6))
plt.plot(generations, mean_diversity_non_twisty, marker="o", markersize=4, label="Non_twisty", linewidth=2, color="#1fb43f")
plt.plot(generations, mean_diversity_twisty, marker="o", markersize=4, label="Twisty", linewidth=2, color="#b41f1f")
plt.fill_between(generations, lower_bound_non_twisty, upper_bound_non_twisty, color="#1fb43f", alpha=0.2, linewidth=0)
plt.fill_between(generations, lower_bound_twisty, upper_bound_twisty, color="#b41f1f", alpha=0.2, linewidth=0)
for i in range(len(generations)):
    gen_twisted = data_matrix_t[:, i]
    gen_untwisted = data_matrix_n[:, i]
    p_val = mannwhitneyu(gen_twisted, gen_untwisted)
    marker = None
    if p_val.pvalue < 0.001: marker = "***"
    elif p_val.pvalue < 0.01: marker = "**"
    elif p_val.pvalue < 0.05: marker = "*"
    if marker:
        high_point = max(mean_diversity_twisty[i] + confidence_margin_t[i], mean_diversity_non_twisty[i] + confidence_margin_n[i])
        plt.text(generations[i], high_point + 0.001, marker, 
                ha='center', va='bottom', fontsize=12)
plt.xlabel('Generation')
plt.ylabel('Diversity score')
plt.title('Diversity over Generations - Non_twisty vs Twisty')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(0.0, 1.0)
plt.savefig(f"NT_diversity.png")