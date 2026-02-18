import json
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import numpy as np
experiment_mixed = []
experiment_twisted = []
experiment_untwisted = []
for i in range(30):
    try:
        with open(f"evo_main/experiment_data_non-twisty_champion{i+210}.json", 'r') as file:
            experiment_mixed.append(json.load(file))
    except:
        with open(f"evo_main/experiment_data_twisty_champion{i+210}.json", 'r') as file:
            experiment_mixed.append(json.load(file))
    try:
        with open(f"evo_main/experiment_data_non-twisty_champion{i+240}.json", 'r') as file:
            experiment_twisted.append(json.load(file))
    except:
        with open(f"evo_main/experiment_data_twisty_champion{i+240}.json", 'r') as file:
            experiment_twisted.append(json.load(file))
    try:
        with open(f"evo_main/experiment_data_non-twisty_champion{i+180}.json", 'r') as file:
            experiment_untwisted.append(json.load(file))
    except:
        with open(f"evo_main/experiment_data_twisty_champion{i+180}.json", 'r') as file:
            experiment_untwisted.append(json.load(file))
# print(len(experiment_untwisted))
generations = []
i = 0
max_mixed_repetition = []
for experiment_data in experiment_mixed:
        maxs = []
        for gen_data in experiment_data["generations"]:
            if i == 0:
                generations.append(gen_data["generation"])

            if "mixed2_amphi" in gen_data:
                stats = gen_data["mixed2_amphi"]
                maxs.append(stats["max"])
        max_mixed_repetition.append(maxs)
        i += 1
max_mixed_repetition = np.array(max_mixed_repetition)
means_mixed = np.mean(max_mixed_repetition, axis=0)
stds_mixed = np.std(max_mixed_repetition, axis=0)
sems_m = stds_mixed / np.sqrt(30)
confidence_margin_m = 1.96 * sems_m
lower_bound_mixed = means_mixed - confidence_margin_m
upper_bound_mixed = means_mixed + confidence_margin_m

generations = []
i = 0
max_twisted_repetitions = []
for experiment_data in experiment_twisted:
        maxs = []
        for gen_data in experiment_data["generations"]:
            if i == 0:
                generations.append(gen_data["generation"])

            if "twisted2_amphi" in gen_data:
                stats = gen_data["twisted2_amphi"]
                maxs.append(stats["max"])
        max_twisted_repetitions.append(maxs)
        i += 1
max_twisted_repetitions = np.array(max_twisted_repetitions)
means_twisted = np.mean(max_twisted_repetitions, axis=0)
stds_twisted = np.std(max_twisted_repetitions, axis=0)
sems_t = stds_twisted / np.sqrt(30)
confidence_margin_t = 1.96 * sems_t
lower_bound_twisted = means_twisted - confidence_margin_t
upper_bound_twisted = means_twisted + confidence_margin_t

generations = []
i = 0
max_untwisted_repetitions = []
for experiment_data in experiment_untwisted:
        maxs = []
        for gen_data in experiment_data["generations"]:
            if i == 0:
                generations.append(gen_data["generation"])

            if "non-twisty_amphi" in gen_data:
                stats = gen_data["non-twisty_amphi"]
                maxs.append(stats["max"])
        max_untwisted_repetitions.append(maxs)
        i += 1
max_untwisted_repetitions = np.array(max_untwisted_repetitions)
means_untwisted = np.mean(max_untwisted_repetitions, axis=0)
stds_untwisted = np.std(max_untwisted_repetitions, axis=0)
sems_u = stds_untwisted / np.sqrt(30)
confidence_margin_u = 1.96 * sems_u
lower_bound_untwisted = means_untwisted - confidence_margin_u
upper_bound_untwisted = means_untwisted + confidence_margin_u

plt.figure(figsize=(10, 6))
plt.plot(generations, means_mixed, marker="o", markersize=4, label="Mean best fitness M", linewidth=2, color='#1f77b4')
plt.plot(generations, means_twisted, marker="o", markersize=4, label="Mean best fitness T", linewidth=2, color="#b41f1f")
plt.fill_between(generations, lower_bound_mixed, upper_bound_mixed, color='#1f77b4', alpha=0.2, linewidth=0)
plt.fill_between(generations, lower_bound_twisted, upper_bound_twisted, color="#b41f1f", alpha=0.2, linewidth=0)

for i in range(len(generations)):
    gen_mixed = max_mixed_repetition[:, i]
    gen_twisty = max_twisted_repetitions[:, i]
    p_val = mannwhitneyu(gen_mixed, gen_twisty)
    marker = None
    if p_val.pvalue < 0.001: marker = "***"
    elif p_val.pvalue < 0.01: marker = "**"
    elif p_val.pvalue < 0.05: marker = "*"

    if marker:
        high_point = max(means_mixed[i] + confidence_margin_m[i], means_twisted[i] + confidence_margin_t[i])
        plt.text(generations[i], high_point + 0.001, marker, 
                ha='center', va='bottom', fontsize=12)

plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness over Generations Amphi - Mixed VS Twisted')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
# plt.ylim(0.17, 0.205)
plt.savefig(f"MT2_fitness_amphi.png")

plt.figure(figsize=(10, 6))
plt.plot(generations, means_untwisted, marker="o", markersize=4, label="Mean best fitness U", linewidth=2, color="#1fb43f")
plt.plot(generations, means_twisted, marker="o", markersize=4, label="Mean best fitness T", linewidth=2, color="#b41f1f")
plt.fill_between(generations, lower_bound_untwisted, upper_bound_untwisted, color="#1fb43f", alpha=0.2, linewidth=0)
plt.fill_between(generations, lower_bound_twisted, upper_bound_twisted, color="#b41f1f", alpha=0.2, linewidth=0)
for i in range(len(generations)):
    gen_untwisted = max_untwisted_repetitions[:, i]
    gen_twisty = max_twisted_repetitions[:, i]
    p_val = mannwhitneyu(gen_untwisted, gen_twisty)
    marker = None
    if p_val.pvalue < 0.001: marker = "***"
    elif p_val.pvalue < 0.01: marker = "**"
    elif p_val.pvalue < 0.05: marker = "*"
    if marker:
        high_point = max(means_untwisted[i] + confidence_margin_u[i], means_twisted[i] + confidence_margin_t[i])
        plt.text(generations[i], high_point + 0.001, marker, 
                ha='center', va='bottom', fontsize=12)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness over Generations Amphi - Untwisted VS Twisted')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
# plt.ylim(0.17, 0.205) 
plt.savefig(f"UT2_fitness_amphi.png")

plt.figure(figsize=(10, 6))
plt.plot(generations, means_untwisted, marker="o", markersize=4, label="Mean best fitness U", linewidth=2, color="#1fb43f")
plt.plot(generations, means_mixed, marker="o", markersize=4, label="Mean best fitness M", linewidth=2, color='#1f77b4')
plt.fill_between(generations, lower_bound_untwisted, upper_bound_untwisted, color="#1fb43f", alpha=0.2, linewidth=0)
plt.fill_between(generations, lower_bound_mixed, upper_bound_mixed, color='#1f77b4', alpha=0.2, linewidth=0)
for i in range(len(generations)):
    gen_untwisted = max_untwisted_repetitions[:, i]
    gen_mixed = max_mixed_repetition[:, i]
    p_val = mannwhitneyu(gen_untwisted, gen_mixed)
    marker = None
    if p_val.pvalue < 0.001: marker = "***"
    elif p_val.pvalue < 0.01: marker = "**"
    elif p_val.pvalue < 0.05: marker = "*"
    if marker:
        high_point = max(means_untwisted[i] + confidence_margin_u[i], means_mixed[i] + confidence_margin_m[i])
        plt.text(generations[i], high_point + 0.001, marker, 
                ha='center', va='bottom', fontsize=12)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness over Generations Amphi - Untwisted VS Mixed')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
# plt.ylim(0.17, 0.205) 
plt.savefig(f"UM2_fitness_amphi.png")

plt.figure(figsize=(10, 6))
plt.plot(generations, means_untwisted, marker="o", markersize=4, label="Mean best fitness U", linewidth=2, color="#1fb43f")
plt.plot(generations, means_mixed, marker="o", markersize=4, label="Mean best fitness M", linewidth=2, color='#1f77b4')
plt.plot(generations, means_twisted, marker="o", markersize=4, label="Mean best fitness T", linewidth=2, color="#b41f1f")
plt.fill_between(generations, lower_bound_untwisted, upper_bound_untwisted, color="#1fb43f", alpha=0.2, linewidth=0)
plt.fill_between(generations, lower_bound_mixed, upper_bound_mixed, color='#1f77b4', alpha=0.2, linewidth=0)
plt.fill_between(generations, lower_bound_twisted, upper_bound_twisted, color="#b41f1f", alpha=0.2, linewidth=0)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness over Generations Amphi - Untwisted, Mixed, Twisted')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
# plt.ylim(0.17, 0.205) 
plt.savefig(f"UMT2_fitness_amphi.png")