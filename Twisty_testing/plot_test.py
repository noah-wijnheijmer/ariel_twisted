import json
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import numpy as np
experiment_mixed = []
experiment_twisted = []
experiment_untwisted = []
for i in range(30):
    try:
        with open(f"evo_main/experiment_data_non-twisty_champion{i+60}.json", 'r') as file:
            experiment_mixed.append(json.load(file))
    except:
        with open(f"evo_main/experiment_data_twisty_champion{i+60}.json", 'r') as file:
            experiment_mixed.append(json.load(file))
    try:
        with open(f"evo_main/experiment_data_non-twisty_champion{i+30}.json", 'r') as file:
            experiment_twisted.append(json.load(file))
    except:
        with open(f"evo_main/experiment_data_twisty_champion{i+30}.json", 'r') as file:
            experiment_twisted.append(json.load(file))
    try:
        with open(f"evo_main/experiment_data_non-twisty_champion{i}.json", 'r') as file:
            experiment_untwisted.append(json.load(file))
    except:
        with open(f"evo_main/experiment_data_twisty_champion{i}.json", 'r') as file:
            experiment_untwisted.append(json.load(file))
# print(len(experiment_untwisted))
generations = []
max_fitness_mixed = np.zeros(len(experiment_mixed[0]["generations"]))
avg_fitness_mixed = np.zeros(len(experiment_mixed[0]["generations"]))
overall_max_mixed = []
i = 0
for experiment_data in experiment_mixed:
        maxs = []
        means = []
        overall_max_mixed.append(experiment_data["final_statistics"]["mixed"]["overall_max"])
        for gen_data in experiment_data["generations"]:
            if i == 0:
                generations.append(gen_data["generation"])

            if "mixed" in gen_data:
                stats = gen_data["mixed"]
                maxs.append(stats["max"])
                means.append(stats["mean"])
        i += 1
        for i in range(len(experiment_data["generations"])):
            max_fitness_mixed[i] += maxs[i]
            avg_fitness_mixed[i] += means[i]
for i in range(len(max_fitness_mixed)):
    max_fitness_mixed[i] = max_fitness_mixed[i]/30
    avg_fitness_mixed[i] = avg_fitness_mixed[i]/30
print(np.argmax(overall_max_mixed))
overall_max_mixed[12] = 0
print(np.argmax(overall_max_mixed))
overall_max_mixed[9] = 0
print(np.argmax(overall_max_mixed))
# print(max_fitness_mixed)
# print(avg_fitness_mixed)
generations = []
max_fitness_twisted = np.zeros(len(experiment_twisted[0]["generations"]))
avg_fitness_twisted = np.zeros(len(experiment_twisted[0]["generations"]))
i = 0
overall_max_twisted = []
for experiment_data in experiment_twisted:
        maxs = []
        means = []
        overall_max_twisted.append(experiment_data["final_statistics"]["twisty"]["overall_max"])
        for gen_data in experiment_data["generations"]:
            if i == 0:
                generations.append(gen_data["generation"])

            if "twisty" in gen_data:
                stats = gen_data["twisty"]
                maxs.append(stats["max"])
                means.append(stats["mean"])
        i += 1
        for i in range(len(experiment_data["generations"])):
            max_fitness_twisted[i] += maxs[i]
            avg_fitness_twisted[i] += means[i]
for i in range(len(max_fitness_twisted)):
    max_fitness_twisted[i] = max_fitness_twisted[i]/30
    avg_fitness_twisted[i] = avg_fitness_twisted[i]/30
print(np.argmax(overall_max_twisted))
overall_max_twisted[22] = 0
print(np.argmax(overall_max_twisted))
overall_max_twisted[28] = 0
print(np.argmax(overall_max_twisted))
# print(max_fitness_twisted)
# print(avg_fitness_twisted)
generations = []
max_fitness_untwisted = np.zeros(len(experiment_untwisted[0]["generations"]))
avg_fitness_untwisted = np.zeros(len(experiment_untwisted[0]["generations"]))
i = 0
overall_max_untwisted = []
for experiment_data in experiment_untwisted:
        maxs = []
        means = []
        overall_max_untwisted.append(experiment_data["final_statistics"]["non_twisty"]["overall_max"])
        for gen_data in experiment_data["generations"]:
            if i == 0:
                generations.append(gen_data["generation"])

            if "non_twisty" in gen_data:
                stats = gen_data["non_twisty"]
                maxs.append(stats["max"])
                means.append(stats["mean"])
        i += 1
        for i in range(len(experiment_data["generations"])):
            max_fitness_untwisted[i] += maxs[i]
            avg_fitness_untwisted[i] += means[i]
for i in range(len(max_fitness_untwisted)):
    max_fitness_untwisted[i] = max_fitness_untwisted[i]/30
    avg_fitness_untwisted[i] = avg_fitness_untwisted[i]/30
print(np.argmax(overall_max_untwisted))
overall_max_untwisted[14] = 0
print(np.argmax(overall_max_untwisted))
overall_max_untwisted[2] = 0
print(np.argmax(overall_max_untwisted))
# print(max_fitness_untwisted)
# print(avg_fitness_untwisted)
plt.figure(figsize=(10, 6))
plt.plot(
        generations, max_fitness_untwisted, marker="o", markersize=4, label="Max U", linewidth=2,)
plt.plot(
        generations, max_fitness_twisted, marker="o", markersize=4, label="Max T", linewidth=2,)
plt.plot(
        generations, avg_fitness_untwisted, marker="o", markersize=4, label="Mean U", linewidth=2,)
plt.plot(
        generations, avg_fitness_twisted, marker="o", markersize=4, label="Mean T", linewidth=2,)
plt.xlabel("Generation No.", fontsize=12)
plt.ylabel("Fitness", fontsize=12)
title = f"Fitness over Generations - Twisted VS Untwisted"
plt.title(title, fontsize=14)
plt.legend(loc="upper left", fontsize=10)
plt.grid(visible=True, alpha=0.3)
plt.ylim((0.15, 0.22)) 
# I force integer ticks on x-axis (half generations don't make sense to me)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.savefig(f"TVU_fitness.png")

plt.figure(figsize=(10, 6))
plt.plot(
        generations, max_fitness_mixed, marker="o", markersize=4, label="Max M", linewidth=2,)
plt.plot(
        generations, max_fitness_twisted, marker="o", markersize=4, label="Max T", linewidth=2,)
plt.plot(
        generations, avg_fitness_mixed, marker="o", markersize=4, label="Mean M", linewidth=2,)
plt.plot(
        generations, avg_fitness_twisted, marker="o", markersize=4, label="Mean T", linewidth=2,)
plt.xlabel("Generation No.", fontsize=12)
plt.ylabel("Fitness", fontsize=12)
title = f"Fitness over Generations - Twisted VS Mixed"
plt.title(title, fontsize=14)
plt.legend(loc="upper left", fontsize=10)
plt.grid(visible=True, alpha=0.3)
plt.ylim((0.15, 0.22))
# I force integer ticks on x-axis (half generations don't make sense to me)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.savefig(f"TVM_fitness.png")

plt.figure(figsize=(10, 6))
plt.plot(
        generations, max_fitness_untwisted, marker="o", markersize=4, label="Max U", linewidth=2,)
plt.plot(
        generations, max_fitness_mixed, marker="o", markersize=4, label="Max M", linewidth=2,)
plt.plot(
        generations, avg_fitness_untwisted, marker="o", markersize=4, label="Mean U", linewidth=2,)
plt.plot(
        generations, avg_fitness_mixed, marker="o", markersize=4, label="Mean M", linewidth=2,)
plt.xlabel("Generation No.", fontsize=12)
plt.ylabel("Fitness", fontsize=12)
title = f"Fitness over Generations - Untwisted VS Mixed"
plt.title(title, fontsize=14)
plt.legend(loc="upper left", fontsize=10)
plt.grid(visible=True, alpha=0.3)
plt.ylim((0.15, 0.22))
# I force integer ticks on x-axis (half generations don't make sense to me)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.savefig(f"UVM_fitness.png")

p1 = mannwhitneyu(max_fitness_mixed, max_fitness_twisted)
p2 = mannwhitneyu(avg_fitness_mixed, avg_fitness_twisted)
p3 = mannwhitneyu(max_fitness_untwisted, max_fitness_twisted)
p4 = mannwhitneyu(avg_fitness_untwisted, avg_fitness_twisted)
p5 = mannwhitneyu(max_fitness_mixed, max_fitness_untwisted)
p6 = mannwhitneyu(avg_fitness_mixed, avg_fitness_untwisted)
print(f"p_value: {p1.pvalue}")
print(f"median max_mixed: {np.median(max_fitness_mixed)}, median max_twisted: {np.median(max_fitness_twisted)}")
print(f"p_value: {p2.pvalue}")
print(f"median avg_mixed: {np.median(avg_fitness_mixed)}, median avg_twisted: {np.median(avg_fitness_twisted)}")
print(f"p_value: {p3.pvalue}")
print(f"median max_untwisted: {np.median(max_fitness_untwisted)}, median max_twisted: {np.median(max_fitness_twisted)}")
print(f"p_value: {p4.pvalue}")
print(f"median avg_untwisted: {np.median(avg_fitness_untwisted)}, median avg_twisted: {np.median(avg_fitness_twisted)}")
print(f"p_value: {p5.pvalue}")
print(f"median max_mixed: {np.median(max_fitness_mixed)}, median max_untwisted: {np.median(max_fitness_untwisted)}")
print(f"p_value: {p6.pvalue}")
print(f"median avg_mixed: {np.median(avg_fitness_mixed)}, median avg_untwisted: {np.median(avg_fitness_untwisted)}")