import json
from scipy.stats import mannwhitneyu
import numpy as np
experiment_mixed = []
experiment_twisted = []
experiment_untwisted = []
for i in range(10):
    try:
        with open(f"evo_main/experiment_data_non-twisty_champion{i}.json", 'r') as file:
            experiment_mixed.append(json.load(file))
    except:
        with open(f"evo_main/experiment_data_twisty_champion{i}.json", 'r') as file:
            experiment_mixed.append(json.load(file))
    try:
        with open(f"evo_main/experiment_data_non-twisty_champion{i+10}.json", 'r') as file:
            experiment_twisted.append(json.load(file))
    except:
        with open(f"evo_main/experiment_data_twisty_champion{i+10}.json", 'r') as file:
            experiment_twisted.append(json.load(file))
    try:
        with open(f"evo_main/experiment_data_non-twisty_champion{i+20}.json", 'r') as file:
            experiment_untwisted.append(json.load(file))
    except:
        with open(f"evo_main/experiment_data_twisty_champion{i+20}.json", 'r') as file:
            experiment_untwisted.append(json.load(file))
# print(len(experiment_untwisted))
generations = []
max_fitness_mixed = np.zeros(len(experiment_mixed[0]["generations"]))
avg_fitness_mixed = np.zeros(len(experiment_mixed[0]["generations"]))
i = 0
for experiment_data in experiment_mixed:
        maxs = []
        means = []
        for gen_data in experiment_data["generations"]:
            if i == 0:
                generations.append(gen_data["generation"])

            if "mixed_twisty" in gen_data:
                stats = gen_data["mixed_twisty"]
                maxs.append(stats["max"])
                means.append(stats["mean"])
        i += 1
        for i in range(len(experiment_data["generations"])):
            max_fitness_mixed[i] += maxs[i]
            avg_fitness_mixed[i] += means[i]
for i in range(len(max_fitness_mixed)):
    max_fitness_mixed[i] = max_fitness_mixed[i]/10
    avg_fitness_mixed[i] = avg_fitness_mixed[i]/10
# print(max_fitness_mixed)
# print(avg_fitness_mixed)
generations = []
max_fitness_twisted = np.zeros(len(experiment_twisted[0]["generations"]))
avg_fitness_twisted = np.zeros(len(experiment_twisted[0]["generations"]))
i = 0
for experiment_data in experiment_twisted:
        maxs = []
        means = []
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
    max_fitness_twisted[i] = max_fitness_twisted[i]/10
    avg_fitness_twisted[i] = avg_fitness_twisted[i]/10
# print(max_fitness_twisted)
# print(avg_fitness_twisted)
generations = []
max_fitness_untwisted = np.zeros(len(experiment_untwisted[0]["generations"]))
avg_fitness_untwisted = np.zeros(len(experiment_untwisted[0]["generations"]))
i = 0
for experiment_data in experiment_untwisted:
        maxs = []
        means = []
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
    max_fitness_untwisted[i] = max_fitness_untwisted[i]/10
    avg_fitness_untwisted[i] = avg_fitness_untwisted[i]/10
# print(max_fitness_untwisted)
# print(avg_fitness_untwisted)

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