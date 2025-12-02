from scipy.stats import mannwhitneyu
import json
import numpy as np
gecko_type1 = "gecko"
gecko_type2 = "gecko_good"
gecko_type3 = "gecko_doubletwist"
gecko_type4 = "gecko_doubletwist_turtle"
gecko_type5 = "gecko_front"
gecko_type6 = "gecko_untwisted"
with open(f"./__data__/{gecko_type1}_fitnesses.json", "r") as f:
            data1 = json.load(f)
with open(f"./__data__/{gecko_type2}_fitnesses.json", "r") as f:
            data2 = json.load(f)
with open(f"./__data__/{gecko_type3}_fitnesses.json", "r") as f:
            data3 = json.load(f)
with open(f"./__data__/{gecko_type4}_fitnesses.json", "r") as f:
            data4 = json.load(f)
with open(f"./__data__/{gecko_type5}_fitnesses.json", "r") as f:
            data5 = json.load(f)
with open(f"./__data__/{gecko_type6}_fitnesses.json", "r") as f:
            data6 = json.load(f)
datax = data5
gecko1 = gecko_type5
datay = data6
gecko2 = gecko_type6
mean_runs1 = np.zeros(len(datax[0]))
mean_slope1 = np.zeros(len(datax[0]))
mean_runs2 = np.zeros(len(datay[0]))
mean_slope2 = np.zeros(len(datay[0]))
for history in datax:
        for i in range(len(history)):
            mean_runs1[i] += history[i]
            if i != (len(history)-1):
                deltax = history[i+1] - history[i]
                mean_slope1[i] += deltax 

    # print(mean_runs)
for j in range(len(mean_runs1)):
        mean_runs1[j] = mean_runs1[j]/(len(datax))
        mean_slope1[j] = mean_slope1[j]/(len(datax))

for history in datay:
        for i in range(len(history)):
            mean_runs2[i] += history[i]
            if i != (len(history)-1):
                deltax = history[i+1] - history[i]
                mean_slope2[i] += deltax 

    # print(mean_runs)
for j in range(len(mean_runs2)):
        mean_runs2[j] = mean_runs2[j]/(len(datay))
        mean_slope2[j] = mean_slope2[j]/(len(datay))
p = mannwhitneyu(mean_runs1, mean_runs2)
p2 = mannwhitneyu(mean_slope1, mean_slope2)
print(p.pvalue)
print(f"{gecko1} median: {np.median(mean_runs1)}")
print(f"{gecko2} median: {np.median(mean_runs2)}")
print(p2.pvalue)
print(f"{gecko1} median: {np.median(mean_slope1)}")
print(f"{gecko2} median: {np.median(mean_slope2)}")