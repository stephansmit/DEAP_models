import pickle
import random
from deap import creator, base
import numpy as np

import matplotlib.pyplot as plt
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

restart_file1='ga_restart2.pkl'
restart_file2='ga_restart3.pkl'

with open(restart_file1, "r") as cp_file:
    cp = pickle.load(cp_file)
population1 = cp["population"]
start_gen1 = cp["generation"]
halloffame1 = cp["halloffame"]
logbook1 = cp["logbook"]
random.setstate(cp["rndstate"])


with open(restart_file2, "r") as cp_file:
    cp = pickle.load(cp_file)
population2 = cp["population"]
start_gen2 = cp["generation"]
halloffame2 = cp["halloffame"]
logbook2 = cp["logbook"]
random.setstate(cp["rndstate"])


gen1 = logbook1.select("gen")
fit_mins1 = logbook1.chapters["fitness"].select("min")
size_avgs1 = logbook1.chapters["individual"].select("max")

gen2 = logbook2.select("gen")
fit_mins2 = logbook2.chapters["fitness"].select("min")
size_avgs2 = logbook2.chapters["individual"].select("max")




fig, ax1 = plt.subplots()
line1 = ax1.plot(gen1, fit_mins1, "b-",marker='x', label="Minimum Fitness")
line2 = ax1.plot(gen2, fit_mins2, "r-", label="Minimum Fitness")

ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness")


# for tl in ax1.get_yticklabels():
#     tl.set_color("b")

ax2 = ax1.twinx()
line12 = ax2.plot(gen1, size_avgs1, label="Average Size")
line22 = ax2.plot(gen2, size_avgs2, marker='x', label="Average Size")

ax2.set_ylabel("Individuals")
# for tl in ax2.get_yticklabels():
#     tl.set_color("r")

# lns = line1 + line2
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc="center right")

plt.show()
