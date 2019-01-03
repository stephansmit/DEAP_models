import random
from deap import base
from deap import creator
from deap import tools
from functools import partial
import numpy as np
import pickle
from deap import algorithms

def objective(individual, generation):
    return random.random()+individual[0],


creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register("attr_bool", random.uniform, 0.90, 1.10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# toolbox.register("map", pool.map)
toolbox.register("mate", tools.cxTwoPoints)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", objective)
pop = toolbox.population(n=30)
CXPB, MUTPB = 0.5, 1.0




if False:
    # A file name has been given, then load the data from the file
    with open(checkpoint, "r") as cp_file:
        cp = pickle.load(cp_file)
    population = cp["population"]
    start_gen = cp["generation"]
    halloffame = cp["halloffame"]
    logbook = cp["logbook"]
    random.setstate(cp["rndstate"])
else:
    population = toolbox.population(n=30)
    start_gen = 0
    halloffame = tools.HallOfFame(maxsize=1)
    logbook = tools.Logbook()



stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)



FREQ=1
for gen in range(start_gen, 100):
    population = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(partial(toolbox.evaluate, generation=gen), invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    halloffame.update(population)
    record = stats.compile(population)
    logbook.record(gen=gen, evals=len(invalid_ind), **record)

    population = toolbox.select(population, k=len(population))

    if gen % FREQ == 0:
        # Fill the dictionary using the dict(key=value[, ...]) constructor
        cp = dict(population=population, generation=gen, halloffame=halloffame,
                  logbook=logbook, rndstate=random.getstate())

        with open("checkpoint_name.pkl", "wb") as cp_file:
            pickle.dump(cp, cp_file)


for i in logbook:
    print(i['avg'])
