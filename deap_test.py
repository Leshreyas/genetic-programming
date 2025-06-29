import random
import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools


GRID = [
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0],]
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def mutates(individual, indpb=0.2):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.randint(-1, 1)
    return individual,


def evaluate(individual, size):
    direction = 1
    food_collected = 0
    x_pos = 0
    y_pos = 0
    counted = []
    for move in individual:
        direction = (move + direction) % 4
        if direction == 0 and y_pos != 0:
            y_pos -= 1
        elif direction == 1 and x_pos != size - 1:
            x_pos += 1
        elif direction == 2 and y_pos != size - 1:
            y_pos += 1
        elif direction == 3 and x_pos != 0:
            x_pos -= 1
        if GRID[y_pos][x_pos] and [x_pos, y_pos] not in counted:
            food_collected += 1
            counted.append([x_pos, y_pos])
    return food_collected,


toolbox = base.Toolbox()
toolbox.register("steps", random.randint, -1, 1)   # -1 is left, 0 is straight, 1 is right
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.steps, 50)
toolbox.register("mutate", mutates, indpb=0.2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate, size=len(GRID))


def main():
    pop = toolbox.population(n=50)
    MUTPB, NGEN = 0.2, 10000
    fitnesses = map(toolbox.evaluate, pop)
    total_weights = []
    for fit, ind in zip(fitnesses, pop):
        ind.fitness.values = fit
    for g in range(NGEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for fit, ind in zip(fitnesses, invalid_ind):
            ind.fitness.values = fit
        best_ind = pop[0]
        for ind in pop:
            if ind.fitness.values > best_ind.fitness.values:
                best_ind = ind
        pop[:] = offspring
        least_ind = pop[0]
        for ind in pop:
            if ind.fitness.values < least_ind.fitness.values:
                least_ind = ind
        pop.remove(least_ind)
        pop.append(best_ind)
        weights = []
        for ind in pop:
            weights.append(ind.fitness.values[0])
        average = sum(weights) / len(weights)
        total_weights.append(average)
    for ind in pop:
        print(ind.fitness.values)
    gens = [gen for gen in range(0, NGEN)]
    plt.plot(gens, total_weights)
    plt.show()


if __name__ == "__main__":
    main()