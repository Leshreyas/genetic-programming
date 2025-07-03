import operator
import random
import matplotlib.pyplot as plt
import numpy as np
import graphviz
from deap import gp
from deap import base, creator, tools, gp
from deap.algorithms import eaSimple

# === GRID ===
GRID = [
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
]


# === ANT CLASS ===
class Ant:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.dir = 1  # 0=up, 1=right, 2=down, 3=left
        self.grid = [row[:] for row in GRID]
        self.food = 0
        self.visited = set()
        self.steps = 0
        self.max_steps = 400

    def forward(self):
        if self.steps >= self.max_steps:
            return
        dx, dy = 0, 0
        if self.dir == 0: dy = -1
        elif self.dir == 1: dx = 1
        elif self.dir == 2: dy = 1
        elif self.dir == 3: dx = -1

        nx, ny = self.x + dx, self.y + dy
        if 0 <= nx < 10 and 0 <= ny < 10:
            self.x, self.y = nx, ny
            if self.grid[ny][nx] == 1:
                self.food += 1
                self.grid[ny][nx] = 0
        self.steps += 1

    def turn_left(self):
        self.dir = (self.dir - 1) % 4

    def turn_right(self):
        self.dir = (self.dir + 1) % 4

    def sense_food_ahead(self):
        dx, dy = 0, 0
        if self.dir == 0: dy = -1
        elif self.dir == 1: dx = 1
        elif self.dir == 2: dy = 1
        elif self.dir == 3: dx = -1
        nx, ny = self.x + dx, self.y + dy
        return 0 <= nx < 10 and 0 <= ny < 10 and self.grid[ny][nx] == 1


# === GP SETUP ===
pset = gp.PrimitiveSet("MAIN", 0)


# Primitives
def if_food_then_else(cond, a, b):
    def wrapped():
        if cond():
            a()
        else:
            b()
    return wrapped


# Terminals (wrapped in lambdas)
def move_forward(): return lambda: ant.forward()
def turn_left(): return lambda: ant.turn_left()
def turn_right(): return lambda: ant.turn_right()
def sense_food_ahead(): return lambda: ant.sense_food_ahead()


# Register primitives and terminals
pset.addPrimitive(if_food_then_else, 3)
pset.addTerminal(move_forward(), name="move_forward")
pset.addTerminal(turn_left(), name="turn_left")
pset.addTerminal(turn_right(), name="turn_right")
pset.addTerminal(sense_food_ahead(), name="sense_food_ahead")


# Create types only once
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)



# Evaluation function
def eval_ant(individual):
    global ant
    ant = Ant()
    func = toolbox.compile(expr=individual)
    try:
        func()
    except:
        pass
    return ant.food,

toolbox.register("evaluate", eval_ant)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# === RUNNING THE GP ===


def run_gp():
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, log = eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=True)
    return pop, log, hof[0]


if __name__ == "__main__":
    # === RUN GP ===
    pop, log, best_ind = run_gp()

    # === PLOT FITNESS CURVE ===
    plt.plot(log.select("avg"), label="Average Fitness")
    plt.plot(log.select("max"), label="Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Food Collected)")
    plt.title("Santa Fe Ant GP Evolution")
    plt.legend()
    plt.grid()
    plt.show()

    # === DRAW TREE OF BEST INDIVIDUAL===
    nodes, edges, labels = gp.graph(best_ind)

    dot = graphviz.Digraph(format="jpeg")
    dot.attr(rankdir="TB", size="10,20")
    dot.attr("node", shape="box", fontsize="10")

    for node in nodes:
        label = labels[node]
        is_terminal = label in ["move_forward", "turn_left", "turn_right", "sense_food_ahead"]
        dot.node(str(node), label, style="filled", fillcolor="lightblue" if is_terminal else "lightcoral")

    for edge in edges:
        dot.edge(str(edge[0]), str(edge[1]))

    dot.render("best_ant_tree", view=True)
