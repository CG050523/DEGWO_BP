import random
import numpy as np
import matplotlib.pyplot as plt

population_size = 20
iterations_max = 300
crossover_pro = 0.5
# 定义搜索空间
search_space_x = (-5, 5)
scaling_factor = 0.2

population = []
for _ in range(population_size):
    x = random.uniform(search_space_x[0], search_space_x[1])
    population.append([x])

population = np.array(population)

a = 2
A = 2 * a * random.random() - a
C = 2 * random.random()
decrease = 2 / iterations_max

def rastrigin(x):
    const = 10
    return const * np.sum(np.square(x) - const * np.cos(2 * np.pi * x))

def mutate(a, A, C):
    return a + scaling_factor * (A - C)

def loss_fuc(x):
    return rastrigin(x)

def select_fuc(population_loc, loss_fuc):
    if loss_fuc(population_loc) >= loss_fuc(mutate(a, A, C)):
        population_loc = mutate(a, A,  C)
    return population_loc

def crossover(population):
    for i in range(population_size):
        if random.random() <= crossover_pro or i == random.randint(1, population_size):
            population[i] = select_fuc(population[i], loss_fuc)
    return population

def update(population):
    objective_values = [loss_fuc(population[x]) for x in range(population_size)]
    sorted_indexes = sorted(range(len(objective_values)), key = lambda i : objective_values[i])
    sorted_population = [population[i] for i in sorted_indexes]
    return sorted_population

for i in range(population_size):
    population[i] = select_fuc(population[i], loss_fuc)

t = 1

result = [None] * iterations_max

for t in range(iterations_max):
    sorted_population = update(population)
    best_individuals = sorted_population[:3]
    X_alpha = best_individuals[0]
    X_beta = best_individuals[1]
    X_gamma = best_individuals[2]
    # fourth_item_onwards = sorted_population[3:]
    for i in np.arange(3, len(sorted_population)):
        D_alpha = abs(C * X_alpha - sorted_population[i])
        D_beta = abs(C * X_beta - sorted_population[i])
        D_gamma = abs(C * X_beta - sorted_population[i])
        X_1 = X_alpha - A * D_alpha
        X_2 = X_beta - A * D_beta
        X_3 = X_gamma - A * D_gamma
        sorted_population[i] = (X_1 + X_2 + X_3) / 3
    population = sorted_population
    a = a - decrease
    A = 2 * a * random.random() - a
    C = 2 * random.random()
    population = crossover(population)
    # population = select_fuc(population, loss_fuc)
    population = update(population)
    result[t]  = loss_fuc(population[0])

plt.figure()
plt.plot(result)
plt.show()
