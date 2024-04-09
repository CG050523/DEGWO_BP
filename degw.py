import random
from scipy.optimize import rastrigin


population_size = 20
iterations_max = 500
crossover_pro = 0.2
# 定义搜索空间
search_space_x = (-5, 5)
scaling_factor = 0.2

population = []
for _ in range(population_size):
    x = random.uniform(search_space_x[0], search_space_x[1])
    population.append([x])

a = 2
A = 2 * a * random.random() - a
C = 2 * random.random()
decrease = 2/iterations_max

def mutate(a,A,C):
    return a + scaling_factor * (A - C)

def loss_fuc(x):
    return rastrigin(x)

def select_fuc(population, loss_fuc):
    for i in range(population):
        if population[i] < loss_fuc(mutate(a,A,C)):
            population[i] = population[i]
        if population[i]>= loss_fuc(mutate(a,A,C)):
            population[i] = mutate(a,A,C)
    return population

def dis_calc(population):
    for i in (3, population):
        individual = population[i]
        D_alpha = abs(C * X_alpha - individual)
        D_beta = abs(C * X_beta - individual)
        D_gamma = abs(C * X_beta - individual)
        X_1 = X_alpha - A * D_alpha
        X_2 = X_beta - A * D_beta
        X_3 = X_gamma - A * D_gamma
        individual = (X_1 + X_2 + X_3) / 3
        population[i] = individual
    return population

population = select_fuc(population, loss_fuc)

t = 1
for t in range(iterations_max):
    objective_values = [loss_fuc(x) for x in population]
    sorted_indexes = sorted(range(len(objective_values)), key=lambda i : objective_values[i])
    sorted_population = [population[i] for i in sorted_indexes]
    best_individuals = sorted_population[:3]
    X_alpha = best_individuals[0]
    X_beta = best_individuals[1]
    X_gamma = best_individuals[2]
    for i in (3, population):
        individual = population[i]
        D_alpha = abs(C * X_alpha - individual)
        D_beta = abs(C * X_beta - individual)
        D_gamma = abs(C * X_beta - individual)
        X_1 = X_alpha - A * D_alpha
        X_2 = X_beta - A * D_beta
        X_3 = X_gamma - A * D_gamma
        individual = (X_1 + X_2 + X_3) / 3
        population[i] = individual
    a = a - t * decrease
    A = 2 * a * random.random() - a
    C = 2 * random.random()
    for j in range(population):
        if random.random() <= crossover_pro | j  == random.randint(1,population_size):
            population[j] = mutate(a, A, C)
        if random.random() > crossover_pro | i!=random.randint(1,population_size):
            population[j] = population[j]
    population =  select_fuc(population, loss_fuc)

