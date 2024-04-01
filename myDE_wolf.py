import numpy as np
import random

# 定义差异灰狼优化类
class DIFFGreyWolf:
    #python类初始化的方法，其中self为固定参数，func为目标函数，即待计算函数，dim为种群个体维度，pop_size为种群中个体的数量
    #max_iter为最大迭代次数，lb为约束下界(low bounder)，ub为约束上界(upper bounder)
    def __init__(self, func, dim, pop_size, max_iter, lb, ub):
        self.func = func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub

    # 初始化种群
    def init_population(self):
        population = np.random.uniform(self.lb, self.ub, [self.pop_size, self.dim])
        return population

    # 计算适应度
    def fitness(self, population):
        fitness_values = []
        for i in range(self.pop_size):
            fitness_values.append(self.func(population[i]))
        return np.array(fitness_values)

    # 排序种群
    def sort_population(self, population, fitness_values):
        sorted_population = population[np.argsort(fitness_values)]
        sorted_fitness_values = fitness_values[np.argsort(fitness_values)]
        return sorted_population, sorted_fitness_values

    # 更新alpha
    def update_alpha(self, population, fitness_values):
        self.alpha = population[0]
        self.alpha_fv = fitness_values[0]
        for i in range(1, self.pop_size):
            if fitness_values[i] > self.alpha_fv:
                self.alpha = population[i]
                self.alpha_fv = fitness_values[i]

    # 更新beta
    def update_beta(self, population, fitness_values):
        self.beta = population[0]
        self.beta_fv = fitness_values[0]
        for i in range(1, self.pop_size):
            if fitness_values[i] > self.beta_fv:
                self.beta = population[i]
                self.beta_fv = fitness_values[i]

    # 更新delta
    def update_delta(self, population, fitness_values):
        self.delta = population[0]
        self.delta_fv = fitness_values[0]
        for i in range(1, self.pop_size):
            if fitness_values[i] > self.delta_fv:
                self.delta = population[i]
                self.delta_fv = fitness_values[i]

    # 交叉
    def crossover(self, parent1, parent2):
        child = np.zeros([1, self.dim])
        for i in range(self.dim):
            child[0, i] = random.uniform(parent1[i], parent2[i])
        return child

    # 变异
    def mutation(self, individual):
        mutation_indices = random.sample(range(self.dim), int(self.dim * 0.1))
        for index in mutation_indices:
            individual[0, index] = random.uniform(self.lb, self.ub)
        return individual

    # 进化
    def evolve(self, population, fitness_values):
        new_population = []
        for i in range(self.pop_size):
            a = random.uniform(0, 1)
            b = random.uniform(0, 1)
            c = random.uniform(0, 1)
            d = random.uniform(0, 1)
            e = random.uniform(0, 1)

            A = self.alpha + a * (self.beta - self.alpha)
            B = self.delta + b * (self.delta - self.delta)
            C = B + c * (A - B)
            D = self.ub - (self.ub - self.lb) * d
            E = self.lb + (self.ub - self.lb) * e

            new_individual1 = self.crossover(population[i], C)
            new_individual2 = self.mutation(new_individual1)
            new_population.append(new_individual2)
        new_population = np.array(new_population)
        new_fitness_values = self.fitness(new_population)
        return new_population, new_fitness_values

    # 运行
    def run(self):
        population = self.init_population()
        fitness_values = self.fitness(population)
        sorted_population, sorted_fitness_values = self.sort_population(population, fitness_values)
        self.alpha = sorted_population[0]
        self.alpha_fv = sorted_fitness_values[0]
        self.beta = sorted_population[1]
        self.beta_fv = sorted_fitness_values[1]
        self.delta = sorted_population[2]
        self.delta_fv = sorted_fitness_values[2]

        for iter in range(self.max_iter):
            population, fitness_values = self.evolve(sorted_population, sorted_fitness_values)
            sorted_population, sorted_fitness_values = self.sort_population(population, fitness_values)

            self.update_alpha(sorted_population, sorted_fitness_values)
            self.update_beta(sorted_population, sorted_fitness_values)
            self.update_delta(sorted_population, sorted_fitness_values)

            if iter % 100 == 0:
                print('iter:', iter, 'best fv:', self.alpha_fv)

        return self.alpha, self.alpha_fv
