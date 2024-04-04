import numpy as np

def objective_function(inputs, targets, params):
    # 定义目标函数，这里以均方误差（MSE）作为损失函数
    mse = 0
    for i in range(len(params)):
        output = np.dot(inputs[i], params)
        mse += (targets[i] - output) ** 2
    return mse / len(targets)

def de_gwo(inputs, targets, weights, population_size, max_iterations, alpha, beta, gamma):
    # 初始化种群
    population = np.random.rand(population_size, inputs.shape[1])

    # 初始化最佳解和最佳适应度值
    best_solution = None
    best_fitness = np.inf

    for i in range(max_iterations):
        # 计算种群适应度值
        fitness_values = np.array([objective_function(inputs, targets, weights)])

        # 更新最佳解和最佳适应度值
        best_index = np.argmin(fitness_values)
        if fitness_values[best_index] < best_fitness:
            best_fitness = fitness_values[best_index]
            best_solution = population[best_index]

        # 差分进化操作
        a = alpha * np.random.rand(population_size, inputs.shape[1])
        b = beta * np.random.rand(population_size, inputs.shape[1])
        c = gamma * np.random.rand(population_size, inputs.shape[1])

        for j in range(population_size):
            for k in range(inputs.shape[1]):
                population[j][k] = a[j][k] * best_solution[k] + b[j][k] * population[j][k] + c[j][k] * (np.random.rand() * (population[j][k] - population[int(np.random.rand() * (population_size - 1))][k]))

    return best_solution

# 示例：使用DE-GWO算法优化BP神经网络权重矩阵
# inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# targets = np.array([[0], [1], [1], [0]])
# population_size = 50
# max_iterations = 100
# alpha = 0.5
# beta = 0.5
# gamma = 0.5

# best_weights = de_gwo(inputs, targets, population_size, max_iterations, alpha, beta, gamma)
# print("Best Weights:", best_weights)
# print("Best Fitness value:", objective_function(best_weights))
