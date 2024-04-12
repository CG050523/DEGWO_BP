import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error

# 文件录入部分  
df = pd.read_excel('/home/lai/文档/创新训练/data/HF08_residual.xlsx', engine='openpyxl')  
  
# 提取位移数据（第二列，第二行至第六十九行）  
displacement_data = df.iloc[0:68, 1].values
  
# 提取影响因子数据（第三列至第七列，第二行至第六十九行）  
factors_data = df.iloc[0:68, 2:7].values  
  
# 分离训练集和测试集（最后十行作为测试集）  
X_train = factors_data[: -10]
y_train = displacement_data[: -10]
y_train = y_train[:, np.newaxis]
X_tmp = factors_data[-10:]
y_test = displacement_data[-10 :]
y_test = y_test[:, np.newaxis]
X_test = X_tmp

# 定义模型架构
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim = input_dim, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(1, activation = 'linear'))
    model.compile(optimizer = 'adam', loss = 'mse')
    return model

def evaluate(x):
    # 创建一个模型实例
    model = create_model(X_train.shape[1])
    # 设置模型权重
    model.set_weights(x)
    # 评估模型
    y_pred = model.predict(X_test)
    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)
    return mse

# 初始化种群
population_size = 30
# 最大迭代次数
iterations_max = 500
# 交叉概率
crossover_pro = 0.2
# 搜索边界 (bounds)
search_space_x = (-5.12, 5.12)
# search_space_y = (-5.12, 5.12)
# 尺度系数
scaling_factor = 0.4

# 种群位置初始化
population = []
for _ in range(population_size):
    x = random.uniform(search_space_x[0], search_space_x[1])
    population.append(x)

# 变异用到的三参数
a = 2
A = 2 * a * random.random() - a
C = 2 * random.random()
decrease = 2 / iterations_max

# 损失函数
def rastrigin(x):
    const = 10
    return np.square(x) - const * A * np.cos(2 * np.pi * x)

# 变异函数
def mutate(a, A, C):
    return a + scaling_factor * (A - C)

def loss_fuc(x):
    return rastrigin(x)

# 选择
def select_fuc(population_loc, loss_fuc):
    if loss_fuc(population_loc) >= loss_fuc(mutate(a, A, C)):
        population_loc = mutate(a, A,  C)
    return population_loc

# 交叉
def crossover(population):
    for i in range(population_size):
        if random.random() <= crossover_pro or i == random.randint(1, population_size):
            population[i] = select_fuc(population[i], loss_fuc)
    return population

# 更新
def update(population):
    objective_values = [loss_fuc(population[x]) for x in range(population_size)]
    sorted_indexes = sorted(range(len(objective_values)), key = lambda i : objective_values[i])
    sorted_population = [population[i] for i in sorted_indexes]
    return sorted_population

for i in range(population_size):
    population[i] = select_fuc(population[i], loss_fuc)

t = 1

loc = [None] * iterations_max
result = [None] * iterations_max

for t in range(iterations_max):
    sorted_population = update(population)
    best_individuals = sorted_population[:3]
    X_alpha = best_individuals[0]
    X_beta = best_individuals[1]
    X_gamma = best_individuals[2]
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
    population = update(population)

    for i in range(population_size):
        fitness = evaluate(population[i])

    loc[t]  = population[0]
    result[t] = loss_fuc(population[0])


plt.figure('X-POS')
plt.plot(loc)
plt.figure('Result')
plt.plot(result)
plt.show()
