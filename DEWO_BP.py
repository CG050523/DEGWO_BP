import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import weight_in as wi

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 初始化权重和偏置
W1 = np.random.rand(5, 10)
b1 = np.random.rand(10)
W2 = np.random.rand(10, 2)
b2 = np.random.rand(2)

learning_rate = 0.01

# 计算损失
def calculate_loss(y, y_pred):
    return np.square(y - y_pred)

# 训练模型
def train_model(x, y, epochs, learning_rate):
    for epoch in range(epochs):
        y_pred = forward_propagation(x)

# 初始化种群
population_size = 30
# 最大迭代次数
iterations_max = 500
# 交叉概率
crossover_pro = 0.2
# 搜索边界 (bounds)
search_space_x = (-5.12, 5.12)
# 尺度系数
scaling_ = 0.6
scaling_factor = 0.4

inputnum = 5
hiddennum = 10
outputnum = 2
# 维度数量
dims = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum
# 权重（链表）之后通过reshape转成矩阵
weight = [None] * dims

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

# 计算损失
def calculate_loss(y, y_pred):
    return np.square(y - y_pred)

# 前向传播
def forward_propagation(x, W_in_hide, b1, W_hide_out, b2):
    z1 = np.dot(x, W_in_hide) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W_hide_out) + b2
    a2 = sigmoid(z2)
    return a2

# 变异函数
def mutate():
    Xp1 = population[random.randint(0, population_size - 1)]
    Xp2 = population[random.randint(0, population_size - 1)]
    Xp3 = population[random.randint(0, population_size - 1)]
    return Xp1 + scaling_factor * (Xp2 - Xp3)

# 选择
def select_fuc(weight):
    W_in_hide, b1, W_hide_out, b2 = wi.func_weight(weight, inputnum, hiddennum, outputnum)
    y_pred = forward_propagation(x=X_train, W_in_hide=W_in_hide, b1=b1, W_hide_out=W_hide_out, b2=b2)
    loss = calculate_loss(y_train, y_pred)
    if loss < learning_rate:
        return weight
    else:
        for i in range(dims):
            weight[i] = update(crossover(population))[0]
        return weight

# 交叉
def crossover(population):
    tmp_population = population
    for i in range(population_size):
        if random.random() <= crossover_pro or i == random.randint(0, population_size - 1):
            tmp_population[i] = mutate()
    return population

# 更新
def update(population):
    y_pred = forward_propagation(x=X_train, W_in_hide=W1, b1=b1, W_hide_out=W2, b2=b2)
    objective_values = [calculate_loss(y, y_pred) for y in y_test]
    sorted_indexes = sorted(range(len(objective_values)), key = lambda i : objective_values[i])
    sorted_population = [population[i] for i in sorted_indexes]
    return sorted_population

def DEWO_OPT(weight):
    weight = select_fuc(weight)
    return weight

weight = DEWO_OPT(population)

t = 1

result = [None] * iterations_max

scaling_factor = scaling_ * (iterations_max - (t - 1)) / iterations_max + 0.2

for t in range(iterations_max):
    for i in range(dims):
        sorted_population = update(population)
        best_individual = sorted_population[:3]
        X_alpha = best_individual[0]
        X_beta = best_individual[1]
        X_delta = best_individual[2]
        for i in np.arange(3, len(sorted_population)):
            D_alpha = abs(C * X_alpha - sorted_population[i])
            D_beta = abs(C * X_beta - sorted_population[i])
            D_gamma = abs(C * X_beta - sorted_population[i])
            X_1 = X_alpha - A * D_alpha
            X_2 = X_beta - A * D_beta
            X_3 = X_delta - A * D_gamma
            sorted_population[i] = (X_1 + X_2 + X_3) / 3
        population = sorted_population
        a = a - decrease
        A = 2 * a * random.random() - a
        C = 2 * random.random()
        population = crossover(population)
        scaling_factor = scaling_ * (iterations_max - (t - 1)) / iterations_max + 0.2
        population = update(population)
        weight[i]  = population[0]
    W_in_hide, b1, W_hide_out, b2 = wi.func_weight(weight, inputnum, hiddennum, outputnum)
    y_pred = forward_propagation(x=X_train, W_in_hide=W_in_hide, b1=b1, W_hide_out=W_hide_out, b2=b2)
    result[t] = calculate_loss(y_train, y_pred)

plt.figure()
plt.plot(result)