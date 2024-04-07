import pandas as pd  
import numpy as np
# import degw as DEW

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
  
# 读取Excel文件  
df = pd.read_excel('/home/lai/文档/创新训练/data/HF08_residual.xlsx', engine='openpyxl')  
  
# 提取位移数据（第二列，第二行至第六十九行）  
displacement_data = df.iloc[1:68, 1].values  
  
# 提取影响因子数据（第三列至第七列，第二行至第六十九行）  
factors_data = df.iloc[1:68, 2:6].values  
  
# 分离训练集和测试集（最后十行作为测试集）  
train_factors = factors_data[:-10]  
train_displacement = displacement_data[:-10]  
test_factors = factors_data[-10:]  
test_displacement = displacement_data[-10:]

# 定义BP神经网络
# 创建Sequential模型
model = Sequential()

# 添加第一个隐藏层，包含10个神经元，使用ReLU激活函数
model.add(Dense(units=64, activation='relu', input_shape=(4,)))

# 添加第二个隐藏层，包含5个神经元，使用ReLU激活函数
model.add(Dense(units=1))

# 添加输出层，包含一个神经元，使用线性激活函数
model.add(Dense(1))

# 编译模型
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(train_factors, train_displacement, epochs=150, batch_size=1)

# 设置参数
population_size = 50
max_iterations = 100

# def objective_function(params):
#     # 定义目标函数
#     return np.linalg.norm(params)

# best_solution = None

def objective_function(W1, W2, X, y):
    # 计算神经网络的输出
    # z1 = np.dot(W1, X)
    # a1 = tf.sigmoid(z1)
    # z2 = np.dot(W2, a1)
    # y_pred = tf.sigmoid(z2)

    # # 计算损失函数
    # loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    return np.mean((W1))

def DEGWO(objective_function, bounds, population_size, iterations, a):
    # 初始化种群
    population = np.random.uniform(bounds[0], bounds[1], (population_size, len(bounds)))
    fitness = np.apply_along_axis(objective_function, 1, population)

    for i in range(iterations):
        # 差分进化
        for j in range(population_size):
            idxs = np.random.choice(np.arange(population_size), 3, replace=False)
            a1, a2, a3 = population[idxs]
            mutant = a1 + a * (a2 - a3)
            population = np.vstack((population, mutant))

        # 计算适应度
        fitness = np.apply_along_axis(objective_function, 1, population)

        # 选择种群
        idxs = np.argsort(fitness)[:population_size]
        population = population[idxs]

        # 灰狼优化
        for j in range(population_size):
            r1, r2 = np.random.uniform(0, 1, 2)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            l = np.abs(C1 * population[j] - population[np.argmin(fitness)])
            x1 = population[j] - A1 * l
            population = np.vstack((population, x1))

        # 计算适应度
        fitness = np.apply_along_axis(objective_function, 1, population)

        # 选择种群
        idxs = np.argsort(fitness)[:population_size]
        population = population[idxs]

    return population[np.argmin(fitness)]

# 设置参数
bounds = [-1, 1]
population_size = 50
iterations = 100
a = 0.5
b = 1
c = 1









weights1 = model.layers[0].get_weights()[0]
weights2 = model.layers[1].get_weights()[0]
print("weights1: \n",weights1)
print("weights2: \n",weights2)
# print(weights)
# weights = weights.reshape(-1)
best_solution = DEGWO(objective_function(weights1,weights2,train_factors,train_displacement),bounds,population_size,iterations,a)
predictions = model.predict(best_solution.reshape(1, -1))

# 使用训练好的模型进行预测
predictions = model.predict(test_factors)

# 打印预测结果
# print("Predictions:", predictions) 

# 保存模型
model.save('bp_model.h5')

# 加载模型
loaded_model = tf.keras.models.load_model('bp_model.h5')

# 使用加载的模型进行预测
new_predictions = loaded_model.predict(test_factors)

# 打印预测结果
print("New predictions:\n", new_predictions)
