import pandas as pd  
import numpy as np
import myDE_wolf as DEW

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
  
# 读取Excel文件  
df = pd.read_excel(r'', engine='openpyxl')  
  
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
model.fit(train_factors, train_displacement, epochs=1000, batch_size=1)

# 使用训练好的模型进行预测
predictions = model.predict(test_factors)

# 打印预测结果
print(predictions)

# 定义适应度函数
def fitness_function(params):
    # 将参数转换为影响因子矩阵
    factors = np.array([params]).T
    # 使用BP神经网络进行预测
    predictions = model.predict(factors)
    # 计算预测误差
    error = np.mean((predictions - test_displacement) ** 2)
    return error

# 设置参数
dim = 10
pop_size = 50
max_iter = 1000
lb = -10
ub = 10
diff_grey_wolf =DEW.DIFFGreyWolf(model, dim, pop_size, max_iter, lb, ub)
# 运行
[best_x, best_fv] = diff_grey_wolf.run()

# 打印最优参数和最优适应度值
print("Best parameters:", best_x)
print("Best fitness:", best_fv)

# 预测结果
predictions = model.predict(best_x.reshape(1, -1))
print("Predictions:", predictions)

# 保存模型
model.save('bp_model.h5')

# 加载模型
loaded_model = tf.keras.models.load_model('bp_model.h5')

# 使用加载的模型进行预测
new_predictions = loaded_model.predict(test_factors)

# 打印预测结果
print("New predictions:", new_predictions)

# 保存预测结果
np.savetxt(r'C:\Users\邵国龙\Desktop\data-dachuang\predictions.txt', new_predictions, delimiter=',')

# 加载预测结果
loaded_predictions = np.loadtxt('predictions.txt', delimiter=',')

# 打印加载的预测结果
print("Loaded predictions:", loaded_predictions)

# 计算预测误差
error = np.mean((loaded_predictions - test_displacement) ** 2)
print("Error:", error)

# 保存预测误差
np.savetxt(r'C:\Users\邵国龙\Desktop\data-dachuang\error.txt', error, delimiter=',')

# 加载预测误差
loaded_error = np.loadtxt('error.txt', delimiter=',')

# 打印加载的预测误差
print("Loaded error:", loaded_error)

# 绘制预测结果和实际结果的对比图
import matplotlib.pyplot as plt

plt.plot(test_displacement, label='Actual')
plt.plot(loaded_predictions, label='Predicted')
plt.legend()
plt.show()

# 绘制预测误差和适应度值的对比图
plt.plot(best_fv, label='Fitness')
plt.plot(loaded_error, label='Error')
plt.legend()
plt.show()

