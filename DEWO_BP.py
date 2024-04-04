import pandas as pd  
import numpy as np
import degw as DEW

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
model.fit(train_factors, train_displacement, epochs=300, batch_size=1)

# 设置参数
dim = 10
pop_size = 50
max_iter = 300
lb = -10
ub = 10

best_solution = None
# 访问权重矩阵
for layer in model.layers:
    weights = layer.get_weights()
    best_solution = DEW.de_gwo(np.array(train_factors).reshape(1, -1),np.array(train_displacement).reshape(1, -1),np.array(weights).reshape(1, -1),pop_size,max_iter,0.5,0.5,0.5)
    predictions = model.predict(best_solution.reshape(1, -1))

# 使用训练好的模型进行预测
# predictions = model.predict(test_factors)

# 打印预测结果
print("Predictions:", predictions) 

# 保存模型
model.save('bp_model.h5')

# 加载模型
loaded_model = tf.keras.models.load_model('bp_model.h5')

# 使用加载的模型进行预测
new_predictions = loaded_model.predict(test_factors)

# 打印预测结果
print("New predictions:\n", new_predictions)
