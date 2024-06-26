import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# 假设你已经加载了数据到 Pandas DataFrame中
# data: 包含所有3850个站点的温度和风速
# era5_data: 包含协变量数据（十米高度的矢量风速、两米高度温度、海平面气压）

# 这里用生成数据作示例
# 站点数量
num_stations = 3850

# 生成训练数据 examples：num_stations * num_timesteps(24 months * 30 days/month * 24 hours/day = 2 years of hourly data)
num_timesteps = 24 * 365 * 2  # 假设24个月小时数据，2年

# 生成全球站点气象要素(temp, wind)
data = np.random.rand(num_stations, num_timesteps, 2)  # 假设 (3850, num_timesteps, 2) 其中2代表温度和风速

# 生成协变量数据(10U, 10V, T2M, MSL)
era5_data = np.random.rand(num_stations, num_timesteps // 3, 4)  # 假设 (3850, num_timesteps//3, 4) 其中4代表协变量

# 由于协变量需要3小时分辨率，我们需要处理这些数据与每小时的气象要素对齐
era5_data_resized = np.repeat(era5_data, 3, axis=1)[:, :num_timesteps, :]

# 合并数据
combined_data = np.concatenate((data, era5_data_resized), axis=2)  # (3850, num_timesteps, 6)

# 每个站点使用168个时间点预测未来24个时间点，因此定义训练和测试数据
input_length = 168
output_length = 24

def create_dataset(data, input_len, output_len):
    X, y_temp, y_wind = [], [], []
    for i in range(data.shape[1] - input_len - output_len):
        X.append(data[:, i:i+input_len, :])
        y_temp.append(data[:, i+input_len:i+input_len+output_len, 0])
        y_wind.append(data[:, i+input_len:i+input_len+output_len, 1])
    return np.array(X), np.array(y_temp), np.array(y_wind)

X, y_temp, y_wind = create_dataset(combined_data, input_length, output_length)
X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
_, _, y_wind_train, y_wind_test = train_test_split(X, y_wind, test_size=0.2, random_state=42)

# 构建LSTM模型
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(output_length))
    model.compile(optimizer='adam', loss='mse')
    return model

input_shape = (input_length, X_train.shape[3])  # (168, 6)

# 预测温度
temp_model = build_lstm_model(input_shape)
temp_model.fit(X_train, y_temp_train, epochs=10, batch_size=32, validation_split=0.2)

# 预测风速
wind_model = build_lstm_model(input_shape)
wind_model.fit(X_train, y_wind_train, epochs=10, batch_size=32, validation_split=0.2)

# 预测结果
temp_pred = temp_model.predict(X_test)
wind_pred = wind_model.predict(X_test)

# 保存预测结果
np.save('temp.npy', temp_pred)
np.save('wind.npy', wind_pred)
