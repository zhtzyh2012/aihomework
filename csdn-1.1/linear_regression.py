# 最小二乘线性回归模型

import data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

x_train = data.deal_data()[0]
x_test = data.deal_data()[1]
y_train = data.deal_data()[2]
y_test = data.deal_data()[3]

# train

lr = LinearRegression()
lr.fit(x_train, y_train)

# predict

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

# calculate

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print('RMSE of training dataset is : ' + str(rmse_train))
print('RMSE of test dataset is : ' + str(rmse_test))

# print(y_test_pred)
#
# # plot
#
# y_test_pred_mean = y_test_pred.mean()
# y_test_pred_std = y_test_pred.std()
# y_test_pred = (y_test_pred - y_test_pred_mean) / y_test_pred_std
#
# fig = plt.figure()
# plt.plot(y_test_pred, c='red', label='pred')
# plt.plot(y_test, c='blue', label='real')
# plt.show()
# print(y_test_pred)
