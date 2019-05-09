# 最小二乘回归模型

import data
import numpy as np
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
y_test = lr.predict(x_test)

# calculate

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
print('RMSE of training dataset is : ' + str(rmse_train))
print('test result is :')
i = 1461
for j in range(len(y_test)):
    print('Id : ' + str(i) + '  sale price : ' + str(y_test[j]))
    i += 1