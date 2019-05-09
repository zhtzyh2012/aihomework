# 岭回归

import data
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

x_train = data.deal_data()[0]
x_test = data.deal_data()[1]
y_train = data.deal_data()[2]
y_test = data.deal_data()[3]

# train

alpha = [0.01, 0.1, 1, 10, 100, 1000]
ridge = RidgeCV(alphas=alpha, store_cv_values=True)
ridge.fit(x_train, y_train)

# alpha

alpha = ridge.alpha_
print('best alpha is : ' + str(alpha))

# test

y_train_pred = ridge.predict(x_train)
y_test = ridge.predict(x_test)

# calculate

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
print('RMSE of training dataset is : ' + str(rmse_train))
print('test result is :')
i = 1461
for j in range(len(y_test)):
    print('Id : ' + str(i) + '  sale price : ' + str(y_test[j]))
    i += 1