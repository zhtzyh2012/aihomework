# Lasso

import data
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

x_train = data.deal_data()[0]
x_test = data.deal_data()[1]
y_train = data.deal_data()[2]
y_test = data.deal_data()[3]

# train

alpha = [0.01, 0.1, 1, 10, 100, 1000]
lasso = LassoCV(alphas=alpha, cv=5)
lasso.fit(x_train, y_train)

# alpha

alpha = lasso.alpha_
print('best alpha is : ' + str(alpha))

# calculate

y_train_pred = lasso.predict(x_train)
y_test = lasso.predict(x_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
print('RMSE of training dataset is : ' + str(rmse_train))
print('test result is :')
i = 1461
for j in range(len(y_test)):
    print('Id : ' + str(i) + '  sale price : ' + str(y_test[j]))
    i += 1
