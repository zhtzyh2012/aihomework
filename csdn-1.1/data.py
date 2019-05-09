# deal with dataset

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split


def deal_data():

    # data exploring

    data = pd.read_csv('day.csv')
    # print(data.head())
    # print(data.info())
    # print(data.describe())

    # feature engineer
    categorical_features = ['season', 'mnth', 'weekday', 'weathersit']
    # for cat_fea in categorical_feature:
    #     print("不同属性出现的次数：" + cat_fea)
    #     print(data[cat_fea].value_counts())
    #     print('\n')

    # one-hot coding
    data_cat = data[categorical_features]
    # print(data_cat)
    data_cat = pd.get_dummies(data_cat)
    # print(data_cat)

    # normalizing
    mms = MinMaxScaler()
    numberical_features = ['temp', 'atemp', 'hum', 'windspeed']
    temp = mms.fit_transform(data[numberical_features])
    data_num = pd.DataFrame(data=temp, columns=numberical_features, index=data.index)
    # print(data_num)

    # features summary
    data = pd.concat([data['instant'], data['yr'], data_cat, data_num, data['holiday'], data['workingday'], data['cnt']], axis=1)
    data.to_csv('FE_day.csv')

    # select data
    x_train, x_test, y_train, y_test = train_test_split(data, data, test_size=0.2)
    x_train = x_train.drop(columns=['instant', 'yr', 'cnt'])
    y_train = y_train['cnt']
    x_test = x_test.drop(columns=['instant', 'yr', 'cnt'])
    y_test = y_test['cnt']

    # normalization
    x_train = scale(x_train)
    x_test = scale(x_test)
    y_train_mean = y_train.mean()
    y_test_mean = y_test.mean()
    y_train_std = y_train.std()
    y_test_std = y_train.std()
    y_train = (y_train - y_test_mean) / y_test_std
    y_test = (y_test - y_test_mean) / y_test_std

    return x_train, x_test, y_train, y_test

