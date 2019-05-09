# deal with data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 消除省略号

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

def deal_data():

    # explore

    train = pd.read_csv('Ames_House_train.csv')
    test = pd.read_csv('Ames_House_test.csv')
    # print(train.head())
    # print(train.info())
    # print(train.describe())
    # print(train.isnull().sum())
    # print('\n')
    # print(test.isnull().sum())
    # print(test.head())
    # print(test.info())
    # print(test.describe())

    # 缺失值处理

    # 0填充

    train_num_misfeatures = ['GarageYrBlt', 'MasVnrArea']
    for train_num_misfeature in train_num_misfeatures:
        train[train_num_misfeature] = train[train_num_misfeature].fillna(0)

    test_num_misfeatures = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
                            'GarageYrBlt', 'GarageCars', 'GarageArea', 'BsmtUnfSF']
    for test_num_misfeature in test_num_misfeatures:
        test[test_num_misfeature] = test[test_num_misfeature].fillna(0)


    # None填充

    train_cal_misfeatures = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','MasVnrType','GarageCond',
                       'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1','GarageQual','GarageType','GarageFinish']
    for train_cal_misfeature in train_cal_misfeatures:
        train[train_cal_misfeature] = train[train_cal_misfeature].fillna('None')

    test_cal_misfeatures = ['MSZoning', 'Alley', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'BsmtQual',
                            'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'KitchenQual', 'Functional',
                            'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC',
                            'Fence', 'MiscFeature', 'SaleType']
    for test_cal_misfeature in test_cal_misfeatures:
        test[test_cal_misfeature] = test[test_cal_misfeature].fillna('None')


    # 众数填充

    lots_features = ['Electrical']
    for lots_feature in lots_features:
        train[lots_feature] = train[lots_feature].fillna(train[lots_feature].mode()[0])

    # 中位数填充

    train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    # print(train.isnull().sum())
    # print(test.isnull().sum())


    # 特征处理

    cal_features=['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
                      'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                      'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                      'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
                      'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
                      'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
                      'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                      'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType',
                      'SaleCondition','MSSubClass', 'OverallCond', 'YrSold', 'MoSold']

    for cal_feature in cal_features:
        lbl = LabelEncoder()
        train[cal_feature] = lbl.fit_transform(train[cal_feature].values)
        test[cal_feature] = lbl.fit_transform(test[cal_feature].values)

    train_cal = train[cal_features]
    train_cal = pd.get_dummies(train_cal)

    # 确定训练集，测试集

    x_train = train.drop(columns=['Id', 'SalePrice'])
    y_train = train['SalePrice']
    y_train = np.log(y_train)
    # print(y_train)
    x_test = test.drop(columns=['Id'])
    y_test = 0


    return x_train, x_test, y_train, y_test




deal_data()