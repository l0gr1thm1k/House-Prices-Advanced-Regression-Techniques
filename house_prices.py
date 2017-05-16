# -*- coding: utf-8 -*-
"""
Created on Fri May 12 08:47:14 2017

@author: Daniel
"""

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer, StandardScaler
from one_hot_encode_pandas_frame import one_hot_encode_pandas_frame as ohepf

def preprocess(data):
    one_hot_encoding_columns = []
    dataset = pd.read_csv(data)
    y = dataset.iloc[:, -1].values.reshape(-1, 1)
    del dataset['SalePrice']
    del dataset['Id']
    for index, col in enumerate(dataset.columns):
        if dataset[col].dtype == object:
            one_hot_encoding_columns.append(col)
            encode_category(dataset, col)
        impute_column(dataset, col)

    dataset = ohepf(dataset, one_hot_encoding_columns, replace=True)[0]
    remove_dummy_variable_columns(dataset)
    
    # Scale the data
    sc = StandardScaler()
    X = sc.fit_transform(dataset)
    
    return X, y


def remove_dummy_variable_columns(dataset):
    import re
    columns_to_remove = [x for x in dataset.columns if re.findall('=0.0', x)]
    for column in columns_to_remove:
        del dataset[column]
def impute_column(dataset, col):
    imputer = Imputer(axis=0)
    imputer.fit(dataset[[col]])
    dataset[col] = imputer.transform(dataset[[col]]).ravel()


def encode_category(dataset, col):
    most_common = pd.get_dummies(dataset[col]).sum().sort_values(ascending=False).index[0] 

    def replace_most_common(x):
        if pd.isnull(x):
            return most_common
        else:
            return x

    new_col = dataset[col].map(replace_most_common)
    label_encoder = LabelEncoder()
    dataset[col] = label_encoder.fit_transform(new_col)


def one_hot_encode(X, index):
    one_hot_encoder = OneHotEncoder(categorical_features=[index])
    results = one_hot_encoder.fit_transform(X).toarray()
    return results


import math

def RMSE(y, y_hat):
    try:
        return math.sqrt( (math.log(y) - math.log(y_hat))**2 )
    except ValueError:
        return 0
    
def get_error(prediction, ground_truth):
    total_error = 0
    for i, j in zip(prediction, ground_truth):
        error = RMSE(i, j)
        if error < 1.0:
            total_error += error
        else:
            print(j, i, error)
    return total_error / len(prediction)


if __name__ == '__main__':
    X, y = preprocess('train.csv')
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                        random_state=0)
#    X_test, y_test = preprocess('test.csv')
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    error = get_error(y_pred, y_test)
    print(error)
    
    import matplotlib.pyplot as plt
    plt.scatter(y_pred, y_test, color='red')
    plt.plot(regressor.predict(X_train), y_train, color='blue')
    plt.title("a regression")
    plt.xlabel("prediction")
    plt.ylabel("actual")
    plt.show()
    