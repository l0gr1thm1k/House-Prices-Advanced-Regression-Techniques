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

def preprocess():
    one_hot_encoding_columns = []
    dataset = pd.read_csv('train.csv')
    y_train = dataset.iloc[:, -1].values.reshape(-1, 1)
    last_row = dataset.shape[0]
    del dataset['Id']
    del dataset['SalePrice']
    
    test_dataset= pd.read_csv('test.csv')
    del test_dataset['Id']
    dataset = pd.concat([dataset, test_dataset])

    for index, col in enumerate(dataset.columns):
        if dataset[col].dtype == object:
            one_hot_encoding_columns.append(col)
            encode_category(dataset, col)
        impute_column(dataset, col)

    # dataset = ohepf(dataset, one_hot_encoding_columns, replace=True)[0]
    # remove_dummy_variable_columns(dataset)
    for col in one_hot_encoding_columns:
        del dataset[col]

    # Scale the data
    sc = StandardScaler()
    X = sc.fit_transform(dataset)

    # add a column of 1s to act as the intercept
    X = np.insert(X, 0, 1, axis=1)
    
    X_train = X[:last_row, :]
    X_test = X[last_row:last_row+1459, :] # hack, cause ohepf returns 5k rows
    return X_train, X_test, y_train
    #return X, y_train

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


def make_submission(y_pred):
    f = open('submission.csv', 'w')
    f.write('Id,SalePrice\n')
    for i, j in enumerate(y_pred):
        f.write(str(i+1461) + "," + str(int(j)) + "\n")
    f.close()
    
if __name__ == '__main__':
    X_train, X_test, y_train = preprocess()
    
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    y_pred = [0 if x < 0 else int(x) for x in regressor.predict(X_test)]
    
    make_submission(y_pred)
    

    #error = get_error(y_pred, y_test)
    #print(error)
    
    #import matplotlib.pyplot as plt
    #plt.scatter(y_pred, y_test, color='red')
    #plt.plot(regressor.predict(X_train), y_train, color='blue')
    #plt.title("a regression")
    #plt.xlabel("prediction")
    #plt.ylabel("actual")
    #plt.show()