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
    for index, col in enumerate(dataset.columns):
        if dataset[col].dtype == object:
        else:
            impute_column(dataset, col)
    dataset = ohepf(dataset, one_hot_encoding_columns, replace=True)[0]
    #X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values.reshape(-1, 1)
    
    # Scale the data
    # sc = StandardScaler()
    # X = sc.fit_transform(X)
    return dataset, y


def impute_column(dataset, col):
    imputer = Imputer(axis=0)
    imputer.fit(dataset[[col]])
    dataset[col] = imputer.transform(dataset[[col]]).ravel()


def encode_category(dataset, col):
    array = np.array(['Nan' if pd.isnull(x) else x for x in dataset[col]])
    label_encoder = LabelEncoder()
    dataset[col] = label_encoder.fit_transform(array)


def one_hot_encode(X, index):
    one_hot_encoder = OneHotEncoder(categorical_features=[index])
    results = one_hot_encoder.fit_transform(X).toarray()
    return results

from sklearn.feature_extraction import DictVectorizer

#if __name__ == '__main__':
X, y = preprocess('train.csv')

def test():
    dataset = pd.read_csv('train.csv')
    one_hot_encoding_columns = []
    for i, j in enumerate(dataset.columns):
        if dataset[j].dtype == object:
            one_hot_encoding_columns.append(j)
    print(one_hot_encoding_columns)
    new_result = ohepf(dataset, one_hot_encoding_columns, replace=True)[0]
    return new_result

new_result = test()


dataset = pd.read_csv('train.csv')
#result = ohepf(dataset, ['MSZoning', 'Street'], replace=True)[0]
#dataset = pd.read_csv('train.csv')
#encode_category(dataset, 'MSZoning')
#test_encode = one_hot_encode(dataset, 2)
#dataset['MSZoning'] = test_encode
