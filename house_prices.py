# -*- coding: utf-8 -*-
"""
Created on Fri May 12 08:47:14 2017

@author: Alison
"""

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer


def preprocess(data):
    one_hot_encoding_columns = [] # these are indexes to columns to be encoded
    dataset = pd.read_csv(data)
    for index, col in enumerate(dataset.columns):
        if dataset[col].dtype == object:
            encode_category(dataset, col)
            one_hot_encoding_columns.insert(0, index)
        # impute_column(dataset, col)
    # one hot encode
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values.reshape(-1, 1)
    return X, y


def impute_column(dataset, col):
    imputer = Imputer(axis=1)
    imputer.fit(dataset[col])
    dataset[col] = imputer.transform(dataset[col])


def encode_category(dataset, col):
    array = np.array(['Nan' if pd.isnull(x) else x for x in dataset[col]])
    label_encoder = LabelEncoder()
    dataset[col] = label_encoder.fit_transform(array)


def one_hot_encode(X, index):
    one_hot_encoder = OneHotEncoder(categorical_features=[index])
    X = one_hot_encoder.fit_transform(X).toarray()
    
#if __name__ == '__main__':
X, y = preprocess('train.csv')

dataset = pd.read_csv('train.csv')
impute_column(dataset, 'LotFrontage')
