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


if __name__ == '__main__':
    X, y = preprocess('train.csv')