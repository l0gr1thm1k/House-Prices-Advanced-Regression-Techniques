# -*- coding: utf-8 -*-
"""
Created on Mon May 15 08:45:13 2017

@author: Daniel
"""

import pandas as pd
from sklearn.feature_extraction import DictVectorizer

def one_hot_encode_pandas_frame(data, cols, replace=False):
    """
    @desc   - given a dataframe and a list of columns, one hot encode the
              columns. The code for this function was taken from the following
              repo:
                  
              https://gist.github.com/kljensen/5452382
              
    @param  - data: a pandas dataframe object.
    @param  - cols: column titles to replace with one hot encoding
    @return - data: the original dataframe, unmodified unless replace param is
              True
    @return - vecData: a dataframe of one hot encoded categories only
    @return - vec: the the DictVectorizer object which has been fitted to the 
              original data
    """
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData, vec)


#result = one_hot_encode_pandas_frame(dataset, ['MSZoning', 'Street'], replace=True)[0]

def test():
    dataset = pd.read_csv('train.csv')
    one_hot_encoding_columns = []
    for i, j in enumerate(dataset.columns):
        if dataset[j].dtype == object:
            one_hot_encoding_columns.append(j)
    new_result = one_hot_encode_pandas_frame(dataset, one_hot_encoding_columns, replace=True)[0]
    return new_result

new_result = test()