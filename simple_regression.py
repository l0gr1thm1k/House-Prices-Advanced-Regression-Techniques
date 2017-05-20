#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:55:37 2017

@author: daniel
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns


fields = ['GrLivArea', 'TotalBsmtSF', 'YearBuilt', 'GarageArea',
          'FullBath']
train = pd.read_csv('train.csv')
y_train = train.iloc[:, -1].values.reshape(-1, 1)
X_train = train[fields].values
X_train = np.nan_to_num(X_train)

test = pd.read_csv('test.csv')
ids = test['Id'].values.reshape(-1, 1)
X_test = test[fields].values
X_test = np.nan_to_num(X_test)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

def make_submission():
    f = open('submission.csv', 'w')
    f.write('Id,SalePrice\n')
    for i, j in zip(ids, y_pred):
        f.write(",".join([str(i[0]), str(j[0])]) + '\n')
    f.close()
    
#plt.scatter(X_train, y_train)
#plt.show()    

make_submission()

#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()