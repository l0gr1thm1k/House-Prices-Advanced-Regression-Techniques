# -*- coding: utf-8 -*-
"""
Created on Fri May 12 08:47:14 2017

@author: Alison
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, :-1].values