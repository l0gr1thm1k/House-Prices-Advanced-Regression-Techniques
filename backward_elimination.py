# -*- coding: utf-8 -*-
"""
Created on Mon May 15 20:46:59 2017

@author: Daniel
"""

import house_prices as hp
import os
import re
import sys



def backward_elimination(dataset, significance_value=0.05):
    """
    @desc   - fit an optimal linear regression model using backward
              elimination.
    @param  - dataset: a filename or pandas dataframe to optimize on
    @param  - significance_value: a float value describing the threshold above
              which, a variable should be excluded from the model
    @return - model: return the best fit model
    """
    import statsmodels.formula.api as sm
    
    # fit model with all predictors
    X, y = hp.preprocess(dataset)
    X_opt = X[:, [i for i,x in enumerate(X[0])]]
       
    regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
    
    # Look for predictor with highest P-value
    # remove this predictor and refit the model
    max_value, max_index = max_p_value(regressor_OLS)
    while max_value > significance_value:
        X_opt = X_opt[:, [i for i,x in enumerate(X_opt[0]) if i != max_index]]
        regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
        max_value, max_index = max_p_value(regressor_OLS)
    # when no predictors exceed the significance value, the model is finished
    return regressor_OLS
    
    
def max_p_value(results):
    """
    @desc   - find the maximum p-value in a linear regression model, and
              return the value, and independent variable index
    @param  - results: a statmodels.formula.api.OLS fit result
    @return - maximum: a tuple containing the maximum index and value
    """
    orig_stdout = sys.stdout
    sys.stdout = open("temp.txt", "w")
    print(results.summary())
    sys.stdout.close()
    str_results = open("temp.txt", "r").read()
    matches = re.findall("x\d+.*\n", str_results)
    sys.stdout = orig_stdout
    os.remove("temp.txt")
    maximum =max([(float(x.split()[4]), int(x.split()[0][1])) for x in matches])
    return maximum
    
    

if __name__ == "__main__":
    model = backward_elimination("train.csv")
    print(model.summary())