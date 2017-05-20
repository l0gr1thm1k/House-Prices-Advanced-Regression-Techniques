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
    X_train, X_test, y = hp.preprocess() #dataset)
    X_opt = X_train[:, [i for i,x in enumerate(X_train[0])]]
    X_opt_test = X_test[:, [i for i,x in enumerate(X_test[0])]]
       
    regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
    # print(regressor_OLS.summary())
    # Look for predictor with highest P-value
    # remove this predictor and refit the model
    max_value, max_index = max_p_value(regressor_OLS)
    while max_value > significance_value:
        X_opt = X_opt[:, [i for i,x in enumerate(X_opt[0]) if i != max_index]]
        X_opt_test = X_opt_test[:, [i for i,x in enumerate(X_opt_test[0]) if i != max_index]]
        regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
        max_value, max_index = max_p_value(regressor_OLS)
        print(max_value)
    # when no predictors exceed the significance value, the model is finished
    return regressor_OLS, X_opt_test
    
    
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
    model, test_data = backward_elimination("train.csv")
    print(model.summary())
    #test_data = test_data.reshape(-1, 1)
    y_pred = model.predict(test_data)
    hp.make_submission(y_pred)