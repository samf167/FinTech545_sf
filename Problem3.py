import numpy as np
import scipy.stats as stats
import pandas as pd
import numpy as np
import statistics as st
import scipy
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
import seaborn
import math
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/samfuller/Desktop/545/FinTech545_sf/Week02/problem3.csv")

# Initialize arrays to store AIC and BIC values
AIC = []
BIC = []

for i in range(1,4):   
    # Compute AR(1)-AR(3) using sm library, print results and add to AIC/BIC arrays
    ar_i = sm.tsa.SARIMAX(data, order = (i,0,0),seasonal_order = (0,0,0,0),trend='c')
    result_ari = ar_i.fit()
    print("AR", i, result_ari.summary())
    AIC.append(result_ari.aic)
    BIC.append(result_ari.bic)

    # Compute MA(1)-MA(3) using sm library, print results and add to AIC/BIC arrays
    mai = sm.tsa.SARIMAX(data, order = (0,0,i),seasonal_order = (0,0,0,0),trend='c')
    result_mai = mai.fit()  
    print("MA", i, result_mai.summary())
    AIC.append(result_mai.aic)
    print(result_mai.aic)
    BIC.append(result_mai.bic)

print("AIC", AIC)
print("BIC", BIC)

print("AR3 is best of fit as AIC/BIC are lowest")