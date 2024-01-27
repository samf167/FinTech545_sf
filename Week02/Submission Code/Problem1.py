import pandas as pd
import numpy as np
import statistics as st
import scipy
from scipy import stats
from scipy.integrate import quad

# Read in data
filepath = "/Users/samfuller/Desktop/545/FinTech545_sf/Week02/problem1.csv"
#filepath = 
data = pd.read_csv(filepath)
x = data[["x"]].values 
moment = 0
mean = 0
variance = 0

# standardized formula calculation

for i in range(1,5):
    moment = 0
    n = len(x)
    for value in range(len(x)):
        if i == 1:
            formula = ((x[value]-0))
        if i == 2:
            formula = ((x[value]-mean)**i)
        if i == 3:
            formula = ((x[value]-mean)**i)
        if i == 4:
            formula = (((x[value]-mean))**i)
        moment = moment + formula
    if i == 1:
        mean = moment*((n-1)**-1)
        moment = mean
    if i == 2:
        moment = moment*((n-1)**-1)
        sd = moment**(0.5)
    if i == 3:
        moment = moment*(n/((n-1)*(n-2)))*(1/(sd**3))
    if i == 4:
        moment = moment*((n*sd**4)**-1)
    print("moment", i, "=", moment)

print("Package Data:")

# Statistical package
for i in range(1,5):
    if i == 1:
        moment = stats.moment(data, moment = i, center = 0)
        mean = moment
    else:
        moment = stats.moment(data, moment = i, center = mean)
    
    print("moment", i, "=", moment)
