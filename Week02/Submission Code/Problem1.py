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
            formula = ((x[value]-0))*((n-1)**-1)
        if i == 2:
            formula = ((x[value]-mean)**i)*((n-1)**-1)
        if i == 3:
            formula = ((x[value]-mean)**i)*(n/((n-1)*(n-2)))*(1/(sd**3))
        if i == 4:
            formula = (((x[value]-mean)/sd)**i)*(n**-1)
        moment = moment + formula
        
    print("moment", i, "=", moment)
    if i == 1:
        mean = moment
    if i == 2:
        sd = moment**(0.5)

print("Package Data:")

# Statistical package
for i in range(1,5):
    moment = stats.moment(data, moment = i)
    print("moment", i, "=", moment)
