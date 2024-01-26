import pandas as pd
import numpy as np
import statistics as st
import scipy
from scipy import stats
from scipy.integrate import quad

# Read in data
data = pd.read_csv("/Users/samfuller/Desktop/545/FinTech545_sf/Week02/problem1.csv")

x = data[["x"]].values 
y = data[["y"]].values
moment = 0
mean = 0

# standardized formula calculation

for i in range(1,5):
    moment = 0
    for value in range(len(x)):
        if i == 1:
            formula = ((x[value]-0)**1)*(y[value])
            #print(((x[value]-0)**1), (y[value]))
        else:
            formula = ((x[value]-mean)**i)*(y[value])
        moment = moment + formula
        #print(moment)
        
    print("moment", i, "=", moment)
    if i == 1:
        mean = moment



# Statistical package
for i in range(1,5):
    moment = stats.moment(data, moment = 1, axis = 0)
    print("moment", i, "=", moment)
