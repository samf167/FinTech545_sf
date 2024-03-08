
For both files, all that is needed is to change the filepaths as specified and the following packages:
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import t
import scipy
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from scipy.integrate import quad
import warnings


Q2 is commented fully.

For question 3, please refer to the comments in "Q3" which
performs the calculations to calculate VaR and ES using the copula 
and requested distributions. The other q3 files calculate the VaR and ES
for each individual portfolio which is effectively the same code with fewer 
porfolios and slightly less complexity. The comments in the "q3" file 
explain all the code.