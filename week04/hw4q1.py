import numpy as np
import math
import statistics


mu, sigma = 0, 0.01 # mean and standard deviation
trials = 1000000
p_last = 100
s = np.random.normal(mu, sigma, trials)
print(s)

# Classical Brownian Motion
cbm = p_last + s
cbm_mean = statistics.mean(cbm)
cbm_sd = statistics.stdev(cbm)
print("mean", cbm_mean, "stdev", cbm_sd)

# Arithmetic Return System
ars = p_last*(1+s)
ars_mean = statistics.mean(ars)
ars_sd = statistics.stdev(ars)
print("mean", ars_mean, "stdev", ars_sd)

# Geometric Brownian Motion
gbm = p_last*(math.e**s)
gbm_mean = statistics.mean(gbm)
gbm_sd = statistics.stdev(gbm)
print("mean", gbm_mean, "stdev", gbm_sd)

