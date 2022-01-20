# %%

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import skewnorm
from iminuit import Minuit

from nll import NLL

# %%

# data generating function
data = skewnorm.rvs(4, size=1000)

# hypothesis
params = ("a", "loc", "scale")

nll = NLL(skewnorm.pdf, data)
# specify initial values here
m = Minuit(nll, a=0, loc=0, scale=1, name=params)

# %%

print("Parameter Values and Errors:\n", m.migrad(), "\n")
print("Correlation Matrix:\n", m.covariance.correlation(), "\n")

print("Final Values:")
for param in params:
    print("{} = {:.3f} +- {:.3f}".format(param, m.values[param], m.errors[param]))

# exclude the null with a hypothesis test of the alternate
# calculate systematic errors along the function by finding the largest shift
# between the possible alternates (the parameter errors define possible fits)

# %%

# plot the fit
bins = 100
limits = (-1.5, 4.5)

x = np.linspace(limits[0], limits[1])

# normalise the  histogram
plt.hist(data, bins, limits, True)
plt.plot(x, skewnorm.pdf(x, *m.values))

plt.title("Histogram Fit")
plt.ylabel("Frequency")

plt.show()
