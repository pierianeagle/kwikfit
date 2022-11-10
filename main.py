# %%

import matplotlib.pyplot as plt
import numpy as np
from iminuit import Minuit, cost
from scipy.stats import expon, skewnorm

# from nll import NLL

# %%

x_range = (0, 5)

# data generating function
x_signal = skewnorm.rvs(a=1.5, loc=2, scale=1/2, size=1000)
x_noise  = expon.rvs(scale=3, size=(len(x_signal)))

x_data = np.append(x_signal, x_noise)
x_data = x_data[(x_range[0] < x_data) & (x_data < x_range[1])]

freq, edges = np.histogram(x_data, bins=50)
x_centre = 0.5 * (edges[1:] + edges[:-1])
x_diff = np.diff(x_centre)

# %%

def pdf(x, z, a, loc, scale, expscale):
    """Custom probability density function.
    
    This must be normalised, which is achieved by the weight, z.
    """
    return z * skewnorm.pdf(x, a, loc, scale) + (1 - z) * expon.pdf(x, scale=expscale)

def cdf(x, z, a, loc, scale, expscale):
    """Custom cumulative distribution function."""
    return z * skewnorm.cdf(x, a, loc, scale) + (1 - z) * expon.cdf(x, scale=expscale)

# %%

# hypothesis
params = ('z', 'a', 'loc', 'scale', 'expscale')

# the negative log-likelihood is now built in
cost = cost.UnbinnedNLL(x_data, pdf) # NLL(data, pdf)
# binned fits are more computationally efficient and numerically stable for large datasets
# cost = cost.ExtendedBinnedNLL(freq, edges, cdf)

# specify initial values here
m = Minuit(cost, z=0.5, a=0, loc=0, scale=1, expscale=1, name=params)

# specify limits here
m.limits['z']        = (0, 1)
m.limits['scale']    = (0, None)
m.limits['expscale'] = (0, None)

# %%

print("Parameter Values and Errors:\n", m.migrad(), "\n")
print("Valid:\n", m.valid, "\n")
print("Correlation Matrix:\n", m.covariance.correlation(), "\n")

print("Final Values:")
for param in params:
    print(f"{param} = {m.values[param]:.3f} +- {m.errors[param]:.3f}")

# exclude the null with a hypothesis test of the alternate
# calculate systematic errors along the function by finding the largest shift
# between the possible alternates (the parameter errors define possible fits)

# %%

# plot the fit
plt.errorbar(x_centre, freq, freq**0.5, fmt='ok')

x_plot = np.linspace(*x_range, len(x_data))

plt.plot(x_plot, pdf(x_plot, *[p.value for p in m.init_params]) * len(x_data) * x_diff[0], ls=':', label='Initial')
plt.plot(x_plot, pdf(x_plot, *m.values) * len(x_data) * x_diff[0],                                 label='Fit')

# plt.stairs(np.diff(cdf(edges, *[p.value for p in m.init_params])) * len(x_data), edges, ls=':', label='Initial')
# plt.stairs(np.diff(cdf(edges, *m.values)) * len(x_data),                         edges,         label='Fit')

plt.legend()

plt.show()
