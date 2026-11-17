import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import pymc as pm
import arviz as az

training_data = np.loadtxt("regression_train.txt")
X = training_data[:, :3] # Keeps x1,x2,x3 (n rows, 3 columns)
y = training_data[:, 3] # (n rows, 1 columns)
model = pm.Model()
num_samples = 1000

with model:
    # Defining our priors
    #wtx

    w0 = pm.Normal('w0', mu=0, sigma=20)
    w = pm.Normal('w', mu=0, sigma=20, shape=3)
    sigma = pm.Uniform('sigma', lower=0, upper=20)

    y_est = w0 + pm.math.dot(X,w)

    likelihood = pm.Normal('y', mu=y_est, sigma=sigma, observed=y)

    # inference
    sampler = pm.NUTS() # Hamiltonian MCMC with No U-Turn Sampler
    # or alternatively
    # sampler = pm.Metropolis()
    idata = pm.sample(num_samples, step=sampler, progressbar=True)

plt.figure(figsize=(5, 5))
plt.hist(idata.posterior['w'].values[0, :, 0], 25, histtype='step')
plt.xlabel('$w_1$')
plt.show()

plt.figure(figsize=(5, 5))
plt.hist(idata.posterior['w'].values[0, :, 1], 25, histtype='step')
plt.xlabel('$w_2$')
plt.show()

plt.figure(figsize=(5, 5))
plt.hist(idata.posterior['w'].values[0, :, 2], 25, histtype='step')
plt.xlabel('$w_3$')
plt.show()


plt.figure(figsize=(5, 5))
plt.hist(idata.posterior['sigma'].values[0,:], 25, histtype='step')
plt.xlabel('$\\sigma$')
plt.show()

print(az.summary(
    idata,
    var_names=["w0", "w", "sigma"]
))

az.plot_trace(
    idata,
    var_names=["w0", "w", "sigma"]
)

plt.tight_layout()
plt.show()
#rhat
