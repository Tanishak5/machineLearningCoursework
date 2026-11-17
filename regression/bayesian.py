import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import pymc as pm
import arviz as az

training_data = np.loadtxt("regression_train.txt")
X = training_data[:, :3] # Keeps x1,x2,x3 (n rows, 3 columns)
y = training_data[:, 3] # (n rows, 1 columns)
model = pm.Model()
num_samples = 2000

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

#BASLIEN MODEL
# Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 2 seconds.
# Qt: Session management error: None of the authentication protocols specified are supported
# mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean  mcse_sd
# w0      0.156  0.106   -0.018     0.33     5890     3409  1.00    0.0014   0.0016
# w[0]    1.376  0.097      1.2      1.5     7199     3532  1.00    0.0011   0.0015
# w[1]   -0.124  0.181    -0.41     0.16     5545     3491  1.00    0.0024   0.0027
# w[2]     0.01  0.035   -0.045    0.066     5651     2960  1.00   0.00046  0.00053
# sigma   2.334  0.074      2.2      2.5     5289     3259  1.00     0.001   0.0012
#
# #MODEL WHERE PRIOR - W HAS STD1
# Qt: Session management error: None of the authentication protocols specified are supported
# mean     sd eti89_lb eti89_ub ess_bulk        ess_tail   r_hat    mcse_mean  mcse_sd
# w0      0.152  0.103   -0.018     0.31     5409     2798  1.00    0.0014   0.0016
# w[0]    1.364  0.095      1.2      1.5     6458     3047  1.00    0.0012   0.0016
# w[1]   -0.114   0.18     -0.4     0.18     6407     3379  1.00    0.0022   0.0026
# w[2]   0.0105  0.035   -0.046    0.067     6368     3316  1.00   0.00044  0.00053
# sigma   2.332  0.073      2.2      2.5     5708     3327  1.00   0.00098   0.0011

#mODEL pior has std 5
# Sampling 4 chains for 1_000 tune and 1_000 draw iterations (4_000 + 4_000 draws total) took 1 seconds.
# Qt: Session management error: None of the authentication protocols specified are supported
# mean     sd eti89_lb eti89_ub ess_bulk            ess_tail r_hat  mcse_mean  mcse_sd
# w0     0.155  0.106   -0.014     0.33     5514     3132  1.00    0.0014   0.0015
# w[0]   1.376  0.094      1.2      1.5     6527     3248  1.00    0.0012   0.0014
# w[1]   -0.12  0.183    -0.41     0.17     7062     3449  1.00    0.0022   0.0028
# w[2]    0.01  0.036   -0.047    0.068     5108     3402  1.00    0.0005  0.00053
# sigma  2.331  0.074      2.2      2.5     6523     3541  1.00   0.00092   0.0013