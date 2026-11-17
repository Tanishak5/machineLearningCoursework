import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler

training_data = np.loadtxt("regression_train.txt")
X = training_data[:, :3] # Keeps x1,x2,x3 (n rows, 3 columns)
y = training_data[:, 3] # (n rows, 1 columns)
model = pm.Model()
num_samples = 2000

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with model:
    # Defining our priors
    #wtx

    w0 = pm.Normal('w0', mu=0, sigma=20)
    w = pm.Normal('w', mu=0, sigma=20, shape=3)
    sigma = pm.Uniform('sigma', lower=0, upper=20)

    y_est = w0 + pm.math.dot(X_scaled,w)

    likelihood = pm.HalfNormal('y', mu=y_est, sigma=5)

    # inference
    sampler = pm.NUTS() # Hamiltonian MCMC with No U-Turn Sampler
    # or alternatively
    # sampler = pm.Metropolis()
    idata = pm.sample( draws=2000,
                      tune=1000,
                      chains=4,
                      target_accept=0.9)

# plt.figure(figsize=(5, 5))
# plt.hist(idata.posterior['w'].values[0, :, 0], 25, histtype='step')
# plt.xlabel('$w_1$')
# plt.show()

# plt.figure(figsize=(5, 5))
# plt.hist(idata.posterior['w'].values[0, :, 1], 25, histtype='step')
# plt.xlabel('$w_2$')
# plt.show()
#
# plt.figure(figsize=(5, 5))
# plt.hist(idata.posterior['w'].values[0, :, 2], 25, histtype='step')
# plt.xlabel('$w_3$')
# plt.show()
#
#
# plt.figure(figsize=(5, 5))
# plt.hist(idata.posterior['sigma'].values[0,:], 25, histtype='step')
# plt.xlabel('$\\sigma$')
# plt.show()

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

#Standard deviation 20
# mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean  mcse_sd
# w0      0.228  0.106    0.058     0.39     9962     6443  1.00    0.0011   0.0011
# w[0]    1.536  0.107      1.4      1.7    10068     5967  1.00    0.0011   0.0012
# w[1]   -0.068  0.106    -0.24      0.1     9400     6075  1.00    0.0011   0.0011
# w[2]     0.03  0.106    -0.14      0.2     9261     5591  1.00    0.0011   0.0012
# sigma   2.332  0.075      2.2      2.5    10414     6697  1.00   0.00073  0.00081
#Standard deviation 5
#
# mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean  mcse_sd
# w0      0.229  0.106    0.061      0.4     9700     5920  1.00    0.0011   0.0012
# w[0]    1.535  0.104      1.4      1.7     8858     5850  1.00    0.0011   0.0012
# w[1]   -0.069  0.103    -0.23    0.097    10018     6383  1.00     0.001   0.0011
# w[2]     0.03  0.103    -0.13     0.19    10203     6632  1.00     0.001   0.0012
# sigma   2.332  0.074      2.2      2.5     8936     5775  1.00   0.00078  0.00086

#Standard deviation 1

# Sampling 4 chains for 1_000 tune and 2_000 draw iterations (4_000 + 8_000 draws total) took 1 seconds.
# mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean  mcse_sd
# w0      0.225  0.103    0.061     0.39    10505     6060  1.00     0.001   0.0011
# w[0]    1.521  0.104      1.4      1.7    10408     6509  1.00     0.001   0.0012
# w[1]   -0.071  0.103    -0.24    0.095    10428     6260  1.00     0.001   0.0011
# w[2]     0.03  0.104    -0.13      0.2     9930     6636  1.00     0.001   0.0012
# sigma   2.333  0.073      2.2      2.5     8660     6058  1.00   0.00079  0.00082
# Qt: Session management error: None of the authentication protocols specified are supported
# (base) [pv24125@it103177 regression]$

# halfnormal - sigma 5 with std 20
# Sampling 4 chains for 1_000 tune and 2_000 draw iterations (4_000 + 8_000 draws total) took 1 seconds.
# mean    sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean mcse_sd
# w0      0.1  20.3      -33       32     9491     5908  1.00      0.21    0.25
# w[0]      0    20      -32       32    10470     5589  1.00       0.2    0.27
# w[1]   -0.1  20.2      -32       32    10470     6101  1.00       0.2    0.24
# w[2]     -0  19.9      -32       32    10672     5830  1.00      0.19    0.25
# sigma   9.9  5.75        1       19     9126     5377  1.00     0.057   0.045
# Qt: Session management error: None of the authentication protocols specified are supported