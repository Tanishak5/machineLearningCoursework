import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# best MODEL SO FAR
# polynomial for x2/x2
# noise = 0,20
# standard deviation = 1?
#NUTS

import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

training_data = np.loadtxt("regression_train.txt")
X = training_data[:, :3] # Keeps x1,x2,x3 (n rows, 3 columns)
y = training_data[:, 3] # (n rows, 1 columns)
model = pm.Model()
num_samples = 2000

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

X1_train = X_train[:, 0:1]
X1_val = X_val[:, 0:1]
# Apply polynomial transformation to x2 and x3
poly = PolynomialFeatures(degree=2, include_bias=False)
X23_train = poly.fit_transform(X_train[:, 1:3])
X23_val = poly.transform(X_val[:, 1:3])
scaler = StandardScaler()

X_features_train = np.column_stack((X1_train, X23_train))
X_features_val = np.column_stack((X1_val, X23_val))

# Combine features
X_scaled_train = scaler.fit_transform(X_features_train)
X_scaled_val = scaler.transform(X_features_val)


with model:
    # Defining our priors
    #wtx

    w0 = pm.Normal('w0', mu=0, sigma=5)
    w = pm.Normal('w', mu=0, sigma=5, shape=X_scaled_train.shape[1])
    sigma = pm.Uniform('sigma', lower=0, upper=20)

    y_est = w0 + pm.math.dot(X_scaled_train,w)

    likelihood = pm.Normal(
        'y',
        mu=y_est,
        sigma=sigma,
        observed=y_train
    )

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

# print(az.summary(
#     idata,
#     var_names=["w0", "w", "sigma"]
# ))
#
# az.plot_trace(
#     idata,
#     var_names=["w0", "w", "sigma"]
# )
#
# plt.tight_layout()
# plt.show()

w0_samples = idata.posterior["w0"].values.reshape(-1)

w_samples = idata.posterior["w"].values.reshape(
    -1, X_scaled_train.shape[1]
)

# Calculate predictions for every posterior draw
prediction_samples = (
        w0_samples[:, None]
        + w_samples @ X_scaled_val.T
)

# Average posterior predictions for each validation observation
y_pred = prediction_samples.mean(axis=0)

# Evaluate predictive performance
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, y_pred)

print("Bayesian polynomial validation MSE:", mse)
print("Bayesian polynomial validation RMSE:", rmse)
print("Bayesian polynomial validation MAE:", mae)

# Compare against a simple mean baseline
baseline_pred = np.full(len(y_val), y_train.mean())

baseline_mse = mean_squared_error(y_val, baseline_pred)

print("Mean baseline validation MSE:", baseline_mse)

# Sampling diagnostics
print(az.summary(idata, var_names=["w0", "w", "sigma"]))




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

# WHEN X = LINEAR but x2, x3 are poly with std = 20 and normal noise

# Sampling 4 chains for 1_000 tune and 2_000 draw iterations (4_000 + 8_000 draws total) took 1 seconds.
# mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean  mcse_sd
# w0      0.227  0.094    0.075     0.38    14690     6115  1.00   0.00078   0.0011
# w[0]    1.435  0.095      1.3      1.6    14135     6149  1.00    0.0008   0.0012
# w[1]   -0.123  0.094    -0.27     0.03    13049     5701  1.00   0.00083   0.0012
# w[2]    0.059  0.093   -0.091     0.21    13163     5815  1.00   0.00081   0.0012
# w[3]     0.79  0.095     0.64     0.94    13464     5575  1.00   0.00082   0.0013
# w[4]    0.017  0.095    -0.13     0.17    14591     6179  1.00   0.00079   0.0011
# w[5]   -0.711  0.094    -0.86    -0.56    13853     6373  1.00    0.0008   0.0013
# sigma   2.083  0.066        2      2.2    12600     5873  1.00   0.00059  0.00082

# DIFF FEATURE MAP BUT WITH SIGMA = 5 AND NORMAL NOISE
# Sampling 4 chains for 1_000 tune and 2_000 draw iterations (4_000 + 8_000 draws total) took 2 seconds.
# mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean  mcse_sd
# w0      0.228  0.093    0.081     0.37    13493     6246  1.00    0.0008   0.0011
# w[0]    1.432  0.093      1.3      1.6    13749     6399  1.00    0.0008   0.0011
# w[1]   -0.124  0.092    -0.27    0.022    12540     6083  1.00   0.00083   0.0011
# w[2]    0.058  0.092   -0.087      0.2    13843     5975  1.00   0.00078   0.0011
# w[3]     0.79  0.095     0.64     0.94    14251     6047  1.00   0.00079   0.0011
# w[4]     0.02  0.094    -0.13     0.16    14033     6437  1.00   0.00079   0.0012
# w[5]   -0.711  0.094    -0.86    -0.56    13123     6394  1.00   0.00082   0.0012
# sigma   2.083  0.066        2      2.2    14212     6394  1.00   0.00055  0.00078

#dIFF feature map but with sigma - 1 an dnormal moise
# Sampling 4 chains for 1_000 tune and 2_000 draw iterations (4_000 + 8_000 draws total) took 1 seconds.
# # mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean  mcse_sd
# # w0      0.226  0.093    0.078     0.37    12773     6058  1.00   0.00082   0.0012
# # w[0]    1.422  0.095      1.3      1.6    13794     6229  1.00   0.00081   0.0011
# # w[1]   -0.121  0.093    -0.27    0.028    14060     6194  1.00   0.00079   0.0012
# # w[2]    0.059  0.094   -0.089     0.21    15523     5852  1.00   0.00075   0.0012
# # w[3]    0.784  0.093     0.64     0.93    13874     5861  1.00   0.00079   0.0012
# # w[4]    0.019  0.094    -0.13     0.17    14271     6411  1.00   0.00079   0.0011
# # w[5]   -0.706  0.095    -0.86    -0.55    13147     5922  1.00   0.00083   0.0012
# # sigma   2.084  0.068        2      2.2    14426     6128  1.00   0.00057  0.00085


### AFTER TEST TRAIN, POLY X2 STD = 1

# Sampling 4 chains for 1_000 tune and 2_000 draw iterations (4_000 + 8_000 draws total) took 1 seconds.
# Bayesian polynomial validation MSE: 5.009128917176883
# Bayesian polynomial validation RMSE: 2.238108334548818
# Bayesian polynomial validation MAE: 1.788313822286213
# Mean baseline validation MSE: 6.4636929124677325
# mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean  mcse_sd
# w0      0.335  0.102     0.17      0.5    13607     6147  1.00   0.00089   0.0013
# w[0]    1.484  0.101      1.3      1.6    14197     6855  1.00   0.00085   0.0012
# w[1]    -0.04  0.102    -0.21     0.12    15633     5945  1.00   0.00082   0.0013
# w[2]     0.04  0.104    -0.12     0.21    15528     5950  1.00   0.00083   0.0013
# w[3]    0.841  0.104     0.67        1    14388     5771  1.00   0.00087   0.0013
# w[4]    0.047  0.104    -0.12     0.21    15936     6419  1.00   0.00082   0.0012
# w[5]   -0.779  0.102    -0.94    -0.62    14839     6389  1.00   0.00084   0.0013
# sigma   2.054  0.072      1.9      2.2    13330     6823  1.00   0.00063  0.00083

# after trest train std = 5
# Sampling 4 chains for 1_000 tune and 2_000 draw iterations (4_000 + 8_000 draws total) took 2 seconds.
# Bayesian polynomial validation MSE: 5.031596639322178
# Bayesian polynomial validation RMSE: 2.24312207410167
# Bayesian polynomial validation MAE: 1.792540381155704
# Mean baseline validation MSE: 6.4636929124677325
# mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean  mcse_sd
# w0      0.338  0.104     0.17     0.51    13969     6181  1.00   0.00088   0.0013
# w[0]    1.499  0.103      1.3      1.7    13070     6250  1.00    0.0009   0.0012
# w[1]   -0.042  0.104    -0.21     0.13    11365     5531  1.00   0.00098   0.0013
# w[2]    0.041  0.101    -0.12      0.2    13836     5380  1.00   0.00086   0.0013
# w[3]    0.849  0.104     0.68        1    14838     6619  1.00   0.00085   0.0012
# w[4]    0.046  0.103    -0.12     0.21    14829     6664  1.00   0.00085   0.0012
# w[5]   -0.784  0.105    -0.95    -0.62    15649     6756  1.00   0.00084   0.0013
# sigma   2.054  0.073      1.9      2.2    11822     6170  1.00   0.00067  0.00089


# # after test train std = 20
# Sampling 4 chains for 1_000 tune and 2_000 draw iterations (4_000 + 8_000 draws total) took 2 seconds.
# Bayesian polynomial validation MSE: 5.036576552276826
# Bayesian polynomial validation RMSE: 2.2442318401352446
# Bayesian polynomial validation MAE: 1.7934931903608442
# Mean baseline validation MSE: 6.4636929124677325
# mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean  mcse_sd
# w0      0.341  0.101     0.18      0.5    13410     6001  1.00   0.00087   0.0012
# w[0]    1.499  0.103      1.3      1.7    13962     6650  1.00   0.00088   0.0013
# w[1]   -0.041  0.104    -0.21     0.12    14314     5951  1.00   0.00087   0.0015
# w[2]    0.043  0.101    -0.12      0.2    13962     6403  1.00   0.00086   0.0012
# w[3]    0.849  0.105     0.68        1    13279     6183  1.00   0.00091   0.0012
# w[4]    0.046  0.106    -0.12     0.22    13555     6409  1.00   0.00091   0.0013
# w[5]   -0.784  0.102    -0.95    -0.62    14272     6126  1.00   0.00086   0.0012
# sigma   2.054  0.073      1.9      2.2    13698     6097  1.00   0.00063  0.00094