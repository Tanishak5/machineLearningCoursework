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

    w0 = pm.Normal('w0', mu=0, sigma=20)
    w = pm.Normal('w', mu=0, sigma=20, shape=X_scaled_train.shape[1])
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




