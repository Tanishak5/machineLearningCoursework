# take in input data
# split data into training / validation sets
# How should I split the set, what linear regression model to use etc.
# test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.preprocessing import StandardScaler

# from skimage import io
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# Check if there are any outliers,
# Check how much data
# will have to use multivariable inear regression
# standard scaler
# choose test/train split
# fit then train model

import pandas as pd
from sklearn.preprocessing import StandardScaler

training_data = np.loadtxt("regression_train.txt")
X = training_data[:, :3] # Keeps x1,x2,x3 (n rows, 3 columns)
y = training_data[:, 3] # (n rows, 1 columns)
dataframe = pd.DataFrame(training_data, columns=['x1', 'x2', 'x3', 'y'])
# plot = sns.pairplot(dataframe)
# plt.show()

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))


#
#
#
# scaler = StandardScaler().
# X_scaled = scaler.transform(X)
# print(X_scaled.mean(axis=0))
# print(X_scaled.std(axis=0))

# Then compute the test/train split
# Run model
# why the chosen model
# Do i need penalties ?