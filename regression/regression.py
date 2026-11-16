
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Check if there are any outliers,
# Check how much data
# will have to use multivariable inear regression
# standard scaler
# choose test/train split
# fit then train model

training_data = np.loadtxt("regression_train.txt")
X = training_data[:, :3] # Keeps x1,x2,x3 (n rows, 3 columns)
y = training_data[:, 3] # (n rows, 1 columns)
dataframe = pd.DataFrame(training_data, columns=['x1', 'x2', 'x3', 'y'])
# plot = sns.pairplot(dataframe)
# plt.show()

poly = PolynomialFeatures(degree=2, include_bias=False)




X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_poly = poly.fit_transform(X_train[:, 1:])
X_val_poly = poly.transform(X_val[:, 1:])

X_train_modified = np.column_stack((X_train[:, 0], X_train_poly))
X_val_modified = np.column_stack((X_val[:, 0], X_val_poly))




scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_modified)
X_val_scaled = scaler.transform(X_val_modified)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
print(lin_reg.coef_)

MSE_train = mean_squared_error(y_train, lin_reg.predict(X_train))
print('Train MSE: ', round(MSE_train, 2))
MSE_val = mean_squared_error(y_val, lin_reg.predict(X_val))
print('val MSE: ', round(MSE_val, 2))

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y_val, lin_reg.predict(X_val))
ax.plot([0,5],[0,5])
ax.set_xlabel('Real value')
ax.set_ylabel('Prediction')
