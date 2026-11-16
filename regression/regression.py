
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
print(lin_reg.coef_)

RMSE_train = np.sqrt(mean_squared_error(y_train, lin_reg.predict(X_train)))
print('Train RMSE: ', round(RMSE_train,2))
RMSE_test = np.sqrt(mean_squared_error(y_val, lin_reg.predict(X_val)))
print('Test RMSE: ', round(RMSE_test,2))

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y_val, lin_reg.predict(X_val))
ax.plot([0,5],[0,5])
ax.set_xlabel('Real value')
ax.set_ylabel('Prediction')
