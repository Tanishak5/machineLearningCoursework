from tkinter.constants import RIDGE

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
#ridge = l2
#lasso = l1
#The mean for the baseline, predicting only from the y mean is 7.727


training_data = np.loadtxt("regression_train.txt")
X = training_data[:, :3] # Keeps x1,x2,x3 (n rows, 3 columns)
y = training_data[:, 3] # (n rows, 1 columns)
dataframe = pd.DataFrame(training_data, columns=['x1', 'x2', 'x3', 'y'])
k_folds = KFold(n_splits = 5, shuffle = True, random_state=40)
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=10
# )

preprocessor = ColumnTransformer(
    transformers = [
        ("linear", "passthrough", [0]),
        ("polynomial", PolynomialFeatures(degree=2, include_bias = False), [1,2])
    ]
)

model = make_pipeline(
    preprocessor,
    StandardScaler(),
    # LinearRegression()
    RidgeCV(
        alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    )
)
# model.fit(X_train, y_train)
# preds = model.predict(X_val)
# mse = mean_squared_error(y_val, preds)
#
# print(f"Random state {10}: MSE = {mse:.4f}")
#
scores = cross_val_score(
    model,
    X,
    y,
    cv = k_folds,
    scoring = 'neg_mean_squared_error'
)
#
print("Mean CV MSE:", round(-scores.mean(), 2))
print("Std CV MSE:", round(scores.std(), 2))


