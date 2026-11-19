import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.compose import ColumnTransformer



training_data = np.loadtxt("regression_train.txt")
X = training_data[:, :3]
y = training_data[:, 3]

k_folds = KFold(n_splits = 5, shuffle = True, random_state=40)

preprocessor = ColumnTransformer(
    transformers = [
        ("linear", "passthrough", [0]),
        ("polynomial", PolynomialFeatures(degree=2, include_bias = False), [1,2])
    ]
)

model = make_pipeline(
    preprocessor,
    StandardScaler(),
    LinearRegression()
    # RidgeCV(
    #     alphas=[0.001]
    # )
)

scores = cross_val_score(
    model,
    X,
    y,
    cv = k_folds,
    scoring = 'neg_mean_squared_error'
)
print("Mean CV MSE:", round(-scores.mean(), 2))
print("Std CV MSE:", round(scores.std(), 2))

