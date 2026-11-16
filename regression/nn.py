import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.metrics import mean_squared_error
import h5py

from machineLearningCoursework.classification.classification_dummy import X_test

training_data = np.loadtxt("regression_train.txt")
X = training_data[:, :3] # Keeps x1,x2,x3 (n rows, 3 columns)
y = training_data[:, 3] # (n rows, 1 columns)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scalar.transform(X_val)

#CAN ADD LEARNIGN RATE ETC here
regressor = MLPRegressor(random_state = 0.42,
                         hidden_layer_sizes = (64,32),
                         activation = 'relu',
                         solver = 'sgd',
                         alpha = 0,
                         max_iter = 1000
                         )

regressor.fit(X_train, y_train)

val_preds = regressor.predict(X_val_scaled)
val_mse = mean_squared_error(y_val, val_preds)
print(f"Validation MSE: {val_mse:.4f}")

from sklearn.metrics import ConfusionMatrixDisplay, mean_squared_error

print(regressor.score(X_train, y_train))
print(regressor.score(X_val, y_val))
ConfusionMatrixDisplay.from_estimator(regressor, X_val, y_val.flatten())