import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt




training_data = np.loadtxt("classification_train.txt")
# df = pd.DataFrame(training_data, columns = ["x1", "x2", "x3", "y"])
# # print(df.corr())
#
# # sns.pairplot(
# #     df,
# #     vars = ["x1", "x2", "x3"],
# #     hue="y",
# #     diag_kind="hist"
# # )
# # plt.show()
# #
# # sns.pairplot(
# #     df,
# #     vars=["x1", "x2", "x3"],
# #     hue="y"
# # )
# # plt.show()
#
# #X1, X3 = POSITIVE CORR
# #x2, x3 = Negative corr
# #x1, x2 = weaker but still positive
#
# X = training_data[:,:-1]
# y = [int(z) for z in training_data[:,-1]]
# #CROSS validation with same num per class
#
# cv = StratifiedKFold(
#     n_splits = 5,
#     shuffle = True,
#     random_state=42
# )
#
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42
# )
#
# #Consider the colinear rel of x1,x2,x3
# #experiement with regularisation
# #how many per class
#
# # make precision, recall, conufsion matrix
#
# model = make_pipeline(
#     StandardScaler(),
#     LogisticRegression(C = 0.01, max_iter=1000)
# )
#
# # model.fit(X_train, y_train)
# # y_pred = model.predict(X_val)
#
# # accuracy = accuracy_score(y_val, y_pred)
# # print("Test accuracy:", accuracy)
#
# scores = cross_val_score(
#     model,
#     X,
#     y,
#     cv=cv,
#     scoring="accuracy"
# )
#
# print("Fold accuracies:", scores)
# print("Mean accuracy:", scores.mean())
# print("Standard deviation:", scores.std())



#DECISION TREE

# X = training_data[:,:-1]
# y = [int(z) for z in training_data[:,-1]]
# #CROSS validation with same num per class
#
# cv = StratifiedKFold(
#     n_splits = 5,
#     shuffle = True,
#     random_state=42
# )
#
#
# model = make_pipeline(
#     DecisionTreeClassifier(
#         criterion='entropy',
#         max_depth=2,
#         min_samples_split=2,
#         min_samples_leaf=1,
#         random_state=42)
# )
#
# scores = cross_val_score(
#     model,
#     X,
#     y,
#     cv=cv,
#     scoring="accuracy"
# )
#
# print(
#
#     f"Mean={scores.mean():.3f}, "
#     f"Std={scores.std():.3f}"
# )
#


#FINAL depth = 2 Mean=0.756, Std=0.051 only

#Mean=0.756, Std=0.051 when you add the split = 2 leaf = 1


# X = training_data[:, :-1]
# y = training_data[:, -1].astype(int)
#
# cv = StratifiedKFold(
#     n_splits=5,
#     shuffle=True,
#     random_state=42
# )
#
#
# for gamma in ["scale", 0.001, 0.01, 0.1, 1]:
#     model = make_pipeline(
#         StandardScaler(),
#         SVC(kernel="rbf", C=1, gamma = gamma)
#     )
#
#     scores = cross_val_score(
#         model, X, y,
#         cv=cv,
#         scoring="accuracy"
#     )
#
#     print("gamma", gamma)
#     print("Fold accuracies:", scores)
#     print("Mean accuracy:", scores.mean())
#     print("Standard deviation:", scores.std())
#
#

#Type of ensemble method
"""
bagging - TRAINING muliple models - on bootstrapped samples
random forest - trains on bootstrapped samples and on bootstrapped features
adaboost - corrects on previoys trees errors, will use the entire dataset pretty sure

run through models
choose 2/3
then tune 
then pick
willprobs just be the randomforest 

"""
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

#BAGGING
X = training_data[:,:-1]
y = [int(z) for z in training_data[:,-1]]


cv = StratifiedKFold(
    n_splits = 5,
    shuffle = True,
    random_state=42
)

# model = RandomForestClassifier(max_depth=2, n_estimators=100)
# model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100, learning_rate=0.1)

random_forest_parameters = {
    "max_depth" : [2, 3, 5, None],
    "min_samples_leaf" : [1,3,5],
    "min_samples_split": [2, 5, 10],
    "max_features" : [1,2,3],
    "n_estimators":  [10, 20, 50, 100, 200],
    "criterion": ['gini', 'cross-entropy', 'log_loss'],
    "bootstrap": [True, False]

}

adaboost_parameters = {
    'learning_rate': [
        0.001, 0.005, 0.01, 0.025,
        0.05, 0.1, 0.25, 0.5, 0.75, 1.0
    ],
    "estimator__max_depth" : [1,2,3],
    "n_estimators" : [10, 20, 50, 100, 200],

}
bagging_parameters = {
    "n_estimators" : [10,20,50,100,200],
    "estimator__max_depth" : [2, 3, 5, None],
    'estimator__min_samples_split': [0.5, 0.75, 1.0],
    "estimator__min_samples_leaf": [1,3,5]

}

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    random_state=42,
    n_jobs=-1
)


ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    random_state=42
)


models = {
    'Bagging': (bagging, bagging_parameters),
    'AdaBoost': (ada, adaboost_parameters),
    'Random Forest': (rf, random_forest_parameters)
}



for name, (model, params) in models.items():

    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X,y)

    # Store best model
    best_model = grid.best_estimator_

    print("\nBest parameters:", grid.best_params_)
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    # Cross-validation scores
    scores = cross_val_score(
        best_model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Std: {scores.std():.4f}")

    # Cross-validated predictions
    y_pred = cross_val_predict(
        best_model,
        X,
        y,
        cv=cv
    )

    # Overall accuracy
    print("Overall accuracy:", accuracy_score(y, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(cmap="Blues")

    plt.title(f"{name} - Confusion Matrix")
    plt.show()
















#
#
# model = BaggingClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100)
# scores = cross_val_score(model,
#     X,
#     y,
#     cv = cv,
#     scoring="accuracy"
# )
#
#
# print(f"Mean accuracy: {scores.mean():.4f}")
# print(f"Std: {scores.std():.4f}")
#
# # Generate predictions for every observation
# y_pred = cross_val_predict(
#     model,
#     X,
#     y,
#     cv=cv,)
#
# # Overall accuracy from cross-validated predictions
# print("Overall accuracy:", accuracy_score(y, y_pred))
#
# # Confusion matrix
# print(confusion_matrix(y, y_pred))
#
# # Classification report
# print(classification_report(y, y_pred))
#
#
# # Create confusion matrix
# cm = confusion_matrix(y, y_pred)
#
# # Display confusion matrix
# disp = ConfusionMatrixDisplay(
#     confusion_matrix=cm
# )
#
# disp.plot(cmap="Blues")
#
# plt.title("Random Forest - Confusion Matrix")
# plt.show()
# rf d=2e=100


# Best parameters: {'estimator__max_depth': 2, 'estimator__min_samples_leaf': 1, 'estimator__min_samples_split': 0.5, 'n_estimators': 10}
# Best CV accuracy: 0.7600
# Mean accuracy: 0.7600
# Std: 0.0506
# Overall accuracy: 0.76
#
# Confusion Matrix:
# [[248  34]
#  [ 86 132]]
#
# Classification Report:
# precision    recall  f1-score   support
#
# 0       0.74      0.88      0.81       282
# 1       0.80      0.61      0.69       218
#
# accuracy                           0.76       500
# macro avg       0.77      0.74      0.75       500
# weighted avg       0.77      0.76      0.75       500
#
# Qt: Session management error: None of the authentication protocols specified are supported
# Fitting 5 folds for each of 150 candidates, totalling 750 fits
#
# Best parameters: {'estimator__max_depth': 1, 'learning_rate': 0.001, 'n_estimators': 10}
# Best CV accuracy: 0.7600
# Mean accuracy: 0.7600
# Std: 0.0506
# Overall accuracy: 0.76
#
# Confusion Matrix:
# [[248  34]
#  [ 86 132]]
#
# Classification Report:
# precision    recall  f1-score   support
#
# 0       0.74      0.88      0.81       282
# 1       0.80      0.61      0.69       218
#
# accuracy                           0.76       500
# macro avg       0.77      0.74      0.75       500
# weighted avg       0.77      0.76      0.75       500
#
#
# Best parameters: {'bootstrap': True, 'criterion': 'log_loss', 'max_depth': 3, 'max_features': 3, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
# Best CV accuracy: 0.7620
# Mean accuracy: 0.7620
# Std: 0.0511
# Overall accuracy: 0.762
#
# Confusion Matrix:
# [[249  33]
#  [ 86 132]]
#
# Classification Report:
# precision    recall  f1-score   support
#
# 0       0.74      0.88      0.81       282
# 1       0.80      0.61      0.69       218
#
# accuracy                           0.76       500
# macro avg       0.77      0.74      0.75       500
# weighted avg       0.77      0.76      0.76       500