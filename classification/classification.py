import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
