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

model = BaggingClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100)
scores = cross_val_score(model,
    X,
    y,
    cv = cv,
    scoring="accuracy"
)


print(f"Mean accuracy: {scores.mean():.4f}")
print(f"Std: {scores.std():.4f}")

# Generate predictions for every observation
y_pred = cross_val_predict(
    model,
    X,
    y,
    cv=cv,)

# Overall accuracy from cross-validated predictions
print("Overall accuracy:", accuracy_score(y, y_pred))

# Confusion matrix
print(confusion_matrix(y, y_pred))

# Classification report
print(classification_report(y, y_pred))


# Create confusion matrix
cm = confusion_matrix(y, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(cmap="Blues")

plt.title("Random Forest - Confusion Matrix")
plt.show()
# rf d=2e=100
# Mean accuracy: 0.7600
# Std: 0.0506
# Overall accuracy: 0.76
# [[248  34]
#  [ 86 132]]
# precision    recall  f1-score   support
#
# 0       0.74      0.88      0.81       282
# 1       0.80      0.61      0.69       218
#
# accuracy                           0.76       500
# macro avg       0.77      0.74      0.75       500
# weighted avg       0.77      0.76      0.75       500


#ADABOOST lr = 1 d = 2 100 est
# Mean accuracy: 0.7540
# Std: 0.0531
# Overall accuracy: 0.754
# [[247  35]
#  [ 88 130]]
# precision    recall  f1-score   support
#
# 0       0.74      0.88      0.80       282
# 1       0.79      0.60      0.68       218
#
# accuracy                           0.75       500
# macro avg       0.76      0.74      0.74       500
# weighted avg       0.76      0.75      0.75       500
#HOW TO READ CONFUSION MATRIX
#00 = TRUE NEGATIVE, 01 (TOP RIGHT) = FALSE POSTIVE
#10 = FALSE NEGATIVE 11 TRUE POSTIVE

#BAGGING d = 2, est = 100, lr = default
# Mean accuracy: 0.7560
# Std: 0.0508
# Overall accuracy: 0.756
# [[249  33]
#  [ 89 129]]
# precision    recall  f1-score   support
#
# 0       0.74      0.88      0.80       282
# 1       0.80      0.59      0.68       218
#
# accuracy                           0.76       500
# macro avg       0.77      0.74      0.74       500
# weighted avg       0.76      0.76      0.75       500