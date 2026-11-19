all_models = []
num_models = 10
sample_size = len(training_data)
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

models = {
    "bagging" : BaggingClassifier(
        estimator = DecisionTreeClassifier(max_depth=2),
        n_estimators = 100,
        bootstrap = True,
        random_state=42
    ),
    "random forest": RandomForestClassifier(
        max_depth=2,
        n_estimators = 100,
        random_state=42,
    ),
    "adaboost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    ),

}

for name, model in models.items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print(f"model: {name}, Mean accuracy: {scores.mean():.4f}, std: {scores.std():.4f}")


#WHEN DEPTH IS UNRESTRICTED FOR BAGGING AND RANDOM FOREST BUT SET TO 1 FOR ADABOOST
# model: bagging, Mean accuracy: 0.6880, std: 0.0223
# model: random forest, Mean accuracy: 0.6900, std: 0.0228
# model: adaboost, Mean accuracy: 0.7580, std: 0.0504
# model: gradient boosting, Mean accuracy: 0.7480, std: 0.0581


#WHEN DEPTH IS2  FOR BAGGING AND RANDOM FOREST BUT SET TO 1 FOR ADABOOST
# model: bagging, Mean accuracy: 0.6880, std: 0.0223
# model: random forest, Mean accuracy: 0.6900, std: 0.0228
# model: adaboost, Mean accuracy: 0.7580, std: 0.0504
# model: gradient boosting, Mean accuracy: 0.7480, std: 0.0581

#Things to check for random forest
random_forest_parameters = {
    "depth" : [2, 3, 5, None],
    "min_samples_leaf" : [1,3,5],
    "max_features" : [1,2,3],
    "random_forest_estimators":  [10, 20, 50, 100, 200],

}

for d in random_forest_parameters["depth"]:
    model = RandomForestClassifier(max_depth = d)
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )
    print(f" RF depth: {d}, Mean accuracy: {scores.mean():.4f}, std: {scores.std():.4f}")

for d in random_forest_parameters["random_forest_estimators"]:
    model = RandomForestClassifier(max_depth=2, n_estimators = d)
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )
    print(f" RF depth = 2 estimators num: {d}, Mean accuracy: {scores.mean():.4f}, std: {scores.std():.4f}")

for d in random_forest_parameters["random_forest_estimators"]:
    model = RandomForestClassifier(max_depth=3, n_estimators = d)
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )
    print(f" RF depth = 3 estimators num: {d}, Mean accuracy: {scores.mean():.4f}, std: {scores.std():.4f}")

#fix depth to 2, and the estimators to 100, 200
for d in random_forest_parameters["min_samples_leaf"]:
    for est in [100, 200]:
        model = RandomForestClassifier(max_depth=2, n_estimators = est, min_samples_leaf=d)
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy"
        )
        print(
            f" RF depth = 2 estimators num of estimators: {est},"
            f" min_samples_leaf: {d},"
            f" Mean accuracy: {scores.mean():.4f},"
            f" std: {scores.std():.4f}"
        )

#ot fixing sample leaf at all just geatures
for d in random_forest_parameters["max_features"]:
    for est in [100, 200]:
        model = RandomForestClassifier(max_depth=2, n_estimators = est, max_features=d)
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy"
        )
        print(
            f" RF depth = 2 estimators num: {est},"
            f" max_features: {d},"
            f" Mean accuracy: {scores.mean():.4f},"
            f" std: {scores.std():.4f}"
        )


#PRETTY SURE BEST DEPTH SHOULD BE 2
# for d in random_forest_parameters["depth"]:
#     model = models["random forest"]
#     scores = cross_val_score(
#         model,
#         X,
#         y,
#         cv=cv,
#         scoring="accuracy"
#     )
#     print(f" RF depth: {d}, Mean accuracy: {scores.mean():.4f}, std: {scores.std():.4f}")
#
#



#Adaboost
adaboost_parameters = {
    "max_depth" : [1,2],
    "learning_rate" : [0.1, 0.5, 1.0],
    "max_features" : [1,2,3],
    "adaboost_forest_estimators" : [10, 20, 50, 100, 200],

}

for d in adaboost_parameters["adaboost_forest_estimators"]:
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators = d)
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )
    print(f" adaboost max_depth = 1 estimators num: {d}, Mean accuracy: {scores.mean():.4f}, std: {scores.std():.4f}")


for d in adaboost_parameters["max_depth"]:
    model = AdaBoostClassifier(
        estimator = DecisionTreeClassifier(max_depth=d),
        n_estimators = 100
    )
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )
    print(f"ADA Estimators fixed to 100, depth: {d}, Mean accuracy: {scores.mean():.4f}, std: {scores.std():.4f}")

#Bagging

bagging_parameters = {
    "bagging_estimators" : [10,20,50,100,200],
    "bagging_depth" : [2, 3, 5, None],
    "min_samples_leaf": [1,3,5]

}


for d in bagging_parameters["bagging_estimators"]:
    model = BaggingClassifier(
        estimator = DecisionTreeClassifier(max_depth=2),
        n_estimators = d
    )
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )
    print(f" bagging d = 2, estimators num: {d}, Mean accuracy: {scores.mean():.4f}, std: {scores.std():.4f}")

for depth in bagging_parameters["bagging_depth"]:

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model = BaggingClassifier(
        estimator=tree,
        random_state=42
    )

    scores = cross_val_score(
        model, X, y, cv=cv, scoring="accuracy"
    )

    print(f"Bagging depth={depth}, Mean: {scores.mean():.4f}, "
          f"std: {scores.std():.4f}")
best_estimators = 100  # Replace with actual best result


for leaf in bagging_parameters["min_samples_leaf"]:

    tree = DecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=leaf,
        random_state=42
    )

    model = BaggingClassifier(
        estimator=tree,
        n_estimators=best_estimators,
        random_state=42
    )

    scores = cross_val_score(
        model, X, y, cv=cv, scoring="accuracy"
    )

    print(f"Bagging leaf={leaf}, "
          f"Mean: {scores.mean():.4f}, std: {scores.std():.4f}")


models = {
    'rf': RandomForestClassifier(
        max_depth = 2,
        n_estimators = 100,
        random_state=4,
    ),
    'abd=1e=100': AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
    ),
    'ab d=1e=50': AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
    ),
    'ab d=2e=50': AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators=50,
    ),
    'ab d=2e=100': AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators=100,
    ),
    'Baggingd=1e=20': BaggingClassifier(
        DecisionTreeClassifier(max_depth=1),
        n_estimators = 20,
    ),
    'Baggingd=1e=50': BaggingClassifier(
        DecisionTreeClassifier(max_depth=1),
        n_estimators = 50,
    ),
    'Baggingd=1e=100': BaggingClassifier(
        DecisionTreeClassifier(max_depth=1),
        n_estimators = 100,
    ),
    'Baggingd=2e=50': BaggingClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators = 50,
    ),
    'Baggingd=2e=100': BaggingClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators = 100,
    )

}

for name, model in models.items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print(f"model: {name}, Mean accuracy: {scores.mean():.4f}, std: {scores.std():.4f}")


