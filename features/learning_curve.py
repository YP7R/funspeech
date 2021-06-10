# Ici sont toutes les courbes d'apprentissage
# ... own_learning_curve_with_params
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import learning_curve, StratifiedShuffleSplit, cross_validate, StratifiedKFold
from sklearn import clone
from sklearn.metrics import get_scorer
import numpy as np


def default_learning_curve(estimator, features, sounds_expected):
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(estimator=estimator, X=features,
                                                                                    y=sounds_expected,
                                                                                    train_sizes=np.linspace(0.1, 1, 10),
                                                                                    cv=10,
                                                                                    n_jobs=-1, shuffle=True,
                                                                                    random_state=0,
                                                                                    return_times=True)


# List of metrics
# print(sorted(sklearn.metrics.SCORERS.keys()))
def new_learning_curve(clf, x, y, x_test, y_test, train_size=np.linspace(0.1, 1.0, 10, dtype=np.float),
                       scoring=['accuracy', 'neg_log_loss']):
    train_size = np.round(train_size, 2)
    test_size = np.round(1 - train_size, 2)
    n_split = 5

    # Results
    results = {
        'train_sizes': np.trunc(train_size[:-1] * y.size)
        'fit_time': [],
        'score_time': [],
    }

    for score in scoring:
        results[f"train_{score}"] = []
        results[f"test_{score}"] = []
        results[f"validation_{score}"] = []

    # Compute
    for tr, te in zip(train_size[:-1], test_size[:-1]):
        print("*" * 100)
        classifier = clone(clf)

        stratified_shuffle_split = StratifiedShuffleSplit(random_state=0, train_size=tr, test_size=te, n_splits=n_split)
        # sss = StratifiedShuffleSplit(random_state=0, train_size=tr, test_size=te, n_splits=10)
        # part_1, part_2 = next(sss.split(x, y))
        # this is used by default # skf = StratifiedKFold(10,shuffle=True, random_state=0) # default

        # (Train/Test) Cross-validation
        cross_validation = cross_validate(classifier, x, y, cv=stratified_shuffle_split,
                                          scoring=scoring, n_jobs=-1,
                                          return_train_score=True, return_estimator=True)
        # Add cross_validation results
        for key, cv in cross_validation.items():
            if key == 'estimator':
                continue
            results[key].append(cv)

        # (Validations) on the other dataset
        estimators = cross_validation['estimator']
        validations = {}
        for score in scoring:
            validations[score] = []

        for estimator in estimators:
            y_prediction = estimator.predict(x_test)
            if 'neg_log_loss' in scoring:
                y_prediction_probability = estimator.predict_proba(x_test)

            for score in scoring:
                if score == 'neg_log_loss':
                    score_metric = get_scorer(score)._score_func(y_test, y_prediction_probability)
                else:
                    score_metric = get_scorer(score)._score_func(y_test, y_prediction)
                validations[score].append(score_metric)

        for score in scoring:
            results[f"validation_{score}"].append(validations[score])

    # transform to numpy array
    for key, item in results.items():
        results[key] = np.array(item)

    return results
