# Ici sont toutes les courbes d'apprentissage
# ... own_learning_curve_with_params
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import learning_curve, StratifiedShuffleSplit, cross_validate, StratifiedKFold
from sklearn import clone
import numpy as np


def default_learning_curve(estimator, features, sounds_expected):
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(estimator=estimator, X=features,
                                                                                    y=sounds_expected,
                                                                                    train_sizes=np.linspace(0.1, 1, 10),
                                                                                    cv=10,
                                                                                    n_jobs=-1, shuffle=True,
                                                                                    random_state=0,
                                                                                    return_times=True)


def new_learning_curve(clf, x, y, x_test, y_test, train_size=np.linspace(0.1, 1.0, 10, dtype=np.float)):
    train_size = np.round(train_size, 2)
    test_size = np.round(1 - train_size, 2)
    for tr, te in zip(train_size[:-1], test_size[:-1]):
        classifier = clone(clf)

        stratified_shuffle_split = StratifiedShuffleSplit(random_state=0, train_size=tr, test_size=te, n_splits=5)
        # sss = StratifiedShuffleSplit(random_state=0, train_size=tr, test_size=te, n_splits=10)
        # part_1, part_2 = next(sss.split(x, y))
        # this is used by default # skf = StratifiedKFold(10,shuffle=True, random_state=0) # default

        cross_validation = cross_validate(classifier, x, y, cv=stratified_shuffle_split,
                                          scoring=['accuracy'], n_jobs=-1,
                                          return_train_score=True, return_estimator=True)
        print("*" * 100)
        print(cross_validation.items())
        estimators = cross_validation['estimator']
        for estimator in estimators:
            y_pred = estimator.predict(x_test)
            y_pred_proba = estimator.predict_proba(x_test)
            score_accuracy = accuracy_score(y_test, y_pred)
            score_log_loss = log_loss(y_test, y_pred_proba)
            print(score_accuracy, score_log_loss)
        '''
        for c in cross_validation.items():
            print(c)
        '''
        # print(cross_validation)
