import sys
import os
import glob
import shutil
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from features.barycentre import BaryCentre
from sklearn import clone
import features.learning_curve as lc
import features.plot as pl

import pandas as pd

csv_directories = ".\\dataset\\processed\\"
features_directories = ".\\dataset\\features\\"
result_directory = ".\\dataset\\results\\"

if not os.path.exists(csv_directories):
    print("CSV directories doesn't exist")
    exit(0)

shutil.rmtree(result_directory, ignore_errors=True)
os.makedirs(result_directory)

csv_files = glob.glob(f"{csv_directories}" + "**/*.csv", recursive=True)

classifiers = [{'clf_name': 'Barycentre', 'clf': BaryCentre(),
                'param_grid': [],
                'scoring': ['accuracy']},
               {'clf_name': 'SVM', 'clf': SVC(probability=True, random_state=0, kernel='linear', gamma='scale'),
                'param_grid': [{'C': np.arange(0.95, 1.01, 0.01)}],
                'scoring': ['accuracy', 'neg_log_loss']},
               {'clf_name': 'KNN', 'clf': KNeighborsClassifier(weights='distance'),
                'param_grid': [{'n_neighbors': np.arange(9, 18, 4)}],
                'scoring': ['accuracy', 'neg_log_loss']
                },
               {'clf_name': 'Bagging_SVM',
                'clf': BaggingClassifier(SVC(probability=True, random_state=0, kernel='linear', gamma='scale')),
                'param_grid': [],
                'scoring': ['accuracy', 'neg_log_loss']},
               {'clf_name': 'Bagging_KNN',
                'clf': BaggingClassifier(KNeighborsClassifier(weights='distance', n_neighbors=13)),
                'param_grid': [],
                'scoring': ['accuracy', 'neg_log_loss']}]

# Pour les datasets
for csv_file in csv_files:
    csv_name, ext = os.path.splitext(os.path.basename(csv_file))
    feature_files = glob.glob(f"{features_directories}{csv_name}" + "**/*.npy", recursive=True)

    output_directory = f"{result_directory}{csv_name}\\"
    shutil.rmtree(output_directory, ignore_errors=True)
    os.makedirs(output_directory)
    print(f" {csv_name.upper()} ".center(100, "*"))

    dataframe = pd.read_csv(csv_file)
    # Pour les features
    for feature_file in feature_files:
        feature_name, ext = os.path.splitext(os.path.basename(feature_file))
        print(f" {feature_name.upper()} ".center(50, "-"))

        # Loading features (X), reshape and minmax scaler
        features = np.load(feature_file)
        features = np.reshape(features, (features.shape[0], -1))
        minmax_scaler = MinMaxScaler()
        features = minmax_scaler.fit_transform(features)

        # Loading expected values (Y)
        sounds_expected = dataframe['sound_id'].values
        label_encoder = LabelEncoder()
        sounds_expected = label_encoder.fit_transform(sounds_expected)

        # Split Train/Test & validation sets
        if csv_name == "funspeech":
            df_child = dataframe.loc[dataframe['base_file'].str.startswith('user4')]
            df_adult = dataframe.drop(df_child.index)
        else:
            df_adult = dataframe.loc[dataframe['category'].isin(["men", "women"])]
            df_child = dataframe.drop(df_adult.index)

        features_child = features[df_child.index.values]
        features_adult = features[df_adult.index.values]
        sounds_child = sounds_expected[df_child.index.values]
        sounds_adult = sounds_expected[df_adult.index.values]

        # Pour les classifieurs
        for classifier in classifiers[0:3]:
            clf = clone(classifier['clf'])
            clf_name = classifier['clf_name']

            clf_scoring = classifier['scoring']
            # Param grid
            clf_param_grid = classifier['param_grid']
            if clf_param_grid:
                grid = GridSearchCV(clf, cv=5, param_grid=clf_param_grid, n_jobs=-1)
                grid.fit(features_adult, sounds_adult)
                clf.set_params(**grid.best_params_)

            # Call learning curve
            result_learning_curve = lc.new_learning_curve(clf, features_adult, sounds_adult, features_child,
                                                          sounds_child, scoring=clf_scoring)
            save_path = f"{output_directory}{csv_name}_{feature_name}_{clf_name}.png"
            # Call plot learning curve
            pl.plot_curves(result_learning_curve, clf_scoring, feature_name, clf_name, csv_name, save_path)


'''
# template
# for csv in csvs:
    # for npy in npy:
        # for clf in clfs:
        
            # learning curve
            # save result +/ model
            # ...
'''
