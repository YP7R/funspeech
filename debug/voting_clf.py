import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os
import shutil
import features.plot as pl
import features.learning_curve as lc

from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from features.barycentre import BaryCentre

csv_files = glob.glob("..\\dataset\\processed\\" + "**/*.csv")
features_directory = "..\\dataset\\features\\"
output_directory = "..\\dataset\\voting_clf\\"

shutil.rmtree(output_directory, ignore_errors=True)
os.makedirs(output_directory)
feature_name = "features_erbfcc"

voting_classifier = VotingClassifier(estimators=[
    ('knn', KNeighborsClassifier(weights='distance', n_neighbors=3, n_jobs=-1)),
    ('svm_1', SVC(probability=True, random_state=0, kernel='linear', gamma='scale')),
    ('svm_2', SVC(probability=True, random_state=0, kernel='rbf', gamma='scale')),
    ('bagging_knn', BaggingClassifier(KNeighborsClassifier(weights='distance'), max_samples=0.5))
], voting='soft')

for csv_file in csv_files:
    dataframe = pd.read_csv(csv_file)

    csv_name, ext = os.path.splitext(os.path.basename(csv_file))
    feature_path = f"{features_directory}{csv_name}\\{feature_name}.npy"

    features = np.load(feature_path)
    features = np.reshape(features, (features.shape[0], -1))
    minmax_scaler = MinMaxScaler()
    features = minmax_scaler.fit_transform(features)

    # Loading expected values (Y)
    sounds_expected = dataframe['sound_id'].values
    label_encoder = LabelEncoder()
    sounds_expected = label_encoder.fit_transform(sounds_expected)

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

    save_path = f"{output_directory}{csv_name}_{feature_name}_votting_classifier.png"

    result_learning_curve = lc.new_learning_curve(voting_classifier, features_adult, sounds_adult, features_child,
                                                  sounds_child, scoring=['accuracy', 'neg_log_loss'])
    pl.plot_curves(result_learning_curve, scoring=['accuracy', 'neg_log_loss'], features_name=feature_name,
                   classifier_name="votting_classifier",
                   dataset_name=csv_name, save_path=save_path)
