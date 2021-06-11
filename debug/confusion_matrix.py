import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os
import shutil

from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from features.barycentre import BaryCentre
from sklearn.svm import SVC
csv_files = glob.glob("..\\dataset\\processed\\" + "**/*.csv")
features_directory = "..\\dataset\\features\\"
output_directory = "..\\dataset\\confusion_matrix\\"


#shutil.rmtree(output_directory, ignore_errors=True)
#os.makedirs(output_directory)
features_name = "features_logfbank"

for csv_file in csv_files:
    dataframe = pd.read_csv(csv_file)

    csv_name, ext = os.path.splitext(os.path.basename(csv_file))
    feature_path = f"{features_directory}{csv_name}\\{features_name}.npy"

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

    bc = BaryCentre()
    SVC(probability=True, random_state=0, kernel='linear', gamma='scale')
    predictions = cross_val_predict(bc, features, sounds_expected, cv=5)
    cm = confusion_matrix(sounds_expected, predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_)
    ax = disp.plot(cmap=plt.cm.Blues)
    ax.ax_.set_title("Confusion Matrix")

    output_path = f"{output_directory}{csv_name}_{features_name}_SVM.png"
    plt.savefig(output_path)
    #plt.show()
