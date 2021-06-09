import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from features.barycentre import BaryCentre
random = np.random.RandomState(0)

df = pd.read_csv("..\\dataset\\processed\\wmusounds\\wmusounds.csv")
data_path = "..\\dataset\\features\\wmusounds\\features_energies.npy"
data_path2 = "..\\dataset\\features\\funspeech\\features_mfcc.npy"


features = np.load(data_path)
features = np.reshape(features, (features.shape[0], -1))
# MinMaxScaler
minmax = MinMaxScaler()
features = minmax.fit_transform(features)


# Prediction
sounds_expected = df['sound_id'].values
labelencoder = LabelEncoder()
sounds_expected = labelencoder.fit_transform(sounds_expected)

df_adult = df.loc[df['category'].isin(["men", "women"])]
df_child = df.drop(df_adult.index)

#df_child = df.loc[df['base_file'].str.startswith('user4')]
#df_adult = df.drop(df_child.index)

features_child = features[df_child.index.values]
features_adult = features[df_adult.index.values]
sounds_child = sounds_expected[df_child.index.values]
sounds_adult = sounds_expected[df_adult.index.values]


bc = BaryCentre()
bc.fit(features_adult, sounds_adult)
print("*"*100)
y = bc.predict(features_child)
v = accuracy_score(y,sounds_child)
print(v)
# TODO... SVM , KNN, BAGGING, confusion matrix, classifieur prof
