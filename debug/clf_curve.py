import features.learning_curve as lc
import features.plot as pl
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from features.barycentre import BaryCentre

random = np.random.RandomState(0)

df = pd.read_csv("..\\dataset\\processed\\wmusounds\\wmusounds.csv")
data_path = "..\\dataset\\features\\wmusounds\\features_erbfcc.npy"

features = np.load(data_path)
features = np.reshape(features, (features.shape[0], -1))
# MinMaxScaler
minmax = MinMaxScaler()
features = minmax.fit_transform(features)

# Prediction
sounds_expected = df['sound_id'].values
label_encoder = LabelEncoder()
sounds_expected = label_encoder.fit_transform(sounds_expected)

df_adult = df.loc[df['category'].isin(["men", "women"])]
df_child = df.drop(df_adult.index)

# df_child = df.loc[df['base_file'].str.startswith('user4')]
# df_adult = df.drop(df_child.index)

features_child = features[df_child.index.values]
features_adult = features[df_adult.index.values]
sounds_child = sounds_expected[df_child.index.values]
sounds_adult = sounds_expected[df_adult.index.values]

svc = SVC(probability=True)

x = lc.new_learning_curve(svc, features_adult, sounds_adult, features_child, sounds_child,
                          scoring=['accuracy', 'neg_log_loss'])

print(x)
pl.plot_curves(x, "ok", "ok", "ok", "ok", scoring=['accuracy', 'neg_log_loss'])
