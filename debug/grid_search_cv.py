import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

random = np.random.RandomState(0)

classifiers = [{'clf_name': 'KNN', 'clf': KNeighborsClassifier(),
                'param_grid': [
                    {'weights': ['distance'],
                     'n_neighbors': np.arange(5, 18, 4)}
                ]},
               {'clf_name': 'SVM', 'clf': SVC(probability=True, random_state=0,kernel='linear',gamma='scale'),
                'param_grid': [
                    {'C': np.arange(0.94, 1.01, 0.01)}
                ]}
               ]

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

features_child = features[df_child.index.values]
features_adult = features[df_adult.index.values]
sounds_child = sounds_expected[df_child.index.values]
sounds_adult = sounds_expected[df_adult.index.values]

grid = GridSearchCV(classifiers[1]['clf'], classifiers[1]['param_grid'], cv=10, n_jobs=-1)
grid.fit(features_adult, sounds_adult)
print(grid.best_params_)
print(grid.best_score_)
