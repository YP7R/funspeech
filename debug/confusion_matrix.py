import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from features.barycentre import BaryCentre

csv_files = glob.glob("..\\dataset\\processed\\"+"**/*.csv")
for csv_file in csv_files:
    print(csv_file)
'''
df = pd.read_csv("..\\dataset\\processed\\wmusounds\\wmusounds.csv")
data_path_1 = "..\\dataset\\features\\wmusounds\\features_energies.npy"
data_path_2 = "..\\dataset\\features\\wmusounds\\features_mfcc.npy"

features = np.load(data_path_1)
features = np.reshape(features, (features.shape[0], -1))
# MinMaxScaler
minmax = MinMaxScaler()

sounds_expected = df['sound_id'].values
labelencoder = LabelEncoder()
sounds_expected = labelencoder.fit_transform(sounds_expected)

features = minmax.fit_transform(features)

df_adult = df.loc[df['category'].isin(["men", "women"])]
df_child = df.drop(df_adult.index)

# df_child = df.loc[df['base_file'].str.startswith('user4')]
# df_adult = df.drop(df_child.index)


features_child = features[df_child.index.values]
features_adult = features[df_adult.index.values]
sounds_child = sounds_expected[df_child.index.values]
sounds_adult = sounds_expected[df_adult.index.values]

bc = BaryCentre()
bc.fit(features_adult, sounds_adult)
# print("*"*100)
# y_pred = bc.predict(features_child)

# confusion = confusion_matrix(sounds_child, y_pred
# disp = plot_confusion_matrix(bc, features_child, sounds_child, cmap=plt.cm.Blues, display_labels=labelencoder.classes_)
# disp.ax_.set_title("Confusion matrix")
# plt.show()

predictions = cross_val_predict(bc, features_adult, sounds_adult, cv=5)
cm = confusion_matrix(sounds_adult, predictions)

disp = ConfusionMatrixDisplay(cm, display_labels=labelencoder.classes_)
ax = disp.plot(cmap=plt.cm.Blues)
ax.ax_.set_title("Confusion Matrix")
#plt.savefig("./ok.png")
plt.show()
'''