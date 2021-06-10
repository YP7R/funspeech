import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from features.barycentre import BaryCentre
import matplotlib.pyplot as plt

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

estimator = SVC()

# Learning curve
train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(estimator=estimator, X=features,
                                                                                y=sounds_expected,
                                                                                train_sizes=np.linspace(0.1, 1, 10),
                                                                                cv=10,
                                                                                n_jobs=-1, shuffle=True,
                                                                                random_state=random, return_times=True)

print(features.shape)
print(train_sizes)
print(train_scores)
print(test_scores)
print(fit_times)  # Time spent for fitting
print(score_times)  # Time spent for scoring
print(np.linspace(0.1, 1.0, 10))

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)
score_times_mean = np.mean(score_times, axis=1)
score_times_std = np.std(score_times, axis=1)

figure, axes = plt.subplots(1, 4, figsize=(22, 5))
figure.suptitle("Courbes d'apprentissage")

axes[0].set_title("Learning curve")
axes[0].set_xlabel("Training examples")
axes[0].set_ylabel("Score")

axes[0].grid()
axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
axes[0].legend(loc="best")

axes[1].grid()
axes[1].plot(fit_times_mean, test_scores_mean, 'o-')
axes[1].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1)
axes[1].set_xlabel("fit_times")
axes[1].set_ylabel("Score")
axes[1].set_title("Performance of the model")

axes[2].grid()
axes[2].plot(train_sizes, fit_times_mean, 'o-')
axes[2].fill_between(train_sizes, fit_times_mean - fit_times_std,
                     fit_times_mean + fit_times_std, alpha=0.1)
axes[2].set_xlabel("Training examples")
axes[2].set_ylabel("fit_times")
axes[2].set_title("Scalability of the model (Training)")

axes[3].grid()
axes[3].plot(train_sizes, score_times_mean, 'o-')
axes[3].fill_between(train_sizes, score_times_mean - score_times_std,
                     score_times_mean + score_times_std, alpha=0.1)
axes[3].set_xlabel("Training examples")
axes[3].set_ylabel("score_times")
axes[3].set_title("Scalability of the model (Testing)")
plt.show()
