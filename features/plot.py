import matplotlib.pyplot as plt
import numpy as np


def plot_curves(results, scoring, features_name, classifier_name, dataset_name, save_path):
    scoring_results = {}
    for score in scoring:
        scoring_results[f"train_{score}_mean"] = np.mean(results[f"train_{score}"], axis=1)
        scoring_results[f"train_{score}_std"] = np.std(results[f"train_{score}"], axis=1)
        scoring_results[f"test_{score}_mean"] = np.mean(results[f"test_{score}"], axis=1)
        scoring_results[f"test_{score}_std"] = np.std(results[f"test_{score}"], axis=1)
        scoring_results[f"validation_{score}_mean"] = np.mean(results[f"validation_{score}"], axis=1)
        scoring_results[f"validation_{score}_std"] = np.std(results[f"validation_{score}"], axis=1)

    train_sizes = results['train_sizes']
    fit_time_mean = np.mean(results['fit_time'], axis=1)
    fit_time_std = np.std(results['fit_time'], axis=1)
    score_time_mean = np.mean(results['score_time'], axis=1)
    score_time_std = np.std(results['score_time'], axis=1)

    figure, axes = plt.subplots(1, 1 + len(scoring), figsize=(20, 5))
    figure.suptitle(f"dataset: {dataset_name}, classifier: {classifier_name}, features: {features_name}")

    for index, score in enumerate(scoring):
        axes[index].set_title(score)
        axes[index].set_xlabel("Training examples")
        axes[index].set_ylabel(score)
        axes[index].grid()

        axes[index].fill_between(train_sizes,
                                 scoring_results[f"train_{score}_mean"] - scoring_results[f"train_{score}_std"],
                                 scoring_results[f"train_{score}_mean"] + scoring_results[f"train_{score}_std"],
                                 alpha=0.1, color="r")
        axes[index].fill_between(train_sizes,
                                 scoring_results[f"test_{score}_mean"] - scoring_results[f"test_{score}_std"],
                                 scoring_results[f"test_{score}_mean"] + scoring_results[f"test_{score}_std"],
                                 alpha=0.1, color="g")

        axes[index].fill_between(train_sizes,
                                 scoring_results[f"validation_{score}_mean"] - scoring_results[f"validation_{score}_std"],
                                 scoring_results[f"validation_{score}_mean"] + scoring_results[f"validation_{score}_std"],
                                 alpha=0.1, color="b")

        axes[index].plot(train_sizes, scoring_results[f"train_{score}_mean"], 'o-', color="r", label=f"Training {score}")
        axes[index].plot(train_sizes, scoring_results[f"test_{score}_mean"], 'o-', color="g", label=f"Cross-validation {score}")
        axes[index].plot(train_sizes, scoring_results[f"validation_{score}_mean"], 'o-', color="b",
                         label=f"Validation / testing {score}")

        axes[index].legend(loc="best")

    axes[len(scoring)].grid()
    axes[len(scoring)].plot(train_sizes, fit_time_mean, 'o-', color="r", label="Fitting time")
    axes[len(scoring)].fill_between(train_sizes, fit_time_mean - fit_time_std,
                                    fit_time_mean + fit_time_std, alpha=0.1, color="r")
    axes[len(scoring)].plot(train_sizes, score_time_mean, 'o-', color="g", label="Scoring time")
    axes[len(scoring)].fill_between(train_sizes, score_time_mean - score_time_std,
                                    score_time_mean + score_time_std, alpha=0.1, color="g")

    axes[len(scoring)].set_xlabel("Training examples")
    axes[len(scoring)].set_ylabel("Time [s]")
    axes[len(scoring)].set_title("Scalability of the model (Fittitng / Testing)")
    axes[len(scoring)].legend(loc="best")
    plt.savefig(save_path)
    # plt.show()