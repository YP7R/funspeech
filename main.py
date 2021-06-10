import sys
import os
import glob
import shutil
import numpy as np

csv_directories = ".\\dataset\\processed\\"
features_directories = ".\\dataset\\features\\"
result_directory = ".\\dataset\\results\\"

if not os.path.exists(csv_directories):
    print("CSV directories doesn't exist")
    exit(0)

shutil.rmtree(result_directory, ignore_errors=True)
os.makedirs(result_directory)

csv_files = glob.glob(f"{csv_directories}" + "**/*.csv", recursive=True)
classifiers = []

# Pour les datasets
for csv_file in csv_files:
    csv_name, ext = os.path.splitext(os.path.basename(csv_file))
    feature_files = glob.glob(f"{features_directories}{csv_name}" + "**/*.npy", recursive=True)

    output_directory = f"{result_directory}{csv_name}\\"
    shutil.rmtree(output_directory, ignore_errors=True)
    os.makedirs(output_directory)

    print(f" {csv_name.upper()} ".center(100, "*"))
    # Pour les features
    for feature_file in feature_files:
        feature_name, ext = os.path.splitext(os.path.basename(feature_file))
        print(feature_file)
        features = np.load(feature_file)
        # Pour les classifieurs
        for clf in classifiers:
            pass

'''
# template
# for csv in csvs:
    # for npy in npy:
        # for clf in clfs:
        
            # learning curve
            # save result +/ model
            # ...
'''
