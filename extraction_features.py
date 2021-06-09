import glob
import os
import pandas as pd
import shutil
import features.energy_features as bp
import librosa
import numpy as np

# Fichiers de référence
input_directory = ".\\dataset\\processed\\"
output_directory = ".\\dataset\\features\\"

sample_rate = 16000

if not os.path.exists(input_directory):
    print(f"{input_directory} doesn't exist")
    exit()

csv_files = glob.glob(input_directory + "**/*.csv", recursive=True)
print(csv_files)

for csv_file in csv_files[0:1]:

    # Crée le repertoire qui va contenir les features
    base_file, ext = os.path.splitext(os.path.basename(csv_file))
    output_path = f"{output_directory}{base_file}\\"
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path)

    # Lecture du csv
    dataframe = pd.read_csv(csv_file)
    minimum_length = dataframe['length'].min()
    print(dataframe.columns)

    # Path of the sounds
    sounds_path = f"{input_directory}{base_file}"

    # Array to store features
    bands_features = []

    for index, row in dataframe.iterrows():
        filename = f"{row['base_file']}{row['sound_id']}.wav"
        filepath = f"{sounds_path}\\{filename}"
        signal, _ = librosa.load(filepath, sample_rate)

        energies = bp.energies_algorithm(signal, sample_rate)
        bands_features.append(energies)

    np.save(f"{output_path}features_energies.npy", energies)
