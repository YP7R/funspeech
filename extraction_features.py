import glob
import os
import shutil
import librosa
import pandas as pd
import numpy as np
import features.energy_features as bp
import features.python_speech_features as psf

random = np.random.RandomState(0)

sample_rate = 16000
sec_sample = 100
sample_size = psf.improved.ms2signal_size(sec_sample, sample_rate)

# Fichiers de référence
input_directory = ".\\dataset\\processed\\"
output_directory = ".\\dataset\\features\\"

if not os.path.exists(input_directory):
    print(f"{input_directory} doesn't exist")
    exit()

csv_files = glob.glob(input_directory + "**/*.csv", recursive=True)
print(csv_files)

parameters_mfcc = {'winlen': 0.025, 'winstep': 0.01,
                   'numcep': 13, 'nfilt': 26, 'nfft': 512, 'lowfreq': 0, 'highfreq': None,
                   'preemph': 0.97, 'ceplifter': 22, 'appendEnergy': True, 'winfunc': np.hamming}

for csv_file in csv_files:
    # Crée le repertoire qui va contenir les features
    base_file, ext = os.path.splitext(os.path.basename(csv_file))
    output_path = f"{output_directory}{base_file}\\"
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path)

    # Lecture du csv
    dataframe = pd.read_csv(csv_file)

    # Full audio, récupere le fichier le plus court et il définit notre nombre de fenêtres
    minimum_ms = dataframe['milliseconds'].min()
    minimum_sig_size = psf.improved.ms2signal_size(minimum_ms, sample_rate)
    nb_frames = psf.improved.get_number_of_frames(minimum_sig_size, sample_rate)

    # Sample audio minimum frame
    minimum_sig_size_sample = psf.improved.ms2signal_size(sec_sample, sample_rate)
    nb_frames_sample = psf.improved.get_number_of_frames(minimum_sig_size_sample, sample_rate)

    print(f"Nombre de fenêtres : {nb_frames}")
    # print(dataframe[dataframe['milliseconds'] == dataframe['milliseconds'].min()])
    # print(dataframe[dataframe['milliseconds'] == dataframe['milliseconds'].max()])

    # Path of the sounds
    sounds_path = f"{input_directory}{base_file}"

    # Array to store features
    features_energies_sample = []
    features_energies = []

    features_mfcc_sample = []
    features_mfcc = []

    for index, row in dataframe.iterrows():
        filename = f"{row['base_file']}{row['sound_id']}.wav"
        filepath = f"{sounds_path}\\{filename}"
        signal, _ = librosa.load(filepath, sample_rate)

        # Signal sample
        stop = len(signal) - sample_size
        rand_index = random.randint(0, stop)
        signal_sample = signal[rand_index:rand_index + sample_size]

        ## Features
        # energies
        energies = bp.energies_algorithm(signal, sample_rate)
        energies_sample = bp.energies_algorithm(signal_sample, sample_rate)
        features_energies.append(energies)
        features_energies_sample.append(energies_sample)

        # mfcc
        mfcc = psf.improved.reduce_mfcc_features(psf.mfcc(signal, sample_rate, **parameters_mfcc), nb_frames, filename)
        mfcc_sample = psf.mfcc(signal_sample, sample_rate, **parameters_mfcc)
        features_mfcc.append(mfcc)
        features_mfcc_sample.append(mfcc_sample)

    np.save(f"{output_path}features_energies.npy", energies)
    np.save(f"{output_path}features_energies_sample.npy", energies_sample)

    np.save(f"{output_path}features_mfcc.npy",features_mfcc)
    np.save(f"{output_path}features_mfcc_sample.npy", features_mfcc_sample)
