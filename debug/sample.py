import librosa
import numpy as np
import features.python_speech_features as psf

filepath = "./../dataset/processed/funspeech/user1-093623a.wav"

sample_rate = 16000
sec_sample = 100  # millisecondes
length = 1348

signal, sr = librosa.load(filepath, sr=sample_rate)

sample_size = psf.improved.ms2signal_size(sec_sample, sample_rate)

stop = len(signal) - sample_size
rand_index = np.random.randint(0, stop)
sample = signal[rand_index:rand_index + sample_size]

# Si on prend la dernière partie du signal , on est bon
limit = signal[stop:]
print(limit.size, sample.size)
