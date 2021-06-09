import librosa
import numpy as np
filepath = "./../dataset/processed/funspeech/user1-093623a.wav"

sample_rate = 16000
sec_sample = 0.1 # Secondes
length = 1348

signal, sr = librosa.load(filepath, sr=sample_rate)

stop = int(len(signal)-np.floor(sample_rate*sec_sample))
rand_index = np.random.randint(0, stop)

sample = signal[rand_index:int(rand_index+sample_rate*sec_sample)]
# Si on prend la dernipre partie du signal, on est bon
s2 = signal[stop:]
print(sample.size,s2.size)
