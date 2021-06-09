import os
import features.python_speech_features as psf
import librosa

filepath = f"..\\dataset\\processed\\funspeech\\user2-094848ou.wav"
length = 20496
milliseconds = 1281


if not os.path.exists(filepath):
    print("File doesn't exist")
    exit()

signal, _ = librosa.load(filepath, sr=16000)
print(len(signal))

# Si je veux 100ms, je dois
signal_size = psf.improved.ms2signal_size(100, 16000)
print(signal_size)
print(psf.improved.get_number_of_frames(len(signal),16000))

psf.mfcc(signal, 16000)
