# Number of frames + reduce mfcc features
import os
import features.python_speech_features as psf
import librosa

filepath1 = f"..\\dataset\\processed\\funspeech\\user2-094848ou.wav"
filepath2 = f"..\\dataset\\processed\\funspeech\\user4-095421u.wav"
length = 20496
milliseconds = 1281

if not os.path.exists(filepath1) or not os.path.exists(filepath2):
    print("File doesn't exist")
    exit()

signal1, _ = librosa.load(filepath1, sr=16000)
signal2, _ = librosa.load(filepath2, sr=16000)

# Si je veux 100ms, je dois
signal_size = psf.improved.ms2signal_size(100, 16000)

print(nb1 := psf.improved.get_number_of_frames(len(signal1), 16000))
print(nb2 := psf.improved.get_number_of_frames(len(signal2), 16000))

mfcc_1 = psf.mfcc(signal1, 16000)
mfcc_2 = psf.mfcc(signal2, 16000)

print(mfcc_1.shape, mfcc_2.shape)

psf.improved.reduce_mfcc_features(mfcc_2, nb1)
