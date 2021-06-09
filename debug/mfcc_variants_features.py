# Number of frames + reduce mfcc features
import os
import features.python_speech_features as psf
import librosa

filepath1 = f"..\\dataset\\processed\\funspeech\\user2-094848ou.wav"

signal, _ = librosa.load(filepath1, sr=16000)

mfcc_1 = psf.improved.new_mfcc(signal, 16000)
mwfcc_1 = psf.improved.new_mfcc(signal, 16000, fbank_name="mel_weight")
lfcc_1 = psf.improved.new_mfcc(signal, 16000, fbank_name="linear")
erbfcc_1 = psf.improved.new_mfcc(signal, 16000, fbank_name="erb")

mfcc_2 = psf.mfcc(signal, 16000)

print(mfcc_1[0])
print(mwfcc_1[0])
print(lfcc_1[0])
print(erbfcc_1[0])
