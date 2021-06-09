# Number of frames + reduce mfcc features
import os
import features.python_speech_features as psf
import librosa

filepath1 = f"..\\dataset\\processed\\funspeech\\user2-094848ou.wav"
filepath2 = f"..\\dataset\\processed\\funspeech\\user4-095421u.wav"

signal1, _ = librosa.load(filepath1, sr=16000)
signal2, _ = librosa.load(filepath2, sr=16000)

mfcc_1 = psf.improved.new_mfcc(signal1, 16000)
mwfcc_1 = psf.improved.new_mfcc(signal1, 16000, fbank_name="mel_weight")
lfcc_1 = psf.improved.new_mfcc(signal1, 16000, fbank_name="linear")
erbfcc_1 = psf.improved.new_mfcc(signal1, 16000, fbank_name="erb")

logfbank_1 = psf.improved.new_logfbank(signal1, 16000, fbank_name="erb")
logfbank_2 = psf.improved.new_logfbank(signal2, 16000, fbank_name="erb")

print(logfbank_1.shape)
print(logfbank_2.shape)
