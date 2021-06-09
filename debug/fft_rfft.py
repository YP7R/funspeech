from pydub import AudioSegment
import librosa
import features.feature_band_power as bp
import numpy as np

filepath = "./../dataset/processed/funspeech/muser1-093623a.wav"

signal, sr = librosa.load(filepath, sr=16000)
signal = bp.pad_signal(signal)
bp.compute_fft(signal)
bp.compute_rfft(signal)

print("*" * 100)
sig = [0, 1, 2, 0, 1, 1, 1, 1]

# sig = signal

f = np.fft.fft(sig)
r = np.fft.rfft(sig)
#######################################################################
print("FFT")
# print(' | '.join(map(str, f))) # Must be commented, complete fft
print(' | '.join(map(str, f[0:len(sig) // 2 + 1])))
print(' | '.join(map(str, bp.compute_fft(sig))))
print(bp.compute_fft(sig).size, f[0:len(sig)//2+1].size)

print("RFFT")
print(' | '.join(map(str, r)))
print(' | '.join(map(str, bp.compute_rfft(sig))))
print(bp.compute_rfft(sig).size, r.size)

#######################################################################
print("*" * 100)

ff = np.fft.fftfreq(len(sig), 1 / 16000)
rf = np.fft.rfftfreq(len(sig), 1 / 16000)

print("FFT FREQ")
print(ff[0], ff[len(sig) // 2], f"Size: {ff.size}")
print(' | '.join(map(str, np.abs(ff[0:len(sig) // 2 + 1]))))
print(' | '.join(map(str, bp.compute_fftfreq(sig, 16000))))

print("RFFT FREQ")
print(rf[0], rf[-1], f"Size: {rf.size}")
print(' | '.join(map(str, rf)))
print(' | '.join(map(str, bp.compute_rfftfreq(sig, 16000))))
