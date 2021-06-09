from pydub import AudioSegment
import librosa
import features.energy_features as bp
import numpy as np

filepath = "./../dataset/processed/funspeech/muser1-093623a.wav"

signal, sr = librosa.load(filepath, sr=16000)

print("*" * 100)
sig = np.array([0, 1, 2, 0, 1, 1])

sig  # = signal

# If signal is loaded from librosa, compute energies
if sig.size > 10:
    print(bp.energies_algorithm(sig, 16000))
    exit(0)

#######################################################################
# FFT VS RFFT
f = np.fft.fft(sig)
r = np.fft.rfft(sig)
print("FFT")
# print(' | '.join(map(str, f))) # Must be commented, complete fft
print(' | '.join(map(str, f[0:len(sig) // 2 + 1])))
print(' | '.join(map(str, bp._compute_fft(sig))))
print(bp._compute_fft(sig).size, f[0:len(sig)//2+1].size)

print("RFFT")
print(' | '.join(map(str, r)))
print(' | '.join(map(str, bp._compute_rfft(sig))))
print(bp._compute_rfft(sig).size, r.size)

#######################################################################
# FFT FREQ vs RFFT FREQ
print("*" * 100)


ff = np.fft.fftfreq(len(sig), 1 / 16000)
rf = np.fft.rfftfreq(len(sig), 1 / 16000)

print("FFT FREQ")
print(ff[0], ff[len(sig) // 2], f"Size: {ff.size}")
print(' | '.join(map(str, np.abs(ff[0:len(sig) // 2 + 1]))))
print(' | '.join(map(str, bp._compute_fftfreq(sig, 16000))))

print("RFFT FREQ")
print(rf[0], rf[-1], f"Size: {rf.size}")
print(' | '.join(map(str, rf)))
print(' | '.join(map(str, bp._compute_rfftfreq(sig, 16000))))

#######################################################################
# Différence entre un signal paddé et non paddé
print("*" * 100)


padded_sig = bp._pad_signal(sig)
fft_values_pad = np.fft.rfft(padded_sig)
fft_values = np.fft.rfft(sig)
print(' | '.join(map(str, fft_values)))
print(' | '.join(map(str, fft_values_pad)))
