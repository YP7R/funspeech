import numpy as np
import matplotlib.pyplot as plt
# This is an improvement version of what was on FunSpeech, coded by me.
# Professor :
'''
# - Normalize volume
# - FFT
# - Log
# - Energy per bands
'''


def normalize_volume(signal):
    return np.sqrt(np.sum(np.square(signal)) / signal.size) * signal


def pad_signal(signal):
    nb_zeros = int(np.power(2, np.ceil(np.log2(signal.size))) - signal.size)
    signal_padded = np.pad(signal, pad_width=(0, nb_zeros), mode='constant')
    return signal_padded


def compute_fft(signal):
    fft_values = np.fft.fft(signal)
    # print(f"FFT size: {fft_values.size}, size if trunc {fft_values[0:len(signal)//2+1].size}")
    values = fft_values[0:len(signal)//2+1]
    return values


def compute_rfft(signal):
    fft_values = np.fft.rfft(signal)
    # print(f"RFFT size: {fft_values.size}")
    return fft_values


def compute_fftfreq(signal, sr):
    fft_freq = np.fft.fftfreq(len(signal), 1/sr)
    freq = np.abs(fft_freq[0:len(signal)//2+1])
    return freq


def compute_rfftfreq(signal, sr):
    freq = np.fft.rfftfreq(len(signal), 1/sr)
    return freq

def compute_power_per_bands(audio, samplerate, bands=np.array([150, 551, 1051, 1901, 3501, 6501])):
    signal_processed = normalize_volume(audio)

    # TODO...
    return


def compute_powers():
    return


# ...


def compute_log(signal, samplerate):
    pass
