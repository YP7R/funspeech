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
    return np.sqrt(np.sum(np.square(signal)) / len(signal)) * signal


def pad_signal(signal):
    nb_zeros = int(np.power(2, np.ceil(np.log2(signal.size))) - signal.size)
    signal_padded = np.pad(signal, pad_width=(0, nb_zeros), mode='constant')
    return signal_padded


def compute_fft(signal):
    fft_values = np.fft.fft(signal)
    # print(f"FFT size: {fft_values.size}, size if trunc {fft_values[0:len(signal)//2+1].size}")
    values = fft_values[0:len(signal) // 2 + 1]
    return values


def compute_rfft(signal):
    fft_values = np.fft.rfft(signal)
    # print(f"RFFT size: {fft_values.size}")
    return fft_values


def compute_fftfreq(signal, sr):
    fft_freq = np.fft.fftfreq(len(signal), 1 / sr)
    freq = np.abs(fft_freq[0:len(signal) // 2 + 1])
    return freq


def compute_rfftfreq(signal, sr):
    freq = np.fft.rfftfreq(len(signal), 1 / sr)
    return freq


def compute_fft_log(fft_values):
    fft_log_values = np.where(fft_values == 0, np.finfo(float).eps, fft_values)
    fft_log_values = np.log(fft_log_values)
    return fft_log_values


def hz2bin(freq, bands):
    delta = (freq[1] - freq[0])
    indices = np.array(np.floor(bands / delta), dtype=np.int)
    return indices


def compute_power_per_bands(freq, fft_values, bands):
    bands_indices = hz2bin(freq, bands)
    features = []
    for index, bi in enumerate(bands[0:-1]):
        start, stop = bands_indices[index], bands_indices[index + 1]
        bin_indices = np.arange(start, stop)
        values = np.take(fft_values, bin_indices)
        energy = np.sum(np.power(values, 2))
        features.append(energy)

    features = np.array(features) / len(features)
    print(features)
    return features


# ...


def algorithm(signal, sample_rate, bands=np.array([150, 551, 1051, 1901, 3501, 6501])):
    # Normalise le volume
    signal_normalized = normalize_volume(signal)

    # Pad le signal
    signal_padded = pad_signal(signal_normalized)

    # Compute la fft
    fft_values = compute_fft(signal_padded)
    # rfft_values = compute_rfft(signal)
    freq = compute_fftfreq(signal_padded, sample_rate)
    # rfreq = compute_rfftfreq(signal, sample_rate)

    # Compute la norme / magnitude
    fft_values = np.absolute(fft_values)

    # Applique le logarithme
    fft_log_values = compute_fft_log(fft_values)

    # Normalisation
    fft_normalized_values = fft_log_values - np.min(fft_log_values)

    # Compute features ...
    powers = compute_power_per_bands(freq, fft_normalized_values, bands)
    # print(' | '.join(map(str, fft_values)))
    # print(' | '.join(map(str, freq)))
    # print(hz2bin(freq, bands))
