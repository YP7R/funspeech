# Functions
import decimal
import numpy as np


def ms2signal_size(ms, sample_rate):
    return int(ms/1000 * sample_rate)


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def get_number_of_frames(length_signal, sample_rate, win_len=0.025, win_step=0.01):
    frame_len = win_len * sample_rate
    frame_step = win_step * sample_rate
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if length_signal <= frame_len:
        nb_frames = 1
    else:
        nb_frames = 1 + int(np.ceil((1.0 * length_signal - frame_len) / frame_step))
    return nb_frames

def reduce_mfcc_features(mfcc, nb_frames, file=""):
    nb_vector = mfcc.shape[0]
    mfcc_reduced = []
    for i in range(nb_frames):
        start = i/nb_frames * nb_vector
        end = (i+1)/nb_frames * nb_vector
        i_start, i_end = int(start), int(end)
        if i_start == i_end:
            print(f"{file}")
            input("")
            exit()
        mfcc_subset = mfcc[i_start:i_end]
        # print(mfcc_subset.shape)
        mfcc_mean = np.mean(mfcc_subset, axis=0)
        mfcc_reduced.append(mfcc_mean)

    return np.array(mfcc_reduced)
