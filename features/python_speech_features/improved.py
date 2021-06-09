# Functions
import decimal
import numpy as np
from features.python_speech_features import base, sigproc


def ms2signal_size(ms, sample_rate):
    return int(ms / 1000 * sample_rate)


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
        start = i / nb_frames * nb_vector
        end = (i + 1) / nb_frames * nb_vector
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


# Scales function
def hz2erb(hz):
    return 21.4 * np.log10(1 + 0.00437 * hz)


def erb2hz(erb):
    return (np.power(10, erb / 21.4) - 1) / 0.00437


## New version
def new_mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
             nfilt=26, nfft=None, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True,
             winfunc=lambda x: np.ones((x,)), fbank_name="mel"):
    """Compute MFCC features from an audio signal."""

    nfft = nfft or base.calculate_nfft(samplerate, winlen)
    feat, energy = new_fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph, winfunc,
                             fbank_name)
    feat = np.log(feat)
    feat = base.dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
    feat = base.lifter(feat, ceplifter)
    if appendEnergy:
        feat[:, 0] = np.log(energy)  # replace first cepstral coefficient with log of frame energy
    return feat


# Can use this as feature
def new_fbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
              nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
              winfunc=lambda x: np.ones((x,)), fbank_name="mel"):
    """Compute Mel-filterbank energy features from an audio signal."""

    highfreq = highfreq or samplerate / 2
    signal = sigproc.preemphasis(signal, preemph)
    frames = sigproc.framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
    pspec = sigproc.powspec(frames, nfft)
    energy = np.sum(pspec, 1)  # this stores the total energy in each frame
    energy = np.where(energy == 0, np.finfo(float).eps, energy)  # if energy is zero, we get problems with log

    fb = new_get_filter_banks(nfilt, nfft, samplerate, lowfreq, highfreq, fbank_name)
    feat = np.dot(pspec, fb.T)  # compute the filterbank energies
    feat = np.where(feat == 0, np.finfo(float).eps, feat)  # if feat is zero, we get problems with log

    return feat, energy


# Can use this feature
def new_logfbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
                 nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
                 winfunc=lambda x: np.ones((x,)), fbank_name="mel"):
    """Compute log Mel-filterbank energy features from an audio signal."""

    feat, energy = new_fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph, winfunc,
                             fbank_name)
    return np.log(feat)


def new_get_filter_banks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None, fbank_name="mel"):
    if fbank_name == "mel":
        return base.get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    elif fbank_name == "mel_weight":
        return get_fbank_mel_weight(nfilt, nfft, samplerate, lowfreq, highfreq)
    elif fbank_name == "erb":
        return get_fbank_erb(nfilt, nfft, samplerate, lowfreq, highfreq)
    elif fbank_name == "linear":
        return get_fbank_linear(nfilt, nfft, samplerate, lowfreq, highfreq)
    else:
        return base.get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)


# Banc de filtre mel équilibré
def get_fbank_mel_weight(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    lowmel = base.hz2mel(lowfreq)
    highmel = base.hz2mel(highfreq)
    mel_points = np.linspace(lowmel, highmel, nfilt + 2)
    bin = np.floor((nfft + 1) * base.mel2hz(mel_points) / samplerate)

    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = 2 * (i - bin[j]) / ((bin[j + 1] - bin[j]) * (bin[j + 2] - bin[j]))
            # fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = 2 * (bin[j + 2] - i) / ((bin[j + 2] - bin[j + 1]) * (bin[j + 2] - bin[j]))
            # fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank


# Banc de filtre erb
def get_fbank_erb(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    lowerb = hz2erb(lowfreq)
    higherb = hz2erb(highfreq)
    erb_points = np.linspace(lowerb, higherb, nfilt + 2)

    bin = np.floor((nfft + 1) * erb2hz(erb_points) / samplerate)

    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank


# Banc de filtre linéaire
def get_fbank_linear(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    linear_points = np.linspace(lowfreq, highfreq, nfilt + 2)
    bin = np.floor((nfft + 1) * (linear_points) / samplerate)

    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank
