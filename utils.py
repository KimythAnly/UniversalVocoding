import librosa
import numpy as np
import scipy
from scipy import signal
from scipy.signal import get_window
import soundfile as sf

def _butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def _pySTFT(x, n_fft=1024, hop_length=256):

    x = np.pad(x, int(n_fft//2), mode='reflect')

    noverlap = n_fft - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, n_fft)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)

    fft_window = get_window('hann', n_fft, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=n_fft).T

    return np.abs(result)

def load_wav(path, sample_rate, trim=True):
        b, a = _butter_highpass(30, sample_rate, order=5)
    #try:
        x, sr = sf.read(path)
        signed_int16_max = 2**15
        if x.dtype == np.int16:
            x = x.astype(np.float32) / signed_int16_max
        if sr != sample_rate:
            x = librosa.resample(x, sr, sample_rate)
        if trim:
            x, _ = librosa.effects.trim(x, top_db=30) # 15
        y = signal.filtfilt(b, a, x)
        y = np.clip(y, -1.0, 1.0)
        return y
    #except:
    #    print(f'Error: {path} is an invalid wavefile.')
    #    return -1

def save_wav(path, wav, sample_rate):
    librosa.output.write_wav(path, wav.astype(np.float32), sr=sample_rate)


def mulaw_encode(x, channels):
    mu = channels - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return ((fx + 1) / 2 * mu + 0.5).astype(np.int32)


def mulaw_decode(y, channels):
    mu = channels - 1
    y = y.astype(np.float32)
    #y = 2 * (y/mu) - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)


def deemphasis(x, preemph):
    return scipy.signal.lfilter([1], [1, -preemph], x)


def melspectrogram(y, sample_rate, preemph, num_mels, num_fft, min_level_db, ref_level_db, hop_length, fmin, fmax):
    y = preemphasis(y, preemph)
    #S = np.abs(librosa.stft(y, n_fft=num_fft, hop_length=hop_length))
    S = _pySTFT(y, n_fft=num_fft, hop_length=hop_length)
    mel_basis = librosa.filters.mel(sample_rate, num_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    S = np.dot(mel_basis, S)
    mel = amp_to_db(S, min_level_db=min_level_db) - ref_level_db
    return normalize(mel, min_level_db=min_level_db).T


def amp_to_db(x, min_level_db):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def normalize(S, min_level_db):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)
