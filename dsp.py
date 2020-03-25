# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''
#from hyperparams import Hyperparams as hp
import numpy as np
import librosa
import librosa.display
import copy
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import get_window
import os
import yaml
from .attrdict import Config
import soundfile as sf
# r9r9 preprocessing
import lws
from typing import Dict, List, Optional, Tuple, Union, Callable


def plot_spectrogram(mag, save=''):
    librosa.display.specshow(mag, x_axis='off')
    plt.title('spectrogram')
    if save != '':
        plt.savefig(save, format='jpg')
    else:
        plt.show()


def _butter_highpass(cutoff, fs:int, order:Optional[int]=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def _pySTFT(x:np.ndarray, n_fft:Optional[int]=1024, hop_length:Optional[int]=256):

    x = np.pad(x, int(n_fft//2), mode='reflect')

    noverlap = n_fft - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, n_fft)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)

    fft_window = get_window('hann', n_fft, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=n_fft).T

    return np.abs(result)



class Dsp():
    def __init__(self, config: Optional[Union[str, Config]] = 'config/dsp.yaml'):
        self.load_config(config)
        self._build_mel_basis()
        self._build_processor()
        self.b, self.a = _butter_highpass(30, self.hparams.sample_rate, order=5)

    def load_config(self, config: Union[str, Config]):
        if isinstance(config, str):
            self.hparams = Config.yaml_load(config)
        elif isinstance(config, Config):
            self.hparams = config

    def load_wav(self, path: 'PathLike[any]', trim=True):
        try:
            x, sr = sf.read(path)
            signed_int16_max = 2**15
            if x.dtype == np.int16:
                x = x.astype(np.float32) / signed_int16_max
            if sr != self.hparams.sample_rate:
                x = librosa.resample(x, sr, self.hparams.sample_rate)
            if trim:
                x, _ = librosa.effects.trim(x, top_db=15)
            x = np.clip(x, -1.0, 1.0)
            y = signal.filtfilt(self.b, self.a, x)
            return y
        except:
            print(f'Error: {path} is an invalid wavefile.')
            return -1

    def save_wav(self, wav: np.ndarray, path: 'PathLike[any]'):
        wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
        sf.write(path, wav.astype(np.int16), self.hparams.sample_rate)

    def spectrogram(self, y):
        D = self.processor.stft(self._preemphasis(y)).T
        S = self._amp_to_db(np.abs(D)) - self.hparams.ref_level_db
        return self._normalize(S).astype(np.float32)

    def inv_spectrogram(self, spectrogram):
        '''Converts spectrogram to waveform using librosa'''
        S = self._db_to_amp(self._denormalize(spectrogram) + self.hparams.ref_level_db)  # Convert back to linear
        D = self.processor.run_lws(S.astype(np.float64).T ** self.hparams.power)
        y = self.processor.istft(D).astype(np.float32)
        return self._inv_preemphasis(y)

    def melspectrogram(self, y:[np.ndarray]):
        y = self._preemphasis(y)
        D = _pySTFT(y, n_fft=self.hparams.fft_size, hop_length=self.hparams.hop_size)
        S = self._amp_to_db(self._linear_to_mel(D)) - self.hparams.ref_level_db
        if not self.hparams.allow_clipping_in_normalization:
            assert ret.max() <= 0 and ret.min() - self.hparams.min_level_db >= 0
        return self._normalize(S).astype(np.float32)


    def inv_melspectrogram(self, melspectrogram):
        '''Converts spectrogram to waveform using librosa'''
        S = self._db_to_amp(self._denormalize(melspectrogram) + self.hparams.ref_level_db)  # Convert back to linear
        S = self._mel_to_linear(S)
        y = self._griffin_lim(S)
        return self._inv_preemphasis(y)

    def melspectrogram2wav(self, melspectrogram, save=None):
        y = self.inv_melspectrogram(melspectrogram)
        if save is not None:
            self.save_wav(y, save)
        return y

    def _build_mel_basis(self):
        if self.hparams.fmax is not None:
            assert self.hparams.fmax <= self.hparams.sample_rate // 2
        self._mel_basis = librosa.filters.mel(self.hparams.sample_rate, self.hparams.fft_size,
                                fmin=self.hparams.fmin, fmax=self.hparams.fmax,
                                n_mels=self.hparams.num_mels)

    def _linear_to_mel(self, spectrogram):
        return np.dot(self._mel_basis, spectrogram)

    def _mel_to_linear(self, melspectrogram):
        def _mel_to_linear_matrix():
            m_t = np.transpose(self._mel_basis)
            p = np.matmul(self._mel_basis, m_t)
            d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
            return np.matmul(m_t, np.diag(d))
        m = _mel_to_linear_matrix()
        mag = np.dot(m, melspectrogram)
        return mag

    def _preemphasis(self, x):
        from nnmnkwii.preprocessing import preemphasis
        return preemphasis(x, self.hparams.preemphasis)

    def _inv_preemphasis(self, x):
        from nnmnkwii.preprocessing import inv_preemphasis
        return inv_preemphasis(x, self.hparams.preemphasis)

    def _amp_to_db(self, x):
        min_level = np.exp(self.hparams.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def _normalize(self, S):
        return np.clip((S - self.hparams.min_level_db) / -self.hparams.min_level_db, 0, 1)

    def _denormalize(self, S):
        return (np.clip(S, 0, 1) * -self.hparams.min_level_db) + self.hparams.min_level_db

    def _build_processor(self):
        self.processor = lws.lws(self.hparams.fft_size, self.hparams.hop_size, mode="speech")


    def _griffin_lim(self, melspectrogram):
        '''Applies Griffin-Lim's raw.
        '''
        X_best = copy.deepcopy(melspectrogram)
        for i in range(100):
            X_t = librosa.istft(X_best, self.hparams.hop_size)
            est = librosa.stft(X_t, self.hparams.fft_size, hop_length=self.hparams.hop_size)
            phase = est / np.maximum(1e-8, np.abs(est))
            X_best = melspectrogram * phase
        X_t = librosa.istft(X_best, self.hparams.hop_size)
        y = np.real(X_t)

        return y



