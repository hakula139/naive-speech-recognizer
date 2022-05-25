import numpy as np


def mel_freq(f: np.ndarray) -> np.ndarray:
    return 2595 * np.log10(1 + f / 700)


def i_mel_freq(mel_f: np.ndarray) -> np.ndarray:
    return 700 * (10 ** (mel_f / 2595) - 1)


def get_mel_filters(
    n_filters: int, sr: int, n_window: int, f_min: float, f_max: float,
) -> np.ndarray:
    '''
    Obtain the Mel filterbanks.

    Args:
        `n_filters`: the number of Mel filterbanks
        `sr`: sample rate
        `n_window`: window length
        `f_min`: minimum frequency
        `f_max`: maximum frequency

    Returns:
        Mel filterbanks of shape(`n_filters`, `n_window` / 2).
    '''

    mel_f_min, mel_f_max = mel_freq(f_min), mel_freq(f_max)
    mel_f = np.linspace(mel_f_min, mel_f_max, n_filters + 2)
    f = np.floor(i_mel_freq(mel_f) * n_window / sr).astype(int)

    filter_len = n_window // 2
    filters = np.array([np.concatenate([
        np.zeros(f[i - 1]),
        np.linspace(0, 1, f[i] - f[i - 1], endpoint=False),
        np.linspace(1, 0, f[i + 1] - f[i], endpoint=False),
        np.zeros(filter_len - f[i + 1]),
    ]) for i in range(1, n_filters + 1)])
    return filters


def mfcc(spec: np.ndarray, n_filters: int, sr: int) -> np.ndarray:
    '''
    Calculate the Mel-Frequency Cepstral Coefficients (MFCCs) from the spectrogram.

    Args:
        `spec`: the spectrogram of the audio signal
        `n_filters`: the number of Mel filterbanks
        `sr`: sample rate
    '''
