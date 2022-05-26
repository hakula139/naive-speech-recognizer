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


def dct(x: np.ndarray, d: int) -> np.ndarray:
    '''
    Perform a Discrete Cosine Transform of a 1D / 2D array.

    Args:
        `x`: source array, shape(n, l)
        `d`: dimension of the DCT matrix

    Returns:
        The result of DCT, shape(d, l)
    '''

    n = x.shape[0]
    c = np.sqrt(2 / n)
    s = np.pi / (2 * n) * np.arange(1, 2 * n, 2)
    dct_m = np.concatenate([
        [np.ones(n) / np.sqrt(n)],
        [c * np.cos(i * s) for i in range(1, d)],
    ])
    return np.dot(dct_m, x)
