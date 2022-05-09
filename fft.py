import numpy as np


def fft(a: np.ndarray) -> np.ndarray:
    '''
    Compute the one-dimensional Discrete Fourier Transform.

    Args:
        `a`: array of `n` complex values, where `n` is a power of 2

    Returns:
        Array of length `n` containing the result of FFT.
    '''

    n = a.shape[0]
    if n == 1:
        return a

    y_e = fft(a[::2])   # even indices of a
    y_o = fft(a[1::2])  # odd indices of a

    y = np.empty(n, dtype=complex)
    w = np.exp(2j * np.pi / n * np.arange(n // 2))  # roots of unity
    for i in range(n // 2):
        y[i] = y_e[i] + w[i] * y_o[i]
        y[i + n // 2] = y_e[i] - w[i] * y_o[i]
    return y


def fft_freq(n: int, sr: float, xs: np.ndarray = None) -> np.ndarray:
    '''
    Return the Discrete Fourier Transform sample frequencies.

    Args:
        `n`: window length
        `sr`: sample rate
        `xs`: the original array to transform to DFT frequencies

    Returns:
        An array of sample frequencies.
    '''

    if xs is None:
        xs = np.concatenate([
            np.arange(0, (n + 1) // 2, dtype=int),
            np.arange(-(n // 2), 0, dtype=int),
        ])
    return xs * sr / n
