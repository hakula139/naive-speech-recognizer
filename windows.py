import numpy as np


def hamming(m: int) -> np.ndarray:
    '''
    Return the Hamming window.

    Args:
        `m`: number of points in the output window

    Returns:
        The Hamming window of size `m`.
    '''

    if m < 1:
        return np.array([])
    if m == 1:
        return np.ones(1)
    n = np.arange(m)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (m - 1))
