import math

import numpy as np
import matplotlib.pyplot as plt


def next_pow2(x: int) -> int:
    '''
    Round up to the next highest power of 2.

    Args:
        `x`: the integer to round up

    Returns:
        The next highest power of 2.
    '''

    return 1 << math.ceil(math.log2(x))


def plot_time_domain(output_path, t: np.ndarray, y: np.ndarray) -> None:
    '''
    Plot the amplitudes of a wave in time domain.

    Args:
        `output_path`: path to the output figure
        `t`: time of samples
        `y`: the amplitudes of samples
    '''

    plt.figure()
    plt.title('Time Domain')
    plt.xlabel('Time / s')
    plt.ylabel('Amplitude')
    plt.plot(t, y, c='blue', label='signal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)


def plot_freq_domain(output_path, f: np.ndarray, y: np.ndarray) -> None:
    '''
    Plot the amplitude spectrum of a wave in frequency domain.

    Args:
        `output_path`: path to the output figure
        `f`: frequency range
        `y`: the amplitude spectrum
    '''

    plt.figure()
    plt.title('Frequency Domain')
    plt.xlabel('Frequency / Hz')
    plt.ylabel('Amplitude')
    plt.plot(f, y, c='red', label='power')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)


def plot_spectrogram(
    output_path,
    spec: np.ndarray,
    xticks: np.ndarray,
    xlabels: np.ndarray,
    yticks: np.ndarray,
    ylabels: np.ndarray,
    n_fft: int,
) -> None:
    '''
    Plot the spectrogram of a wave.

    Args:
        `output_path`: path to the output figure
        `spec`: the spectrogram to plot
        `xticks`: tick locations of the x-axis
        `xlabels`: tick labels of the x-axis
        `yticks`: tick locations of the y-axis
        `ylabels`: tick labels of the y-axis
        `n_fft`: the number of samples for the FFT
    '''

    plt.figure()
    plt.title(f'Spectrogram ({n_fft} window size, hamming window)')
    plt.xticks(xticks, xlabels)
    plt.xlabel('Time / s')
    plt.yticks(yticks, ylabels)
    plt.ylabel('Frequency / Hz')
    plt.imshow(spec, origin='lower', aspect='auto')
    plt.colorbar(use_gridspec=True)
    plt.tight_layout()
    plt.savefig(output_path)
