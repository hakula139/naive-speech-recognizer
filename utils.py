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


def plot_time_domain(
    output_path, t: np.ndarray, y: np.ndarray, ranges: np.ndarray = None,
) -> None:
    '''
    Plot the amplitudes of a wave in time domain.

    Args:
        `output_path`: path to the output figure
        `t`: time of samples
        `y`: the amplitudes of samples
        `ranges`: a list of ranges where voice activity is detected
    '''

    plt.figure(figsize=(15, 4))
    plt.title('Time Domain')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.plot(t, y, c='blue')
    if ranges is not None:
        for start, stop in ranges:
            plt.axvspan(start, stop, facecolor='r', alpha=0.2)
            plt.axvline(x=start, linestyle='--')
            plt.axvline(x=stop, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_freq_domain(output_path, f: np.ndarray, y: np.ndarray) -> None:
    '''
    Plot the amplitude spectrum of a wave in frequency domain.

    Args:
        `output_path`: path to the output figure
        `f`: frequency range
        `y`: the amplitude spectrum
    '''

    plt.figure(figsize=(15, 4))
    plt.title('Frequency Domain')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.plot(f, y, c='red')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_spectrogram(
    output_path,
    spec: np.ndarray,
    xticks: np.ndarray,
    xlabels: np.ndarray,
    yticks: np.ndarray,
    ylabels: np.ndarray,
    n_window: int,
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
        `n_window`: the number of samples used in each window
    '''

    plt.figure(figsize=(9, 6))
    plt.title(f'Spectrogram ({n_window} window size, Hamming window)')
    plt.xticks(xticks, xlabels)
    plt.xlabel('Time [s]')
    plt.yticks(yticks, ylabels)
    plt.ylabel('Frequency [Hz]')
    plt.imshow(spec, origin='lower', aspect='auto')
    plt.colorbar(use_gridspec=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_energy_spec(
    output_path,
    spec: np.ndarray,
    xticks: np.ndarray,
    xlabels: np.ndarray,
    yticks: np.ndarray,
    ylabels: np.ndarray,
    n_window: int,
) -> None:
    '''
    Plot the energy spectrum of a wave.

    Args:
        `output_path`: path to the output figure
        `spec`: the energy spectrum to plot
        `xticks`: tick locations of the x-axis
        `xlabels`: tick labels of the x-axis
        `yticks`: tick locations of the y-axis
        `ylabels`: tick labels of the y-axis
        `n_window`: the number of samples used in each window
    '''

    plt.figure(figsize=(9, 6))
    plt.title(f'Energy Spectrum ({n_window} window size, Hamming window, Mel filtered)')
    plt.xticks(xticks, xlabels)
    plt.xlabel('Time [s]')
    plt.yticks(yticks, ylabels)
    plt.ylabel('Frequency [Hz]')
    plt.imshow(spec, origin='lower', aspect='auto')
    plt.colorbar(use_gridspec=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_zcr(output_path, t: np.ndarray, y: np.ndarray) -> None:
    '''
    Plot the average zero-crossing rate (ZCR) of a wave in time domain.

    Args:
        `output_path`: path to the output figure
        `t`: the indices of samples
        `y`: the average ZCR curve to plot
    '''

    plt.figure(figsize=(15, 4))
    plt.title('Zero-Crossing Rate (ZCR)')
    plt.xlabel('Sample index')
    plt.ylabel('ZCR [kHz]')
    plt.plot(t, y)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_mel_filters(output_path, y: np.ndarray) -> None:
    '''
    Plot the Mel filter banks.

    Args:
        `output_path`: path to the output figure
        `y`: the Mel filter banks to plot
    '''

    plt.figure(figsize=(15, 4))
    plt.title('Mel Filter Banks')
    plt.xlabel(r'$f(k)$')
    plt.ylabel(r'$H_m(k)$')
    for n in range(y.shape[0]):
        plt.plot(y[n])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_mfcc(output_path, y: np.ndarray) -> None:
    '''
    Plot the Mel-frequency cepstral coefficients (MFCCs).

    Args:
        `output_path`: path to the output figure
        `y`: the MFCC to plot
    '''

    plt.figure()
    plt.title('Mel-Frequency Cepstral Coefficients')
    plt.xlabel('Windows')
    plt.ylabel('Dimensions')
    plt.imshow(y, origin='lower', aspect='auto')
    plt.colorbar(use_gridspec=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
