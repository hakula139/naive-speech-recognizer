import numpy as np
import matplotlib.pyplot as plt


def plot_time_domain(output_path: str, t: np.ndarray, y: np.ndarray) -> None:
    '''
    Plot the amplitudes of a wave in time domain.

    Args:
        `output_path`: path to the output figure
        `t`: time of samples
        `y`: amplitudes of samples
    '''

    plt.figure()
    plt.title('Time Domain')
    plt.xlabel('Time / s')
    plt.ylabel('Amplitude')
    plt.plot(t, y, c='blue', label='signal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)

def plot_freq_domain(output_path: str, f: np.ndarray, y: np.ndarray) -> None:
    '''
    Plot the amplitude spectrum of a wave in frequency domain.

    Args:
        `output_path`: path to the output figure
        `f`: frequency range
        `y`: amplitude spectrum
    '''

    plt.figure()
    plt.title('Frequency Domain')
    plt.xlabel('Frequency / Hz')
    plt.ylabel('Amplitude')
    plt.plot(f, y, c='red', label='power')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
