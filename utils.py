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

    plt.title('Time Domain')
    plt.xlabel('Time / s')
    plt.ylabel('Amplitude')
    plt.plot(t, y, label='signal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
