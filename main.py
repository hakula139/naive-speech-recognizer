from typing import Tuple, Union
from pathlib import Path

import numpy as np
import librosa

from fft import fft, fft_freq
import utils


# Parameters
wav_path = Path('data/dev_set')
fig_path = Path('assets/spectrogram/dev_set')
t_windows = [5, 10, 15]  # millisecond


def load_audio(path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    '''
    Load the audio file from path.

    Args:
        `path`: path to the input file

    Returns:
        `y`: time series of the audio signal
        `sr`: sample rate of the audio signal
    '''

    y, sr = librosa.load(path, sr=None)
    print(f'Loaded audio "{path}" @ {sr} Hz.')
    return y, sr


def plot_waveform(filename: str, y: np.ndarray, sr: int) -> None:
    '''
    Plot the waveform of the audio signal.

    Args:
        `filename`: filename of the output figure
        `y`: time series of the audio signal
        `sr`: sample rate of the audio signal
    '''

    fig_time_path = fig_path / filename
    n_samples = y.shape[0]
    t = np.arange(n_samples) / sr
    utils.plot_time_domain(fig_time_path, t, y)
    print(f'Output figure to "{fig_time_path}".')


def create_spectrogram(y: np.ndarray, n_window: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Create the spectrogram of the audio signal.

    Args:
        `y`: time series of the audio signal
        `n_window`: the number of samples used in each window

    Returns:
        `i_starts`: the starting indices of each window
        `spec`: the spectrum of frequencies of the audio signal as it varies with time
    '''

    n_samples = y.shape[0]
    i_starts = np.arange(0, n_samples, n_window // 2, dtype=int)
    i_starts = i_starts[i_starts + n_window < n_samples]
    n_fft = utils.round_up(n_window)
    zero_padding = np.zeros(n_fft - n_window)
    spec = np.array([np.abs(
        fft(np.concatenate((y[i:i+n_window], zero_padding)))[:n_fft // 2]
    ) for i in i_starts])
    # Rescale the absolute value of the spectrogram.
    spec = 10 * np.log10(spec.T + np.finfo(float).eps)
    return i_starts, spec


def plot_spectrogram(
    filename: str, i_starts: np.ndarray, spec: np.ndarray, n_samples: int, sr: int
) -> None:
    '''
    Plot the spectrogram of the audio signal.

    Args:
        `filename`: filename of the output figure
        `i_starts`: the starting indices of each window
        `spec`: the spectrogram to plot
        `n_samples`: the number of samples
        `sr`: sample rate
    '''

    fig_spec_path = fig_path / filename
    xticks = np.linspace(0, spec.shape[1], 10)
    xlabels = [f'{i:4.2f}' for i in np.linspace(0, i_starts[-1] / sr, 10)]
    yticks = np.linspace(0, spec.shape[0], 10)
    ylabels = np.floor(fft_freq(spec.shape[0], sr, yticks)).astype(int)
    utils.plot_spectrogram(
        fig_spec_path, spec, xticks, xlabels, yticks, ylabels, n_window,
    )
    print(f'Output figure to "{fig_spec_path}".')


if __name__ == '__main__':
    if wav_path.exists():
        wav_paths = [entry for entry in wav_path.rglob('*.dat')]
        try:
            for p in wav_paths:
                y, sr = load_audio(p)
                fig_time_path = f'{p.stem}_time_domain.png'
                plot_waveform(fig_time_path, y, sr)
                for t_window in t_windows:
                    n_window = t_window * sr // 1000
                    i_starts, spec = create_spectrogram(y, n_window)
                    fig_spec_path = f'{p.stem}_spec_domain_{t_window}ms.png'
                    plot_spectrogram(
                        fig_spec_path, i_starts, spec, y.shape[0], sr,
                    )

        except KeyboardInterrupt:
            print('\nAborted.')
