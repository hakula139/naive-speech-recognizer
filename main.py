from typing import List, Tuple, Union
from pathlib import Path

import numpy as np
import librosa

from fft import fft, fft_freq
import utils
from windows import hamming


# Parameters
wav_path = Path('data/dev_set')
fig_path = Path('assets/mfcc/dev_set')
t_window = 10  # milliseconds
amp_th = [2e-3, 6e-3]  # amplitude threshold for voice activity
zcr_th = 1. * 3  # zero-crossing rate (ZCR) threshold for voice activity
zcr_step_th = 5  # threshold of loop iterations when expanding ranges by ZCR


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


def detect_voice_activity(y: np.ndarray, n_window: int) -> np.ndarray:
    '''
    Detect voice activity in the audio signal.

    Args:
        `y`: time series of the audio signal
        `n_window`: the number of samples used in each window

    Returns:
        A list of ranges where voice activity is detected.
    '''

    n_samples = y.shape[0]
    i_starts = np.arange(0, n_samples, n_window // 2, dtype=int)
    i_starts = i_starts[i_starts + n_window < n_samples]

    avg_amps = [np.average(np.abs(y[i:i+n_window])) for i in i_starts]
    avg_zcrs = [np.sum(np.abs(
        np.sign(y[i:i+n_window-1]) - np.sign(y[i+1:i+n_window])
    )) / 2 / t_window for i in i_starts]
    fig_zcr_path = fig_path / 'zcr.png'
    utils.plot_zcr(fig_zcr_path, i_starts, avg_zcrs)

    # Step 1: Find the ranges by judging whether the average amplitude is
    # higher than threshold `amp_th[1]`.
    ranges_1: List[List[int]] = []
    for k, avg_amp in enumerate(avg_amps):
        if avg_amp > amp_th[1]:
            if len(ranges_1) > 0 and ranges_1[-1][1] >= k - 2:  # overlaps
                ranges_1[-1][1] = k
            else:
                ranges_1.append([k, k])

    # Step 2: Expand the ranges by judging whether the average amplitude is
    # higher than threshold `amp_th[0]`.
    ranges_2: List[List[int]] = []
    for r in ranges_1:
        i_start, i_stop = r[0], r[1]
        i_stop_prev = ranges_2[-1][1] if len(ranges_2) > 0 else 0
        while i_start > i_stop_prev and avg_amps[i_start] > amp_th[0]:
            i_start -= 1
        while i_stop < len(i_starts) - 1 and avg_amps[i_stop] > amp_th[0]:
            i_stop += 1
        if i_start <= i_stop_prev and i_stop_prev != 0:  # overlaps
            ranges_2[-1][1] = i_stop
        else:
            ranges_2.append([i_start, i_stop])

    # Step 3: Expand the ranges by judging whether the average zero-crossing
    # rate (ZCR) is higher than threshold `zcr_th`.
    ranges_3: List[List[int]] = []
    for r in ranges_2:
        i_start, i_stop = r[0], r[1]
        i_stop_prev = ranges_3[-1][1] if len(ranges_3) > 0 else 0
        i_start_min = max(i_stop_prev, r[0] - zcr_step_th)
        i_stop_max = min(len(i_starts) - 1, r[1] + zcr_step_th)
        while i_start > i_start_min and avg_zcrs[i_start] > zcr_th:
            i_start -= 1
        while i_stop < i_stop_max and avg_zcrs[i_stop] > zcr_th:
            i_stop += 1
        if i_start <= i_stop_prev and i_stop_prev != 0:  # overlaps
            ranges_3[-1][1] = i_stop
        else:
            ranges_3.append([i_start, i_stop])

    ranges = [[i_starts[r[0]], i_starts[r[1]] + n_window] for r in ranges_3]
    return np.array(ranges, dtype=float)


def plot_waveform(
    filename: str, y: np.ndarray, sr: int, ranges: np.ndarray = None,
) -> None:
    '''
    Plot the waveform of the audio signal.

    Args:
        `filename`: filename of the output figure
        `y`: time series of the audio signal
        `sr`: sample rate of the audio signal
        `ranges`: a list of ranges where voice activity is detected
    '''

    fig_time_path = fig_path / filename
    n_samples = y.shape[0]
    t = np.arange(n_samples) / sr
    if ranges is not None:
        ranges /= sr
    print(f'Detected voice activities: {ranges}.')
    utils.plot_time_domain(fig_time_path, t, y, ranges)
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
    n_fft = utils.next_pow2(n_window)
    zero_padding = np.zeros(n_fft - n_window)
    spec = np.array([np.abs(fft(
        np.concatenate((hamming(n_window) * y[i:i+n_window], zero_padding))
    )[:n_fft // 2]) for i in i_starts])
    # Rescale the absolute value of the spectrogram.
    spec = 10 * np.log10(spec.T + np.finfo(float).eps)
    return i_starts, spec


def plot_spectrogram(
    filename: str, i_starts: np.ndarray, spec: np.ndarray, sr: int,
) -> None:
    '''
    Plot the spectrogram of the audio signal.

    Args:
        `filename`: filename of the output figure
        `i_starts`: the starting indices of each window
        `spec`: the spectrogram to plot
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
                n_window = t_window * sr // 1000

                ranges = detect_voice_activity(y, n_window)
                fig_time_path = f'{p.stem}_time_domain.png'
                plot_waveform(fig_time_path, y, sr, ranges)

        except KeyboardInterrupt:
            print('\nAborted.')
