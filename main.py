from typing import List, Tuple, Union
from functools import reduce
from pathlib import Path
from multiprocessing import Pool
import signal
import sys

import numpy as np
import librosa

from fft import fft, fft_freq
from model import labels, Model
from mfcc import dct, get_mel_filters
import utils
from windows import hamming


# Parameters
train_in_path = Path('data/dev_set')
test_in_path = Path('data/test_set')
out_path = Path('tmp')
timeout = 5    # seconds
t_window = 16  # milliseconds
pre_emphasis = 0.97
amp_th = [1e-2, 3e-2]  # amplitude threshold for voice activity
zcr_th = 5       # zero-crossing rate (ZCR) threshold for voice activity
zcr_step_th = 5  # threshold of loop iterations when expanding ranges by ZCR
overlap_th = 20  # threshold of range overlap judgement
n_mel_filters = 14
dim_mfcc = 13  # dimension of the Mel-frequency cepstral coefficients (MFCCs)


def load_audio(path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    '''
    Load the audio file from path.

    Args:
        `path`: path to the input file

    Returns:
        `y`: time series of the audio signal
        `sr`: sample rate of the audio signal
    '''

    # print(f'[INFO ] Loading audio "{path}".')
    y, sr = librosa.load(path, sr=None)
    print(f'[INFO ] Loaded audio "{path}" @ {sr} Hz.')
    return y, sr


def detect_voice_activity(
    y: np.ndarray, n_window: int,
) -> Tuple[List[List[int]], np.ndarray, List[float]]:
    '''
    Detect voice activity in the audio signal.

    Args:
        `y`: time series of the audio signal
        `n_window`: the number of samples used in each window

    Returns:
        `ranges`: a list of ranges where voice activity is detected
        `i_starts`: the starting indices of each window
        `avg_zcrs`: the average ZCR of the audio signal as it varies with time
    '''

    n_samples = y.shape[0]
    i_starts = np.arange(0, n_samples, n_window // 2, dtype=int)
    i_starts: np.ndarray = i_starts[i_starts + n_window < n_samples]

    avg_amps = [np.average(np.abs(y[i:i+n_window])) for i in i_starts]
    avg_zcrs = [np.sum(np.abs(
        np.sign(y[i:i+n_window-1]) - np.sign(y[i+1:i+n_window])
    )) / 2 / t_window for i in i_starts]

    # Step 1: Find the ranges by judging whether the average amplitude is
    # higher than threshold `amp_th[1]`.
    ranges_1: List[List[int]] = []
    for k, avg_amp in enumerate(avg_amps):
        if avg_amp > amp_th[1]:
            if len(ranges_1) > 0 and k <= ranges_1[-1][1] + overlap_th:
                ranges_1[-1][1] = k
            else:
                ranges_1.append([k, k])

    # Step 2: Expand the ranges by judging whether the average amplitude is
    # higher than threshold `amp_th[0]`.
    ranges_2: List[List[int]] = []
    for r in ranges_1:
        i_start, i_stop = r
        i_stop_prev = ranges_2[-1][1] if len(ranges_2) > 0 else 0
        while i_start > i_stop_prev and avg_amps[i_start] > amp_th[0]:
            i_start -= 1
        while i_stop < len(i_starts) - 1 and avg_amps[i_stop] > amp_th[0]:
            i_stop += 1
        if i_start <= i_stop_prev + overlap_th and i_stop_prev != 0:
            ranges_2[-1][1] = i_stop
        else:
            ranges_2.append([i_start, i_stop])

    # Step 3: Expand the ranges by judging whether the average zero-crossing
    # rate (ZCR) is higher than threshold `zcr_th`.
    ranges_3: List[List[int]] = []
    for r in ranges_2:
        i_start, i_stop = r
        i_stop_prev = ranges_3[-1][1] if len(ranges_3) > 0 else 0
        i_start_min = max(i_stop_prev, r[0] - zcr_step_th)
        i_stop_max = min(len(i_starts) - 1, r[1] + zcr_step_th)
        while i_start > i_start_min and avg_zcrs[i_start] > zcr_th:
            i_start -= 1
        while i_stop < i_stop_max and avg_zcrs[i_stop] > zcr_th:
            i_stop += 1
        if i_start <= i_stop_prev + overlap_th and i_stop_prev != 0:
            ranges_3[-1][1] = i_stop
        else:
            ranges_3.append([i_start, i_stop])

    # print(ranges_1, ranges_2, ranges_3, sep='\n')
    ranges = [[i_starts[r[0]], i_starts[r[1]] + n_window] for r in ranges_3]
    return ranges, i_starts, avg_zcrs


def plot_waveform(
    filename: str, y: np.ndarray, sr: int, ranges: List[List[int]] = None,
) -> None:
    '''
    Plot the waveform of the audio signal.

    Args:
        `filename`: filename of the output figure
        `y`: time series of the audio signal
        `sr`: sample rate of the audio signal
        `ranges`: a list of ranges where voice activity is detected
    '''

    fig_time_path = out_path / filename
    n_samples = y.shape[0]
    t = np.arange(n_samples) / sr
    if ranges is not None:
        ranges = np.array(ranges, dtype=float) / sr
    # print(f'[INFO ] Detected voice activities: {ranges}')
    if len(ranges) > 1:
        print(f'[WARN] Improper voice activities found: {ranges}')
    utils.plot_time_domain(fig_time_path, t, y, ranges)
    # print(f'[INFO ] Output figure to "{fig_time_path}".')


def plot_zcr(filename: str, i_starts: np.ndarray, zcr: np.ndarray) -> None:
    '''
    Plot the average zero-crossing rate (ZCR) of a wave in time domain.

    Args:
        `filename`: filename of the output figure
        `i_starts`: the starting indices of each window
        `zcr`: the average ZCR of the audio signal as it varies with time
    '''

    fig_zcr_path = out_path / filename
    utils.plot_zcr(fig_zcr_path, i_starts, zcr)
    # print(f'[INFO ] Output figure to "{fig_zcr_path}".')


def plot_mel_filters(filename: str, filters: np.ndarray) -> None:
    '''
    Plot the Mel filter banks.

    Args:
        `filename`: filename of the output figure
        `filters`: the function values of each Mel filter in frequency domain
    '''

    fig_filters_path = out_path / filename
    utils.plot_mel_filters(fig_filters_path, filters)
    # print(f'[INFO ] Output figure to "{fig_filters_path}".')


def create_spectrogram(y: np.ndarray, n_window: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Create the spectrogram of the audio signal.

    Args:
        `y`: time series of the audio signal
        `n_window`: the number of samples used in each window

    Returns:
        `spec`: the spectrum of frequencies of the audio signal as it varies with time,
                shape(`n_window` / 2, `len(i_starts)`)
        `i_starts`: the starting indices of each window
    '''

    n_samples = y.shape[0]
    i_starts = np.arange(0, n_samples, n_window // 2, dtype=int)
    i_starts: np.ndarray = i_starts[i_starts + n_window < n_samples]
    h = hamming(n_window)
    spec = np.array([
        np.abs(fft(h * y[i:i+n_window])[:n_window // 2])
        for i in i_starts
    ])
    spec = spec.T + np.finfo(float).eps
    return spec, i_starts


def plot_spectrogram(
    filename: str, i_starts: np.ndarray, spec: np.ndarray, sr: int, n_window: int,
) -> None:
    '''
    Plot the spectrogram of the audio signal.

    Args:
        `filename`: filename of the output figure
        `i_starts`: the starting indices of each window
        `spec`: the spectrogram to plot
        `sr`: sample rate
        `n_window`: the number of samples used in each window
    '''

    fig_spec_path = out_path / filename
    xticks = np.linspace(0, spec.shape[1], 10)
    xlabels = [f'{i:4.2f}' for i in np.linspace(0, i_starts[-1] / sr, 10)]
    yticks = np.linspace(0, spec.shape[0], 9)
    ylabels = np.floor(fft_freq(spec.shape[0], sr, yticks)).astype(int)
    utils.plot_spectrogram(
        fig_spec_path, spec, xticks, xlabels, yticks, ylabels, n_window,
    )
    # print(f'[INFO ] Output figure to "{fig_spec_path}".')


def plot_energy_spec(
    filename: str, i_starts: np.ndarray, spec: np.ndarray, sr: int, n_window: int,
) -> None:
    '''
    Plot the energy spectrum of the audio signal.

    Args:
        `filename`: filename of the output figure
        `i_starts`: the starting indices of each window
        `spec`: the energy spectrum to plot
        `sr`: sample rate
        `n_window`: the number of samples used in each window
    '''

    fig_spec_path = out_path / filename
    xticks = np.linspace(0, spec.shape[1], 10)
    xlabels = [f'{i:4.2f}' for i in np.linspace(0, i_starts[-1] / sr, 10)]
    yticks = np.linspace(0, spec.shape[0], 9)
    ylabels = np.floor(fft_freq(spec.shape[0], sr, yticks)).astype(int)
    utils.plot_energy_spec(
        fig_spec_path, spec, xticks, xlabels, yticks, ylabels, n_window,
    )
    # print(f'[INFO ] Output figure to "{fig_spec_path}".')


def plot_mfcc(filename: str, mfcc: np.ndarray) -> None:
    '''
    Plot the Mel-frequency cepstral coefficients (MFCCs).

    Args:
        `filename`: filename of the output figure
        `mfcc`: the MFCC to plot
    '''

    fig_mfcc_path = out_path / filename
    utils.plot_mfcc(fig_mfcc_path, mfcc)
    # print(f'[INFO ] Output figure to "{fig_mfcc_path}".')


def store_mfcc(filename: str, mfcc: np.ndarray) -> None:
    '''
    Store the Mel-frequency cepstral coefficients (MFCCs) to local file.

    Args:
        `filename`: filename of the output file
        `mfcc`: the MFCC to store
    '''

    txt_mfcc_path = out_path / filename
    np.savetxt(txt_mfcc_path, mfcc.T, fmt='%.7f')
    # print(f'[INFO ] Output data to "{txt_mfcc_path}".')


def get_mfcc(path: Path) -> np.ndarray:
    '''
    Load the audio file from path, and calculate its MFCC.

    Args:
        `path`: path to the input file

    Returns:
        The MFCC of the audio signal.
    '''

    filename = path.stem
    (out_path / filename).mkdir(parents=True, exist_ok=True)

    # Load audio signal from disk.
    y, sr = load_audio(path)

    # Pre-emphasize and normalize the signal.
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    y = y / np.max(np.abs(y))

    # Get the window length for FFT.
    n_window = t_window * sr // 1000
    n_window = utils.next_pow2(n_window)

    # Detect the ranges of voice activity.
    ranges, i_starts, zcr = detect_voice_activity(y, n_window)
    fig_zcr_path = filename + '/zcr.png'
    plot_zcr(fig_zcr_path, i_starts, zcr)

    fig_time_path = filename + '/time_domain.png'
    plot_waveform(fig_time_path, y, sr, ranges)

    # Obtain the Mel filter banks.
    f_min, f_max = 20, sr // 2
    filters = get_mel_filters(
        n_mel_filters, sr, n_window, f_min, f_max,
    )
    # fig_filters_path = f'mel_filters_{n_window}_{f_min}-{f_max}.png'
    # plot_mel_filters(fig_filters_path, filters)

    r = reduce(
        lambda x, y: y if x[1] - x[0] < y[1] - y[0] else x, ranges,
    )

    # Get the spectrogram using STFT.
    spec, i_starts = create_spectrogram(y[r[0]:r[1]], n_window)
    energy_spec = np.square(spec)
    # log_spec = 10 * np.log10(spec)
    # fig_spec_path = f'{filename}/spectrogram_{t_window}ms_hamming.png'
    # plot_spectrogram(
    #     fig_spec_path, i_starts, log_spec, sr, n_window,
    # )

    # Filter the energy spectrum with the Mel filter banks.
    filtered_spec = np.dot(filters, energy_spec)
    log_filtered_spec = 10 * np.log10(filtered_spec)
    # fig_filtered_spec_path = f'{filename}/energy_spec_{t_window}ms_hamming_filtered.png'
    # plot_energy_spec(
    #     fig_filtered_spec_path,
    #     i_starts, log_filtered_spec, sr, n_window,
    # )

    # Generate the MFCC.
    cc = dct(log_filtered_spec, dim_mfcc)
    # fig_mfcc_path = filename + '/mfcc.png'
    # plot_mfcc(fig_mfcc_path, cc)
    txt_mfcc_path = filename + '/mfcc.txt'
    store_mfcc(txt_mfcc_path, cc)
    return cc


def batch_get_mfcc(paths: List[Path]) -> List[np.ndarray]:

    mfcc_data = []
    sig_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    with Pool() as pool:
        signal.signal(signal.SIGINT, sig_handler)
        results = [pool.apply_async(get_mfcc, args=(p,)) for p in paths]
        try:
            mfcc_data = [res.get(timeout) for res in results]
        except TimeoutError:
            print('\n[FATAL] Timeout.')
        except KeyboardInterrupt:
            print('\n[INFO ] Aborted.')
    return mfcc_data


if __name__ == '__main__':

    # Training

    if not train_in_path.exists():
        sys.exit('Training set not found.')
    model = Model()
    train_paths = list(train_in_path.rglob('*.dat'))
    meta_data = [utils.get_meta_data(p.stem) for p in train_paths]
    history = model.train(batch_get_mfcc(train_paths), meta_data)

    # Testing

    if not test_in_path.exists():
        sys.exit('Testing set not found.')
    test_paths = list(test_in_path.rglob('*.dat'))
    meta_data = [utils.get_meta_data(p.stem) for p in test_paths]
    preds = model.predict(batch_get_mfcc(test_paths))
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    for i, pred in enumerate(preds):
        expected = meta_data[i][1]
        confusion[expected, pred] += 1
    print(confusion)
