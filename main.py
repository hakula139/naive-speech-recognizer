from typing import Union
from pathlib import Path

import numpy as np
import librosa

from utils import *


# Parameters
wav_path = Path('data/dev_set')
fig_path = Path('assets/spectrogram/dev_set')


def plot_waveform(path: Union[str, Path]) -> None:
    '''
    Plot the waveform of the audio file.

    Args:
        `path`: path to the input file
    '''

    y, sr = librosa.load(path, sr=None)
    print(f'Loaded audio "{path}" @ {sr} Hz.')

    n_samples = y.shape[0]
    t = np.arange(n_samples) / sr
    fig_time_path = fig_path / (Path(path).stem + '_time_domain.png')
    plot_time_domain(fig_time_path, t, y)
    print(f'Output figure to "{fig_time_path}".')


if __name__ == '__main__':
    if wav_path.exists():
        wav_paths = [entry for entry in wav_path.rglob('*.dat')]
        try:
            for p in wav_paths:
                plot_waveform(p)
        except KeyboardInterrupt:
            print('\nAborted.')
