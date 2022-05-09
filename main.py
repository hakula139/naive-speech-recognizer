import numpy as np
import librosa

from utils import *


# Parameters
wav_path = 'data/signal_8000.wav'
fig_time_path = 'assets/spectrogram/time_domain.png'
fig_freq_path = 'assets/spectrogram/freq_domain.png'


def main() -> None:
    y: np.ndarray
    sr: int
    y, sr = librosa.load(wav_path, sr=None)
    print(f'Loaded audio {wav_path} @ {sr} Hz')

    # Extract n_samples points.
    n_samples = y.shape[0]
    t = np.arange(n_samples) / sr
    plot_time_domain(fig_time_path, t, y)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nAborted.')
