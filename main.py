import numpy as np
import librosa

from utils import plot_time_domain


# Parameters
wav_path = 'data/signal.wav'
fig_time_path = 'assets/time_domain.png'
sample_rate = 8000.
n_samples = 1024


def main() -> None:
    # Resample to required sample_rate.
    y, sr = librosa.load(wav_path, sr=sample_rate)

    # Extract n_samples points.
    t0 = np.arange(0, n_samples, 1) / sr
    y0 = y[:n_samples]
    plot_time_domain(fig_time_path, t0, y0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nAborted.')
