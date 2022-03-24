import numpy as np
# import numpy.fft as nf
import librosa

from fft import fft, fft_freq
from utils import *


# Parameters
wav_path = 'data/signal.wav'
fig_time_path = 'assets/time_domain.png'
fig_freq_path = 'assets/freq_domain.png'
sample_rate = 8000.
n_samples = 1024


def main() -> None:
    # Resample to required sample_rate.
    y, sr = librosa.load(wav_path, sr=sample_rate)

    # Extract n_samples points.
    t0 = np.arange(n_samples) / sr
    y0 = y[:n_samples]
    plot_time_domain(fig_time_path, t0, y0)

    # Compute FFT.
    # y0_freqs = nf.fftfreq(n_samples, 1 / sr)
    y0_freqs = fft_freq(n_samples, sr)
    # y0_fft = np.abs(nf.fft(y0))
    y0_fft = np.abs(fft(y0))
    plot_freq_domain(
        fig_freq_path, y0_freqs[y0_freqs >= 0],  y0_fft[y0_freqs >= 0],
    )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nAborted.')
