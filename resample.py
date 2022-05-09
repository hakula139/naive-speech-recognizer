import os
import sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf


# Parameters
default_sr = 8000


def resample(path: str, sr: int, write_to_file=False) -> np.ndarray:
    '''
    Resample the audio file to target sample rate.

    Args:
        `path`: path to the input file
        `sr`: target sample rate

    Returns:
        The resampled audio time series.
    '''

    # Resample to target sample_rate.
    y, sr = librosa.load(path, sr=sr)
    if write_to_file:
        p = Path(path)
        p_out = p.with_suffix('.dat')
        sf.write(p_out, y, sr, subtype='PCM_16', format='WAV')
        print(f'Output audio to {p_out}')
    return y


if __name__ == '__main__':
    try:
        wav_dir = sys.argv[1]
        sr = int(sys.argv[2]) if len(sys.argv) >= 3 else default_sr
        wav_count = 0

        for entry in os.scandir(wav_dir):
            if entry.is_file() and entry.path.endswith('.wav'):
                print(
                    f'Resampling audio {entry.path} to sample rate {sr} Hz...')
                resample(entry.path, sr, write_to_file=True)
                wav_count += 1
        print(f'Processed {wav_count} audio files.')

    except IndexError:
        print('Invalid arguments.')

    except KeyboardInterrupt:
        print('\nAborted.')
