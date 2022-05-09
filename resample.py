import sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf


# Parameters
default_sr = 8000


def resample(path: str, sr: int, write_to_file = False) -> np.ndarray:
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
        p_out = p.with_stem(f'{p.stem}_{sr}')
        sf.write(p_out, y, sr)
        print(f'Output audio to {p_out}')
    return y


if __name__ == '__main__':
    try:
        path = sys.argv[1]
        sr = int(sys.argv[2]) if len(sys.argv) >= 3 else default_sr
        print(f'Resampling audio {path} to sample rate {sr} Hz')
        resample(path, sr, write_to_file=True)
    except IndexError:
        print('Invalid arguments.')
    except KeyboardInterrupt:
        print('\nAborted.')
