from typing import Union
from concurrent import futures
from pathlib import Path
import soundfile as sf
import sys

import numpy as np
import librosa


# Parameters
default_sr = 8000
timeout = 5  # second


def resample(path: Union[str, Path], sr: int, write_to_file=False) -> np.ndarray:
    '''
    Resample the audio file to target sample rate.

    Args:
        `path`: path to the input file
        `sr`: target sample rate

    Returns:
        The resampled audio time series.
    '''

    print(f'Resampling audio "{path}" to sample rate {sr} Hz...')
    y, sr = librosa.load(path, sr=sr)
    if write_to_file:
        p_out = Path(path).with_suffix('.dat')
        sf.write(p_out, y, sr, subtype='PCM_16', format='WAV')
        print(f'Output audio to "{p_out}".')
    return y


if __name__ == '__main__':
    try:
        wav_path = Path(sys.argv[1])
        sr = int(sys.argv[2]) if len(sys.argv) >= 3 else default_sr
    except IndexError:
        sys.exit('Invalid arguments.')

    with futures.ThreadPoolExecutor() as e:
        results = []
        if wav_path.is_file():
            wav_paths = [wav_path]
        elif wav_path.is_dir():
            wav_paths = [entry for entry in wav_path.rglob('*.wav')]
        else:
            wav_paths = []
        results = [e.submit(resample, p, sr, True) for p in wav_paths]

        success_count = 0
        try:
            for r in futures.as_completed(results):
                r.result(timeout)
                success_count += 1
        except TimeoutError:
            print('Timeout.')
        except KeyboardInterrupt:
            for r in results:
                r.cancel()
            print('\nAborted.')

        print(f'Processed {success_count} audio files.')
