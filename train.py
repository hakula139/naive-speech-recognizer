from typing import List
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch

import utils


# Parameters
in_path = Path('tmp/dev_set')
random_seed = 233
train_ratio = 0.8

random.seed(random_seed)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

labels = [
    '数字', '语音', '语言', '识别', '中国',
    '忠告', '北京', '背景', '上海', '商行',
    'Speech', 'Speaker', 'Signal', 'Sequence', 'Process',
    'Print', 'Project', 'File', 'Open', 'Close',
]


def train(mfcc_data: List[np.ndarray], meta_data: List[np.ndarray]) -> None:

    assert(len(mfcc_data) == len(meta_data))
    meta_data = np.array(meta_data)

    person_labels: np.ndarray = meta_data[:, 0]
    word_labels: np.ndarray = meta_data[:, 1]

    batches = [(cc, word_labels[i]) for i, cc in enumerate(mfcc_data)]
    random.shuffle(batches)
    train_size = math.floor(len(batches) * train_ratio)
    train_set = batches[:train_size]
    valid_set = batches[train_size:]


def predict(mfcc_data: List[np.ndarray]) -> List[str]:

    preds = [labels[0] for cc in mfcc_data]
    return preds


if __name__ == '__main__':

    if not in_path.exists():
        sys.exit('MFCCs not found, please run main.py first.')

    mfcc_paths = list(in_path.iterdir())
    mfcc_data = [np.loadtxt(p / 'mfcc.txt').T for p in mfcc_paths]
    meta_data = [utils.get_meta_data(p.stem) for p in mfcc_paths]

    try:
        train(mfcc_data, meta_data)
    except KeyboardInterrupt:
        print('\nAborted.')
