from typing import List
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

from classifier import Classifier, fit
from dataset import MfccDataset
import utils


# Parameters
in_path = Path('tmp/dev_set')
random_seed = 233
train_ratio = 0.8
batch_size = 64
num_epochs = 30
learning_rate = 0.001

torch.manual_seed(random_seed)

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

    # Prepare dataset.

    assert len(mfcc_data) == len(meta_data), \
        'Dataset size and label size not match.'
    data_size = len(mfcc_data)
    dim_mfcc = mfcc_data[0].shape[0]
    max_len = max(cc.shape[1] for cc in mfcc_data)

    data = np.zeros((data_size, 1, dim_mfcc, max_len))
    for i, cc in enumerate(mfcc_data):
        data[i, 0, :, :cc.shape[1]] = cc

    meta_data = np.array(meta_data)
    targets = meta_data[:, 1]
    dataset = MfccDataset(data, targets)

    train_size = math.floor(data_size * train_ratio)
    valid_size = data_size - train_size
    train_data, valid_data = random_split(dataset, [train_size, valid_size])
    train_dl = DataLoader(
        train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )
    valid_dl = DataLoader(
        valid_data, batch_size, num_workers=4, pin_memory=True,
    )

    # Start training.

    model = Classifier(len(labels))
    history = fit(
        model,
        train_dl,
        valid_dl,
        num_epochs,
        learning_rate,
        torch.optim.Adam,
    )


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
