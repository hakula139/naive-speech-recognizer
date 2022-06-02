from typing import List
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

from classifier import Classifier, Result, fit
from dataset import MfccDataset
import utils


# Parameters
in_path = Path('tmp')
random_seed = 233
train_ratio = 0.8
batch_size = 256
num_epochs = 30
learning_rate = 0.001

labels = [
    '数字', '语音', '语言', '识别', '中国',
    '忠告', '北京', '背景', '上海', '商行',
    'Speech', 'Speaker', 'Signal', 'Sequence', 'Process',
    'Print', 'Project', 'File', 'Open', 'Close',
]


class Model():

    def __init__(self) -> None:

        torch.manual_seed(random_seed)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.model: Classifier = None
        self.data_len = 0

    def train(
        self, mfcc_data: List[np.ndarray], meta_data: List[np.ndarray],
    ) -> List[Result]:

        # Prepare dataset.

        assert len(mfcc_data) == len(meta_data), \
            'Dataset size and label size not match.'
        data_size = len(mfcc_data)
        if data_size == 0:
            return []
        dim_mfcc = mfcc_data[0].shape[0]
        self.data_len = utils.next_pow2(max(cc.shape[1] for cc in mfcc_data))

        data = np.zeros((data_size, 1, dim_mfcc, self.data_len))
        for i, cc in enumerate(mfcc_data):
            data[i, 0, :, :cc.shape[1]] = cc

        meta_data = np.array(meta_data)
        targets = meta_data[:, 1]
        dataset = MfccDataset(data, targets)

        train_size = math.floor(data_size * train_ratio)
        valid_size = data_size - train_size
        train_data, valid_data = random_split(
            dataset, [train_size, valid_size],
        )
        train_dl = DataLoader(
            train_data, batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=False,
        )
        valid_dl = DataLoader(
            valid_data, batch_size,
            num_workers=4, pin_memory=True, drop_last=False,
        )

        # Start training.

        self.model = Classifier(
            self.data_len, len(labels), self.device,
        ).to(self.device)

        history = fit(
            self.model,
            train_dl,
            valid_dl,
            num_epochs,
            learning_rate,
            torch.optim.Adam,
        )
        return history

    def predict(self, mfcc_data: List[np.ndarray]) -> List[int]:

        # Prepare dataset.

        data_size = len(mfcc_data)
        if data_size == 0:
            return []
        dim_mfcc = mfcc_data[0].shape[0]

        data = np.zeros((data_size, 1, dim_mfcc, self.data_len))
        for i, cc in enumerate(mfcc_data):
            data[i, 0, :, :cc.shape[1]] = cc[:self.data_len]

        # Start predicting.

        data = Tensor(data).to(self.device)
        out: Tensor = self.model(data)
        preds = self.model.get_predictions(out).tolist()
        return preds


if __name__ == '__main__':

    if not in_path.exists():
        sys.exit('MFCCs not found, please run main.py first.')
    model = Model()
    mfcc_paths = list(in_path.iterdir())
    mfcc_data = [np.loadtxt(p / 'mfcc.txt').T for p in mfcc_paths]
    meta_data = [utils.get_meta_data(p.stem) for p in mfcc_paths]

    try:
        history = model.train(mfcc_data, meta_data)
    except KeyboardInterrupt:
        print('\n[INFO ] Aborted.')
