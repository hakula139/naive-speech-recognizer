from typing import Tuple
import numpy as np
from torch import LongTensor, Tensor
from torch.utils.data import Dataset


class MfccDataset(Dataset):

    def __init__(self, data: np.ndarray, targets: np.ndarray) -> None:
        self.data = Tensor(data)
        self.targets = LongTensor(targets)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self) -> int:
        return len(self.data)
