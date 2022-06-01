from typing import List, NamedTuple, Tuple, Type

import torch
from torch import nn, Tensor, LongTensor
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader


class Result(NamedTuple):
    train_loss: Tensor
    valid_loss: Tensor
    valid_acc: Tensor


class ClassifierBase(nn.Module):

    def get_predictions(self, outputs: Tensor) -> LongTensor:
        return torch.argmax(outputs, dim=1)

    def get_accuracy(self, outputs: Tensor, labels: LongTensor) -> Tensor:
        preds = self.get_predictions(outputs)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def train_step(self, batch: Tuple[Tensor, LongTensor]) -> Tensor:
        data, labels = batch
        out: Tensor = self(data)
        loss = F.cross_entropy(out, labels)
        return loss

    def valid_step(self, batch: Tuple[Tensor, LongTensor]) -> Result:
        data, labels = batch
        out: Tensor = self(data)
        loss = F.cross_entropy(out, labels)
        acc = self.get_accuracy(out, labels)
        return Result(None, loss.detach(), acc)

    def train_epoch_end(self, epoch: int, result: Result) -> None:
        train_loss, valid_loss, valid_acc = result
        print('Epoch {}, train_loss: {:.4f}, valid_loss: {:.4f}, valid_acc: {:.2f}%'.format(
            epoch, train_loss.item(), valid_loss.item(), valid_acc.item() * 100,
        ))

    def valid_epoch_end(self, results: List[Result]) -> Result:
        batch_losses = [x.valid_loss for x in results]
        avg_loss = torch.stack(batch_losses).mean()
        batch_accs = [x.valid_acc for x in results]
        avg_acc = torch.stack(batch_accs).mean()
        return Result(None, avg_loss, avg_acc)


class Classifier(ClassifierBase):

    def __init__(self, data_len: int, label_size: int) -> None:

        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(32 * data_len, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, label_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@torch.no_grad()
def evaluate(model: Classifier, valid_dl: DataLoader) -> Result:
    model.eval()
    outputs = [model.valid_step(batch) for batch in valid_dl]
    return model.valid_epoch_end(outputs)


def fit(
    model: Classifier,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    num_epochs: int,
    lr: float,
    optim: Type[torch.optim.Optimizer]
) -> List[Result]:
    '''
    Train the model.

    Args:
        `model`: model to train
        `train_dl`: data loader of training set
        `valid_dl`: data loader of validation set
        `num_epochs`: number of epochs
        `lr`: learning rate
        `optim`: optimizer to use

    Return:
        The history of training process.
    '''

    optimizer = optim(model.parameters(), lr)

    history: List[Result] = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses: List[Tensor] = []

        for batch in train_dl:
            optimizer.zero_grad()
            loss = model.train_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()

        _, valid_loss, valid_acc = evaluate(model, valid_dl)
        train_loss = torch.stack(train_losses).mean()
        result = Result(train_loss, valid_loss, valid_acc)
        model.train_epoch_end(epoch, result)
        history.append(result)

    return history
