from typing import Iterator

import torch
from torch.optim import Adam, SGD, AdamW
from torch.nn.parameter import Parameter


def build_optimizer(
        name: str, param: Iterator[Parameter], lr: float,
        weight_decay: float, momentum: float = 0.9
):
    assert name in ['Adam', 'SGD', 'AdamW']

    if name == 'Adam':
        return Adam(
            params=param, lr=lr, weight_decay=weight_decay
        )
    elif name == 'SGD':
        return SGD(
            params=param, lr=lr, momentum=momentum,
            weight_decay=weight_decay, nesterov=True
        )
    else:
        return AdamW(params=param, lr=lr, weight_decay=weight_decay)

