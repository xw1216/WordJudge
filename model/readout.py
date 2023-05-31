import torch

import torch_geometric.nn as gnn
from torch import Tensor

from .pool import PoolSelector


class ReadOut(torch.nn.Module):
    def __init__(self, conv_len):
        super().__init__()
        self.conv_len = conv_len

    # noinspection PyMethodMayBeStatic
    def forward(
            self,
            select: list[PoolSelector],
    ):
        read = []

        for i in range(self.conv_len):
            read.append(
                torch.cat((
                    gnn.global_mean_pool(select[i].x, select[i].batch),
                    gnn.global_max_pool(select[i].x, select[i].batch)
                ), dim=1)
            )
        if len(read) < 2:
            return read
        else:
            return torch.cat(read, dim=1)

    def __repr__(self):
        return f'{self.__class__.__name__}(mean || max)'
