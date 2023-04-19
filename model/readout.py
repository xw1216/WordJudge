import torch

import torch_geometric.nn as gnn
from torch import Tensor


class ReadOut(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(
            self,
            x_1: Tensor, batch_1: Tensor,
            x_2: Tensor, batch_2: Tensor
    ):
        read_1 = torch.cat((
            gnn.global_mean_pool(x_1, batch_1),
            gnn.global_max_pool(x_1, batch_1)
        ), dim=1)

        read_2 = torch.cat((
            gnn.global_mean_pool(x_2, batch_2),
            gnn.global_max_pool(x_2, batch_2)
        ), dim=1)

        return torch.cat((read_1, read_2), dim=1)

    def __repr__(self):
        return f'{self.__class__.__name__}(mean || max)'
