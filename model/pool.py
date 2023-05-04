from typing import Union, Callable

import torch

from torch import Tensor
import torch_geometric.nn as gnn
from torch_geometric.nn.pool.topk_pool import topk, filter_adj


class PoolSelector:
    def __init__(
            self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
            batch: Tensor, pos: Tensor, score: Tensor
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.batch = batch
        self.pos = pos
        self.score = score


class TopKPool(gnn.TopKPooling):
    r"""
    .. math::
            \mathbf{y} &= \frac{\mathbf{X}\mathbf{w}}{\| \mathbf{w} \|}

            \mathbf{y}^{\prime} &= \frac{\mathbf{y} - mean(\mathbf{y})}{var(\mathbf{y})}

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y}^{\prime})

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot
            \mathrm{sigmoid}(\mathbf{y}^{\prime}))_{\mathbf{i}}

            \mathbf{E}^{\prime} &= \mathbf{E}_{\mathbf{i},\mathbf{i}}
    """
    def __init__(self, in_channels: int, ratio: Union[int, float] = 0.5,
                 multiplier: float = 1., nonlinearity: Union[str, Callable] = 'sigmoid'):
        super().__init__(in_channels, ratio, None, multiplier, nonlinearity)

    # noinspection PyMethodOverriding
    def forward(
            self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
            batch: Tensor, pos: Tensor
    ) -> PoolSelector:
        attn = x
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn

        # s after L2 norm
        score = (attn * self.weight).sum(dim=-1)
        score = score / self.weight.norm(p=2, dim=-1)

        # s after normal distribution transform
        score = (score - torch.mean(score.detach(), dim=0, keepdim=False)) / torch.var(
            score.detach(), dim=0, keepdim=False)

        # s applied with additional non-linear transform
        score = self.nonlinearity(score)

        # select top K of num * ratio, return perm selector
        perm = topk(score, self.ratio, batch, self.min_score)

        # scale x by score and multiplier
        x = x[perm] * score[perm].view(-1, 1)
        x_out = self.multiplier * x if self.multiplier != 1 else x

        # select remaining edge
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

        # TODO choose score index and check score loss chain
        return PoolSelector(x_out, edge_index, edge_attr, batch[perm], pos[perm], score)
        # return PoolSelector(x, edge_index, edge_attr, batch[perm], pos[perm], score[perm])
