from typing import Optional

import torch

from torch import nn, Tensor
from torch_geometric import nn as gnn
from torch_geometric.utils import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes

from .param_init import uniform


class GConv(gnn.MessagePassing):
    def __init__(
            self,
            in_channel: int, out_channel: int,
            num_cluster: int, num_roi: int,
            is_out_norm: bool = False,
            is_conv_bias: bool = True,
            is_out_activate: bool = True
    ):
        super().__init__(
            aggr='add',
            flow='source_to_target',
            node_dim=0
        )

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.is_out_norm = is_out_norm
        self.is_out_activate = is_out_activate
        self.is_conv_bias = is_conv_bias

        self.embed_linear = nn.Sequential(
            nn.Linear(num_roi, num_cluster, bias=False),
            nn.PReLU(),
            nn.Linear(num_cluster, self.in_channel * self.out_channel, bias=True)
        )
        self.conv_activate = nn.PReLU()

        if self.is_conv_bias:
            # bias vector for conv layer, shape = (out_channel,)
            self.bias = nn.Parameter(torch.Tensor(self.out_channel))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channel, self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, pos: Tensor) -> Tensor:
        # turn (edge_num, 1) to (edge_num,) vector
        edge_weight = torch.squeeze(edge_weight)

        # calcu weight linear combination represent community basis function
        # shape = (num_node, in_channel, out_channel)
        weight = self.embed_linear(pos.float()).view(-1, self.in_channel, self.out_channel)

        # (num_node, 1, out_channel) = (num_node, 1, in_channel) x (num_node, in_channel, out_channel)
        x_out = torch.matmul(x.unsqueeze(1), weight).squeeze(1)

        # call internal node message passing
        return self.propagate(
            edge_index=edge_index,
            size=None,
            x=x_out,
            edge_weight=edge_weight
        )

    @classmethod
    def group_norm(
            cls, src: Tensor, index: Optional[Tensor] = None,
            num_nodes: Optional[int] = None, dim: int = 0
    ):
        dim_size = maybe_num_nodes(index, num_nodes)
        src_abs = torch.abs(src)
        src_sum = scatter(src_abs, index, dim, dim_size=dim_size, reduce='sum').index_select(dim, index)

        return src / src_sum

    # noinspection PyMethodOverriding
    def message(
            self, edge_index_i: Tensor, edge_weight: Tensor,
            size_i: int, x_j: Tensor,
    ) -> Tensor:
        # send message from neighbor node j that belongs to Neighbor(i) to central node i

        # TODO choose message function
        # edge weight normalize
        # shape = (edge_num, 1)
        # 1. e_ij = e_ij / sum_{j in Neighbor(i)}(abs(e_ij))
        edge_weight_group_norm = self.group_norm(
            src=edge_weight,
            index=edge_index_i,
            num_nodes=size_i
        ).view(-1, 1)

        # 2. e_ij = group_softmax(E)
        # edge_weight_group_norm = pyg.utils.softmax(
        #     src=edge_weight,
        #     index=edge_index_i,
        #     num_nodes=size_i
        # ).view(-1, 1)

        # scale node feature by normed edge weight, use broadcasting element-wise mul
        # shape = (edge_num, out_channel)
        x_j_scaled = edge_weight_group_norm * x_j
        return x_j_scaled

    # noinspection PyMethodOverriding
    def update(self, inputs: Tensor, x: Tensor) -> Tensor:
        # input, x shape = (num_node, out_channel)
        x_new = inputs + x
        if self.bias is not None:
            x_new += self.bias
        if self.is_out_norm:
            # L2 normalization in feature wise
            x_new = nn.functional.normalize(x_new, p=2, dim=-1)

        if self.is_out_activate:
            x_aggr = self.conv_activate(x_new)
            return x_aggr

        return x_new

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(in_dim={self.in_channel}, out_dim={self.out_channel}, '
                f'norm={self.is_out_norm}, bias={self.is_conv_bias})')
