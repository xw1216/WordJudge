import torch

import torch.nn as nn
from torch import Tensor
import torch_geometric as pyg
import torch_sparse as sparse
import torch_geometric.nn as gnn
from torch_geometric.data import Data

from .conv import GConv
from .pool import TopKPool, PoolSelector
from .readout import ReadOut


class BrainGNN(torch.nn.Module):
    def __init__(
            self, dim_in: int, num_class: int, num_cluster: int,
            pool_ratio: float, drop_ratio: float
    ):
        super().__init__()

        self.num_cluster = num_cluster
        self.num_roi = dim_in

        self.pool_ratio = pool_ratio

        self.dim_in = dim_in
        self.dim1 = 32
        self.dim2 = 32
        self.dim3 = 512
        self.dim_out = num_class

        self.conv1 = GConv(
            in_channel=self.dim_in,
            out_channel=self.dim1,
            num_cluster=self.num_cluster,
            num_roi=self.num_roi
        )
        self.pool1 = TopKPool(
            in_channels=self.dim1,
            ratio=self.pool_ratio,
            multiplier=1,
            nonlinearity=nn.Sigmoid
        )
        self.conv2 = GConv(
            in_channel=self.dim1,
            out_channel=self.dim2,
            num_cluster=self.num_cluster,
            num_roi=self.num_roi
        )
        self.pool2 = TopKPool(
            in_channels=self.dim2,
            ratio=self.pool_ratio,
            multiplier=1,
            nonlinearity=nn.Sigmoid
        )

        self.readout = ReadOut()

        self.mlp1 = nn.Sequential(
            nn.Linear((self.dim1 + self.dim2) * 2, self.dim2, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.dim2),
            nn.Dropout(p=drop_ratio)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(self.dim2, self.dim3, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.dim3),
            nn.Dropout(p=drop_ratio)
        )

        self.smx = nn.Sequential(
            nn.Linear(self.dim3, self.dim_out),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, data: Data):
        h1 = self.conv1(data.x, data.edge_index, data.edge_attr, data.pos)
        res1: PoolSelector = self.pool1(h1, data.edge_index, data.edge_attr, data.batch, data.pos)

        res1.edge_index, res1.edge_attr = BrainGNN.augment_adj_mat(
            res1.edge_index, res1.edge_attr.squeeze(), res1.x.size(0)
        )

        h2 = self.conv2(res1.x, res1.edge_index, res1.edge_attr, res1.pos)
        res2: PoolSelector = self.pool2(h2, res1.edge_index, res1.edge_attr, res1.batch, res1.pos)

        x = self.readout(res1.x, res1.batch, res2.x, res2.batch)

        x1 = self.mlp1(x)
        x2 = self.mlp2(x1)
        x = self.smx(x2)

        score1 = torch.sigmoid(res1.score).view(x.size(0), -1)
        score2 = torch.sigmoid(res2.score).view(x.size(0), -1)

        return x, self.pool1.weight, self.pool2.weight, score1, score2

    @classmethod
    def augment_adj_mat(cls, edge_index, edge_weight, num_nodes):
        # add node self loop with 1
        edge_index, edge_weight = pyg.utils.add_self_loops(edge_index, edge_weight, num_nodes=num_nodes)
        edge_index, edge_weight = pyg.utils.sort_edge_index(edge_index, edge_weight, num_nodes)

        # square of edge adj matrix
        edge_index, edge_weight = sparse.spspmm(
            edge_index, edge_weight,
            edge_index, edge_weight,
            num_nodes, num_nodes, num_nodes
        )
        edge_index, edge_weight = pyg.utils.remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight


