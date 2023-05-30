import torch

import torch.nn as nn
import torch_geometric as pyg
import torch_sparse as sparse

from .conv import GConv
from .pool import TopKPool, PoolSelector
from .readout import ReadOut


class BrainGNN(torch.nn.Module):
    def __init__(
            self, dim_in: int, num_class: int, num_cluster: int,
            pool_ratio: float, drop_ratio: float,
            dim_conv1: int, dim_conv2: int, dim_conv3: int, dim_mlp: int
    ):
        super().__init__()

        self.num_cluster = num_cluster
        self.num_roi = dim_in
        self.pool_ratio = pool_ratio
        self.drop_ratio = drop_ratio

        self.dim_in = dim_in
        self.dim1 = dim_conv1
        self.dim2 = dim_conv2
        self.dim3 = dim_conv3
        self.dim4 = dim_mlp
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
            nonlinearity=nn.Sigmoid()
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
            nonlinearity=nn.Sigmoid()
        )
        self.conv3 = GConv(
            in_channel=self.dim2,
            out_channel=self.dim3,
            num_cluster=self.num_cluster,
            num_roi=self.num_roi
        )
        self.pool3 = TopKPool(
            in_channels=self.dim3,
            ratio=self.pool_ratio,
            multiplier=1,
            nonlinearity=nn.Sigmoid()
        )

        self.readout = ReadOut()

        self.mlp1 = nn.Sequential(
            nn.Linear((self.dim1 + self.dim2 + self.dim3) * 2, self.dim3, bias=True),
            nn.PReLU(),
            nn.BatchNorm1d(self.dim3),
            nn.Dropout(p=self.drop_ratio)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(self.dim3, self.dim4, bias=True),
            nn.PReLU(),
            nn.BatchNorm1d(self.dim4),
            nn.Dropout(p=self.drop_ratio)
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(self.dim4, self.dim_out),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, data):
        h1 = self.conv1(data.x, data.edge_index, data.edge_attr, data.pos)
        res1: PoolSelector = self.pool1(h1, data.edge_index, data.edge_attr, data.batch, data.pos)

        # res1.edge_index, res1.edge_attr = BrainGNN.augment_adj_mat(
        #     res1.edge_index, res1.edge_attr.squeeze(), res1.x.size(0)
        # )

        h2 = self.conv2(res1.x, res1.edge_index, res1.edge_attr, res1.pos)
        res2: PoolSelector = self.pool2(h2, res1.edge_index, res1.edge_attr, res1.batch, res1.pos)

        h3 = self.conv3(res2.x, res2.edge_index, res2.edge_attr, res2.pos)
        res3: PoolSelector = self.pool3(h3, res2.edge_index, res2.edge_attr, res2.batch, res2.pos)

        x = self.readout(res1.x, res1.batch, res2.x, res2.batch, res3.x, res3.batch)

        x1 = self.mlp1(x)
        x2 = self.mlp2(x1)
        x_out = self.mlp3(x2)

        score1 = res1.score_norm.view(x_out.size(0), -1)
        score2 = res2.score_norm.view(x_out.size(0), -1)
        score3 = res3.score_norm.view(x_out.size(0), -1)
        score1_uni = res1.score.view(x_out.size(0), -1)

        # x_out = torch.nn.functional.softmax(x3, dim=-1)

        return x_out, \
            [self.pool1.weight, self.pool2.weight, self.pool3.weight], \
            [score1, score2, score3], \
            score1_uni

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
