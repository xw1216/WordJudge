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
            dim_conv: list[int], dim_mlp: int
    ):
        super().__init__()

        self.num_cluster = num_cluster
        self.num_roi = dim_in
        self.pool_ratio = pool_ratio
        self.drop_ratio = drop_ratio

        self.dim_in = dim_in
        self.dim_conv = dim_conv
        self.dim_mlp = dim_mlp
        self.dim_out = num_class
        self.conv_len = len(dim_conv)

        self.conv: nn.ModuleList = nn.ModuleList()
        self.pool: nn.ModuleList = nn.ModuleList()

        for i in range(self.conv_len):
            conv_in_channel = self.dim_conv[i-1] if i > 0 else self.dim_in
            self.conv.append(
                GConv(
                    in_channel=conv_in_channel,
                    out_channel=self.dim_conv[i],
                    num_cluster=self.num_cluster,
                    num_roi=self.num_roi
                )
            )
            self.pool.append(
                TopKPool(
                    in_channels=self.dim_conv[i],
                    ratio=self.pool_ratio,
                    multiplier=1,
                    nonlinearity=nn.Sigmoid()
                )
            )

        self.readout = ReadOut(self.conv_len)

        mlp_sum_dim = 0
        for dim in self.dim_conv:
            mlp_sum_dim += dim

        self.mlp1 = nn.Sequential(
            nn.Linear(mlp_sum_dim * 2, self.dim_mlp, bias=True),
            nn.PReLU(),
            nn.BatchNorm1d(self.dim_mlp),
            nn.Dropout(p=self.drop_ratio)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(self.dim_mlp, self.dim_mlp, bias=True),
            nn.PReLU(),
            nn.BatchNorm1d(self.dim_mlp),
            nn.Dropout(p=self.drop_ratio)
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(self.dim_mlp, self.dim_out),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, data):
        res_list: list[PoolSelector] = []
        for i in range(self.conv_len):
            feed = res_list[i-1] if i > 0 else data
            graph = self.conv[i](feed.x, feed.edge_index, feed.edge_attr, feed.pos)
            res = self.pool[i](graph, feed.edge_index, feed.edge_attr, feed.batch, feed.pos)
            res_list.append(res)

        x = self.readout(res_list)

        x1 = self.mlp1(x)
        x2 = self.mlp2(x1)
        x_out = self.mlp3(x2)

        weight_list = []
        for i in range(self.conv_len):
            weight_list.append(self.pool[i].weight)

        score_norm_list = []
        for i in range(self.conv_len):
            score_norm_list.append(
                res_list[i].score_norm.view(x_out.size(0), -1)
            )
        score1_uni = res_list[0].score.view(x_out.size(0), -1)

        return x_out, weight_list, score_norm_list, score1_uni

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
