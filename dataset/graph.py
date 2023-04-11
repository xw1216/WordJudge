import logging
import torch
import numpy as np
import networkx as nx
import scipy.sparse as sci_spar

import torch_geometric as pyg
import torch_sparse as pys

from torch_geometric.data import Data


class CooTensor:
    def __init__(self, index: np.ndarray, value: np.ndarray):
        self.index = torch.from_numpy(index).long()
        self.value = torch.from_numpy(value)

    def to_tuple(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.index, self.value


class GraphBuilder:
    def __init__(self, fill_val: int, log: logging.Logger):
        self.num_graph = 0
        self.num_node = 0
        self.log = log
        self.fill_val = fill_val

    def check_consist(self, nodes: np.ndarray, edges: np.ndarray, labels: np.ndarray):
        if not (
                nodes.shape[0] == edges.shape[0] == len(labels)
                and nodes.shape[1] == edges.shape[1]
                and nodes.shape[2] == edges.shape[2]
        ):
            self.log.error(
                f'Inconsistent shape between '
                f'nodes: {nodes.shape}, '
                f'edges: {edges.shape}, '
                f'labels: {labels.shape}'
            )
            raise OSError

    def single_build(
            self, edges: np.ndarray
    ) -> CooTensor:
        edge_coo_orig = GraphBuilder.to_coo_tensor(edges)
        edge_coo_no_loop = GraphBuilder.remove_self_loops(edge_coo_orig)
        edge_sparse_tensor = GraphBuilder.coalesce(edge_coo_no_loop, self.num_node)

        return edge_sparse_tensor

    def do(self, nodes: np.ndarray, edges: np.ndarray, labels: np.ndarray) -> tuple[Data, dict]:
        self.check_consist(nodes, edges, labels)

        self.num_graph = len(labels)
        self.num_node = nodes.shape[1]

        # create 3D node tensor and concat 2D row-wise
        # shape = [num_node * num_sample, num_node], type = float32
        node_tensor = torch.from_numpy(nodes).float()
        node_tensor = torch.cat(torch.unbind(node_tensor), dim=0)

        # turn label tensor into column vector for graph classification
        # shape = [num_sample, 1], type = int64
        labels_tensor = torch.from_numpy(labels.reshape((-1, 1))).long()

        # diagonal matrix v stack num_sample times
        # shape = [num_node * num_sample, num_node], type = float32
        pseudo_tensor = torch.tile(torch.eye(self.num_node), (self.num_graph, 1))

        # cat remaining sparse element index & weight horizontally
        # shape = [2, num_remain_node], type = int64
        edge_index_tensor = torch.Tensor()
        # shape = [num_remain_node, 1], type = float32
        edge_tensor = torch.Tensor()

        # batch indicator row vector
        # shape = [num_node * num_sample], type = int64
        batch_tensor = torch.Tensor()

        for i in range(self.num_graph):
            self.log.info(f'Building subject graph for {i}/{self.num_graph}')
            edge_sparse_tensor = self.single_build(edges[i])

            torch.cat((edge_index_tensor, edge_sparse_tensor.index), dim=1)
            torch.cat((edge_tensor, edge_sparse_tensor.value), dim=0)

            torch.cat((batch_tensor, torch.tile(torch.Tensor(i).long(), self.num_node)))

        edge_tensor = torch.transpose(edge_tensor, 0, 1)

        [node_slice, edge_slice, edge_index_tensor] = \
            self.align_slice(edge_index_tensor, batch_tensor)

        data = Data(
            x=node_tensor,
            edge_index=edge_index_tensor,
            edge_attr=edge_tensor,
            y=labels_tensor,
            pos=pseudo_tensor
        )

        slices = {
            'x': node_slice,
            'edge_index': edge_slice,
            'edge_attr': edge_slice,
            'y': torch.arange(0, self.num_graph + 1, dtype=torch.long),
            'pos': node_slice,
        }

        return data, slices

    @classmethod
    def to_coo_tensor(cls, edges: np.ndarray) -> CooTensor:
        graph: nx.DiGraph = nx.convert_matrix.from_numpy_matrix(edges, create_using=nx.DiGraph)

        # 0 weight elements are removed
        sparse_arr: sci_spar.csr_array = nx.to_scipy_sparse_array(graph)
        adj_coo_arr: sci_spar.coo_array = sparse_arr.tocoo()
        coo_len = len(adj_coo_arr)

        # ndarray [2, dim^2] element index of sparse matrix and 1: row index 2: col index
        edge_index = np.stack([adj_coo_arr.row, adj_coo_arr.col])

        # ndarray [1, dim^2] edge weight of sparse matrix - dim^2 edges in total
        edge_feature = np.zeros(coo_len)
        for i in range(coo_len):
            edge_feature[i] = edges[adj_coo_arr.row[i], adj_coo_arr.col[i]]

        return CooTensor(edge_index, edge_feature)

    @staticmethod
    def remove_self_loops(coo: CooTensor) -> CooTensor:
        # prevent self looping
        # normally it will remove edges on the diag of matrix
        res = pyg.utils.remove_self_loops(*coo.to_tuple())
        return CooTensor(*res)

    @staticmethod
    def coalesce(coo: CooTensor, n: int) -> CooTensor:
        res = pys.coalesce(coo.index, coo.value, n, n)
        return CooTensor(*res)

    @classmethod
    def align_slice(cls, edge_index_tensor: torch.Tensor, batch_tensor: torch.Tensor) -> list[torch.Tensor]:
        node_cnt = torch.bincount(batch_tensor)
        node_slice = torch.cumsum(node_cnt, dim=0)
        node_slice = torch.cat((torch.Tensor(0).long(), node_slice))

        row_tensor = edge_index_tensor[0, :]
        edge_cnt = torch.bincount(batch_tensor[row_tensor])
        edge_slice = torch.cumsum(edge_cnt, dim=0)
        edge_slice = torch.cat((torch.Tensor([0]).long(), node_slice))

        # use broadcast to align edge index for every graph back in range(0, num_node)
        edge_start_index_tensor = node_slice[batch_tensor[row_tensor]]
        edge_index_tensor -= torch.unsqueeze(edge_start_index_tensor, 0)

        return [node_slice, edge_slice, edge_index_tensor]
