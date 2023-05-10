import os
import logging
from pathlib import Path
from typing import Union, List, Tuple

import torch
import pandas as pd
import numpy as np

from omegaconf import DictConfig
from torch_geometric.data import InMemoryDataset

from .graph import GraphBuilder


class ConnDataset(InMemoryDataset):
    def __init__(self, cfg: DictConfig) -> None:
        self.path_data: Path = Path(cfg.dataset.data_path)
        self.path_save: Path = Path(cfg.train.save_path)

        self.name_node_feat: str = cfg.dataset.node_type
        self.name_edge_feat: str = cfg.dataset.edge_type
        self.name_label_type: str = cfg.dataset.label_type

        self.name_label_file: str = cfg.dataset.label_file_name
        self.name_syn_file: str = cfg.train.syn_file_name

        self.aug_noise: bool = cfg.dataset.aug_noise
        self.aug_mixup: bool = cfg.dataset.aug_mixup
        self.aug_noise_bound: int = cfg.dataset.noise_bound
        self.aug_mixup_alpha: int = cfg.dataset.mixup_alpha

        self.aug_noise_mul: int = cfg.dataset.sample_aug_noise_mul
        self.aug_mixup_mul: int = cfg.dataset.sample_aug_mixup_mul
        self.n_sample_orig: int = cfg.dataset.n_sample
        self.n_sample_noise: int = self.aug_noise_mul * self.n_sample_orig if self.aug_noise else 0
        self.n_sample_mixup: int = self.aug_mixup_mul * self.n_sample_orig if self.aug_mixup else 0
        self.n_sample: int = self.n_sample_orig + self.n_sample_noise + self.n_sample_mixup

        self.val_node_fill = cfg.dataset.node_mat_fill_val
        self.eps: float = 1e-10
        self.logger: logging.Logger = logging.getLogger(cfg.log.log_name)
        self.override_data(cfg.dataset.override_data)

        # will invoke process if processed file do not exist
        super().__init__(root=str(self.path_save))
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self.name_syn_file

    def download(self):
        pass

    def override_data(self, can_override: bool) -> None:
        path_syn = Path(self.path_save, 'processed', self.name_syn_file)
        if can_override and path_syn.exists():
            self.logger.warning(f'Removing existed synthesized graph data {path_syn}')
            os.remove(path_syn)

    def load_label(self) -> np.ndarray:
        series: pd.Series = \
            pd.read_csv(
                str(Path(self.path_data, self.name_label_file))
            ).loc[:, self.name_label_type]
        return np.array(series.tolist(), dtype=np.int)

    def min_max_norm(self, x: np.ndarray):
        temp = np.nan_to_num(x, copy=True, nan=0)
        max_val = temp.max(initial=None)
        min_val = temp.min(initial=None)

        return ((x - min_val) / (max_val - min_val)) + self.eps

    def transform_edge_matrix(self, edges: np.ndarray):
        # 1. shift all value to positive according to min value
        # edges_temp = edges.copy()
        # edges_temp = np.nan_to_num(edges_temp, False, 1e10)
        # offset = np.abs(edges_temp.min(initial=None)) + 1
        # self.val_edge_offset = offset
        # return edges + offset

        # 2. plus fixed offset
        # return edges + offset

        # 3. original value
        # return edges

        # 4. absolute value
        # return np.abs(edges)

        # 5. Resealing to [-20, 20]
        return self.min_max_norm(edges) * 40 - 20

    def transform_node_matrix(self, nodes: np.ndarray):
        temp = np.nan_to_num(nodes, copy=True, nan=self.val_node_fill)
        return self.min_max_norm(temp) * 40 - 20

    def data_augment(self, nodes: np.ndarray, edges: np.ndarray, labels: np.ndarray):
        nodes_out, edges_out, labels_out = nodes.copy(), edges.copy(), labels.copy()
        if self.aug_noise:
            for i in range(self.aug_noise_mul):
                nodes_new = np.triu(
                    nodes.copy() + np.random.uniform(-self.aug_noise_bound, self.aug_noise_bound, nodes.shape)
                )
                nodes_new = nodes_new + np.transpose(nodes_new, (0, 2, 1))

                edges_new = np.triu(
                    edges.copy() + np.random.uniform(-self.aug_noise_bound, self.aug_noise_bound, nodes.shape)
                )
                edges_new += np.transpose(edges_new, (0, 2, 1))
                labels_new = labels.copy()

                nodes_out = np.concatenate((nodes_out, nodes_new))
                edges_out = np.concatenate((edges_out, edges_new))
                labels_out = np.concatenate((labels_out, labels_new))

        # if self.aug_mixup:
        #     num_sample = labels.shape[0]
        #     for i in range(self.aug_mixup_mul):
        #         mixup_lamb = np.random.beta(self.aug_mixup_alpha, self.aug_mixup_alpha)
        #         perm = np.random.permutation(num_sample)
        #
        #         nodes_mix = nodes.copy()[perm]
        #         edges_mix = edges.copy()[perm]
        #         labels_mix = labels.copy()[perm]
        #
        #         nodes_mix = (1 - mixup_lamb) * nodes_mix + mixup_lamb * nodes
        #         edges_mix = (1 - mixup_lamb) * edges_mix + mixup_lamb * edges
        #         labels_mix = (1 - mixup_lamb) * labels_mix + mixup_lamb * labels
        #
        #         nodes_out = np.concatenate(nodes_out, nodes_mix)
        #         edges_out = np.concatenate(edges_out, edges_mix)
        #         labels_out = np.concatenate(labels_out, labels_mix)

        return nodes_out, edges_out, labels_out

    def process(self):
        data_files = [
            f'{self.name_node_feat}.npy',
            f'{self.name_edge_feat}.npy',
        ]

        nodes: np.ndarray = np.load(str(Path(self.path_data, data_files[0])))
        edges: np.ndarray = np.load(str(Path(self.path_data, data_files[1])))
        labels: np.ndarray = self.load_label()

        nodes, edges, labels = self.data_augment(nodes, edges, labels)

        nodes = self.transform_node_matrix(nodes)
        edges = self.transform_edge_matrix(edges)

        builder = GraphBuilder(self.logger)
        self.data, self.slices = builder.do(nodes, edges, labels)

        torch.save((self.data, self.slices), self.processed_paths[0])
