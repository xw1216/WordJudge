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

        self.val_node_fill = cfg.dataset.node_mat_fill_val
        # self.val_edge_offset = cfg.dataset.edge_mat_offset
        self.eps: float = 1e-10
        self.logger: logging.Logger = logging.getLogger(cfg.log.log_name)
        self.override_data(cfg.dataset.override_data)

        # will invoke process if processed file do not exist
        super().__init__(root=str(self.path_save))
        self.data, self.slices = torch.load(self.processed_paths[0])

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

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self.name_syn_file

    def download(self):
        pass

    def process(self):
        data_files = [
            f'{self.name_node_feat}.npy',
            f'{self.name_edge_feat}.npy',
        ]

        nodes: np.ndarray = np.load(str(Path(self.path_data, data_files[0])))
        edges: np.ndarray = np.load(str(Path(self.path_data, data_files[1])))
        labels: np.ndarray = self.load_label()

        nodes = self.transform_node_matrix(nodes)
        edges = self.transform_edge_matrix(edges)

        builder = GraphBuilder(self.logger)
        self.data, self.slices = builder.do(nodes, edges, labels)

        torch.save((self.data, self.slices), self.processed_paths[0])


class KFoldGroup:
    def __init__(self, dataset: ConnDataset, log: logging.Logger, fold: int, seed: int, shuffle: bool = True):
        self.dataset = dataset
        self.log = log
        self.fold = fold
        self.seed = seed
        self.shuffle = shuffle
        self.n_sample = len(dataset)

        self.__check_index()

    def __check_index(self):
        if self.fold <= 2 \
                or self.fold > self.n_sample \
                or int(self.n_sample % self.fold) != 0:
            self.log.error(f'Incorrect k fold config: {self.n_sample} divided by {self.fold}')
            raise RuntimeError

    def __create_mask(self, ids: np.ndarray, start: int, stop: int):
        mask = np.zeros(self.n_sample, dtype=bool)
        mask[ids[start:stop]] = True
        return mask

    def __group_test_mask(self):
        ids = np.arange(0, self.n_sample)

        if self.shuffle:
            np.random.RandomState(self.seed).shuffle(ids)

        fold_sizes = np.full(self.fold, self.n_sample // self.fold, dtype=int)
        fold_sizes[: self.n_sample % self.fold] += 1

        test_mask = self.__create_mask(ids, self.n_sample - fold_sizes[-1], self.n_sample)
        # except last group as test set
        fold_sizes = np.delete(fold_sizes, -1)

        cur = 0
        for step in fold_sizes:
            start, stop = cur, cur + step
            valid_mask = self.__create_mask(ids, start, stop)
            train_mask = np.logical_not(np.logical_or(valid_mask, test_mask))

            yield train_mask, valid_mask, test_mask
            cur = stop

    def split(self):
        index = np.arange(0, self.n_sample)
        for train_mask, valid_mask, test_mask in self.__group_test_mask():
            train_index: np.ndarray = index[train_mask]
            valid_index = index[valid_mask]
            test_index = index[test_mask]

            self.log.info(f'Train set index array: {train_index.tolist()}')
            self.log.info(f'Validate set index array: {valid_index.tolist()}')
            self.log.info(f'Test set index array: {test_index.tolist()}')

            yield self.dataset[train_index], \
                self.dataset[valid_index], \
                self.dataset[test_index]
