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
        self.path_data: Path = Path(cfg.train.data_path)
        self.path_save: Path = Path(cfg.train.save_path)

        self.name_node_feat: str = cfg.dataset.node_type
        self.name_edge_feat: str = cfg.dataset.edge_type
        self.name_label_type: str = cfg.dataset.label_type

        self.label_file_name: str = cfg.dataset.label_file_name
        self.syn_file_name: str = cfg.train.syn_file_name

        self.fill_val = cfg.dataset.fill_val
        self.offset = cfg.dataset.ppi_offset
        self.log: logging.Logger = cfg.logger
        self.override_data(cfg.override_data)

        # will invoke process if processed file do not exist
        super().__init__(root=str(self.path_save))
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data.x = self.transform_node_matrix(self.data.x)

    def override_data(self, can_override: bool) -> None:
        path_syn = Path(self.path_save, 'processed', self.syn_file_name)
        if can_override and path_syn.exists():
            self.log.warning(f'Removing existed synthesized graph data {path_syn}')
            os.remove(path_syn)

    def load_label(self) -> np.ndarray:
        series: pd.Series = \
            pd.read_csv(
                str(Path(self.path_data, self.label_file_name))
            ).loc[:, self.name_label_type]
        return np.array(series.tolist(), dtype=torch.int)

    def transform_edge_matrix(self, edges: np.ndarray):
        # TODO dynamic change the val offset
        if self.name_edge_feat == 'gPPI':
            edges += self.offset

    def transform_node_matrix(self, nodes: torch.Tensor):
        t = torch.where(torch.isnan(nodes), torch.full_like(nodes, self.fill_val), nodes)
        t = torch.where(torch.isinf(t), torch.full_like(t, self.fill_val), t)
        return t

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self.syn_file_name

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

        self.transform_edge_matrix(edges)

        builder = GraphBuilder(self.fill_val, self.log)
        self.data, self.slices = builder.do(nodes, edges, labels)

        torch.save((self.data, self.slices), self.processed_paths[0])


class KFoldGroup:
    def __init__(self, dataset: ConnDataset, log: logging.Logger, fold: int, seed: int, shuffle: bool = True):
        self.n_sample = len(dataset)
        self.fold = fold
        self.shuffle = shuffle
        self.seed = seed
        self.dataset = dataset
        self.log = log

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

        cur = 0
        for i in fold_sizes:
            # except last group as validation set
            if i >= len(fold_sizes - 1):
                break

            start, stop = cur, cur + fold_sizes[i]
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
