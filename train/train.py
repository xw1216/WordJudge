import logging
import pathlib

import torch

from torch_geometric.data import DataLoader
from omegaconf import DictConfig, OmegaConf

import util
import dataset
import model


class Train:
    def __init__(self, cfg: DictConfig):
        self.cfg: DictConfig = cfg
        self.logger: logging.Logger = cfg.logger
        self.device: torch.device = self.define_device()

        self.loader = None
        self.model = None
        self.optim = None
        self.sched = None

    def define_device(self):
        if self.cfg.cuda and torch.cuda.is_available():
            self.logger.info('Torch Running on Cuda')
            return torch.device("cuda")
        else:
            self.logger.info('Torch Running on CPU')
            return torch.device("cpu")

    def convert_data(self):
        self.logger.info('Convert raw data to npy files')
        util.mat2ny(self.cfg)
        self.logger.info('Data preparation complete')

    def build_kfold_graph_loader(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        conn = dataset.ConnDataset(self.cfg)

        loader = dataset.KFoldGroup(
            conn,
            fold=self.cfg.train.fold,
            seed=self.cfg.seed,
            shuffle=True
        )

        batch_size = self.cfg.train.batch_size

        for train_data, test_data, valid_data in loader.split():
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
            yield train_loader, test_loader, valid_loader

    def run(self):
        self.logger.info('Training Starting...')
        self.logger.info('Printing program configurations')
        self.logger.info(OmegaConf.to_yaml(self.cfg))

        self.convert_data()
        train_loader, test_loader, valid_loader = self.build_kfold_graph_loader()

        # TODO build network



