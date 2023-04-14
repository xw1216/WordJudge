import logging

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
        util.mat2ny(self.cfg)
        self.logger.info('Data preparation complete')

    def build_kfold_graph_loader(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        self.logger.info('Building in memory dataset')
        conn = dataset.ConnDataset(self.cfg)

        loader = dataset.KFoldGroup(
            conn,
            log=self.logger,
            fold=self.cfg.train.fold,
            seed=self.cfg.seed,
            shuffle=True
        )

        batch_size = self.cfg.train.batch_size
        k_cnt = 1

        self.logger.info('Splitting data into K Fold')
        for train_data, test_data, valid_data in loader.split():
            self.logger.info(f'Fold {k_cnt}/{loader.fold} split complete')
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
            yield train_loader, test_loader, valid_loader

    def build_model(self):
        self.model = model.BrainGNN(
            dim_in=self.cfg.dataset.n_roi,
            num_class=self.cfg.dataset.n_class,
            num_cluster=self.cfg.train.n_cluster,
            pool_ratio=self.cfg.train.pool_ratio,
            drop_ratio=self.cfg.train.drop_ratio
        ).to(self.device)
        self.logger.info(self.model)

        self.optim = model.build_optimizer(
            name=self.cfg.train.optim,
            param=self.model.parameters(),
            lr=self.cfg.model.lr,
            weight_decay=self.cfg.model.weight_decay
        )
        self.logger.info(self.optim)

        self.sched = model.build_scheduler(
            name=self.cfg.train.sched,
            optim=self.optim,
            step_size=self.cfg.model.step_size,
            gamma=self.cfg.model.gamma,
            epoch=self.cfg.train.epochs
        )
        self.logger.info(self.sched)

    def loss_batch(
            self, output: torch.Tensor, labels: torch.Tensor,
            weight1: torch.Tensor, weight2: torch.Tensor,
            score1: torch.Tensor, score2: torch.Tensor
    ) -> torch.Tensor:
        loss_ce = model.cross_entropy_loss(output, labels)

        loss_unit1 = model.unit_loss(weight1)
        loss_unit2 = model.unit_loss(weight2)

        loss_top1 = model.top_k_loss(
            score1, self.cfg.model.pool_ratio, self.cfg.model.eps
        )
        loss_top2 = model.top_k_loss(
            score2, self.cfg.model.pool_ratio, self.cfg.model.eps
        )

        loss_consist = model.consist_loss(
            score1, labels, self.cfg.dataset.n_class, self.device
        )

        loss = loss_ce + (
                loss_unit1 + loss_unit2) + (
                loss_top1 + loss_top2) * self.cfg.model.lamb_top + (
                loss_consist) * self.cfg.model.lamb_consist
        return loss

    def run(self):
        self.logger.info('Training Start...')
        self.logger.info('Printing program configurations')
        self.logger.info(OmegaConf.to_yaml(self.cfg))

        self.logger.info('Converting raw data to npy files')
        self.convert_data()

        self.logger.info('Printing model construction')
        self.build_model()

        train_loader, valid_loader, test_loader = self.build_kfold_graph_loader()
