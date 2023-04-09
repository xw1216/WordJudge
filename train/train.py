import logging
import pathlib

import torch
import wandb

from omegaconf import DictConfig, OmegaConf

import util
import dataset
import model


class Train:
    def __init__(self, cfg: DictConfig, log: logging.Logger):
        self.cfg: DictConfig = cfg
        self.logger: logging.Logger = log
        self.device = self.define_device()

        self.loader = None
        self.model = None
        self.optim = None
        self.sched = None

    def define_device(self):
        return torch.device(
            "cuda" if self.cfg.cuda and torch.cuda.is_available() else "cpu"
        )


    def start(self):
        self.logger.info('Training Starting...')
        self.logger.info(OmegaConf.to_yaml(self.cfg))

