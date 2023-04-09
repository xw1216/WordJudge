import torch

from omegaconf import DictConfig

import util
import dataset
import model


class Train:
    def __init__(self, cfg: DictConfig):
        self.log = util.logger(cfg)

    def start(self):
        pass
