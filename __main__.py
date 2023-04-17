from typing import Optional

import hydra
import wandb


from omegaconf import OmegaConf, DictConfig

import util
from train import Train


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    util.create_project_dir(cfg)
    util.append_logger(cfg)

    program = Train(cfg)
    program.run()
    program.test()


if __name__ == '__main__':
    main()
