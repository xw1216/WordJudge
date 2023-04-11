import logging
import pathlib

from omegaconf import DictConfig


def create_project_dir(cfg: DictConfig) -> None:
    log = cfg.logger
    pack_list = ['log', 'train']
    dir_list = [
        ['log_path', 'wandb_path'],
        ['save_path', 'data_path']
    ]

    for i in range(len(pack_list)):
        for j in range(len(dir_list[i])):
            path = pathlib.Path(
                cfg[pack_list[i]][dir_list[i][j]]
            )
            path.mkdir(exist_ok=True)
            log.info(f'Checking project folder {str(path)}')
