import logging
from datetime import datetime
from pathlib import Path

import wandb

from wandb.sdk.lib.runid import generate_id
from omegaconf import DictConfig, open_dict


def init_formatter(log_format: str, time_format: str) -> logging.Formatter:
    return logging.Formatter(fmt=log_format, datefmt=time_format)


def clear_handler(log: logging.Logger):
    for handler in log.handlers:
        handler.close()
    log.handlers.clear()


def init_logger(
        log_name: str, log_path: str, log_format: str, time_format: str
) -> logging.Logger:
    log = logging.getLogger(log_name)
    log.setLevel(logging.INFO)
    clear_handler(log)

    formatter = init_formatter(log_format, time_format)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    return log


def append_logger(cfg: DictConfig) -> logging.Logger:
    log_name = cfg.log.log_name
    log_file = cfg.log.log_file
    log_rel_path = cfg.log.log_path

    time_format = cfg.log.time_format
    dir_format = cfg.log.dir_format
    log_format = cfg.log.log_format

    dir_name = datetime.now().strftime(dir_format)
    log_path = Path(log_rel_path, dir_name)
    log_path.mkdir(exist_ok=True, parents=True)
    log_path = str(Path(log_path, log_file))

    log = init_logger(log_name, log_path, log_format, time_format)

    with open_dict(cfg):
        cfg.logger = log

    return log
