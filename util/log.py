import logging
import pathlib
from datetime import datetime

from omegaconf import DictConfig


def init_formatter(cfg: DictConfig) -> logging.Formatter:
    return logging.Formatter(
        fmt=cfg.log.log_format,
        datefmt=cfg.log.time_format,
    )


def clear_handler(log: logging.Logger):
    for handler in log.handlers:
        handler.close()
    log.handlers.clear()


def init_logger(cfg: DictConfig, path: pathlib.Path) -> logging.Logger:
    log = logging.getLogger('WordJudge')
    log.setLevel(logging.INFO)
    clear_handler(log)

    formatter = init_formatter(cfg)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)

    file_handler = logging.FileHandler(
        str(pathlib.Path(path, cfg.log.log_name)),
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    return log


def logger(cfg: DictConfig) -> logging.Logger:
    time_str = datetime.now().strftime(cfg.log.dir_format)
    path_log = pathlib.Path(cfg.train.log_path, time_str)
    path_log.mkdir(exist_ok=True, parents=True)

    log = init_logger(cfg, path_log)
    return log
