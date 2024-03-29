import os
import re
import logging
from pathlib import Path

import numpy as np
from scipy import io as sio
from omegaconf import DictConfig

from .timer import Timer


def load_mat(mat_path: str) -> np.ndarray:
    z_mat = sio.loadmat(mat_path)['Z']
    return z_mat


def convert_mat_in_path(conn_path: Path, log: logging.Logger):
    file_list = os.listdir(conn_path)
    file_list.sort()

    pattern = re.compile(
        r'resultsROI_Subject(?P<sub_id>\d+)_Condition(?P<con_id>\d+)\.mat'
    )

    log.info(f'Diving into {conn_path}')

    for filename in file_list:
        if pattern.match(filename):
            id_tuple = pattern.match(filename).groups()
            mat_path = str(Path(conn_path, filename))

            np_arr = load_mat(mat_path)
            npy_filename = f'sub{id_tuple[0]}_con{id_tuple[1]}.npy'
            npy_path = str(Path(conn_path, npy_filename))

            log.info(f'Converting {filename} into {npy_filename}')
            np.save(npy_path, np_arr)
            os.remove(mat_path)


def load_split_npy(path: Path, file_list: list[str], num_roi: int) -> np.ndarray:
    pattern = re.compile(r'.*\.npy')
    data = None
    for file in file_list:
        if pattern.match(file):
            sub_data = np.load(str(Path(path, file)))
            if data is not None:
                data = np.concatenate((data, sub_data.reshape((1, num_roi, num_roi))))
            else:
                data = sub_data.reshape((1, num_roi, num_roi))
    return data


def concat_npy(data_path: Path, conn_type: str,
               sample_num: int, roi_num: int,
               override: bool, log: logging.Logger):
    npy_path = Path(data_path, f'{conn_type}.npy')
    if os.path.exists(npy_path):
        if override:
            log.warning(f'Removing existed concat {conn_type} data')
            os.remove(npy_path)
        else:
            log.info(f'File {npy_path} detected, skip concat')
            return
    conn_path = Path(data_path, conn_type)
    file_list = os.listdir(conn_path)
    file_list.sort()

    if len(file_list) != sample_num:
        log.error(f'Unmatch sample number detected.')
        raise RuntimeError

    data = load_split_npy(conn_path, file_list, roi_num)
    np.save(str(npy_path), data)
    log.info(f'Saving synthesized {conn_type} data to {npy_path}')


def mat2ny(cfg: DictConfig):
    node_type = cfg.dataset.node_type
    edge_type = cfg.dataset.edge_type

    data_path = Path(cfg.dataset.data_path)
    node_path = Path(data_path, node_type)
    edge_path = Path(data_path, edge_type)

    timer = Timer()
    log: logging.Logger = logging.getLogger(cfg.log.log_name)
    sample_num = cfg.dataset.n_sample
    roi_num = cfg.dataset.n_roi
    can_override = cfg.dataset.override_data

    timer.start()
    convert_mat_in_path(node_path, log)
    convert_mat_in_path(edge_path, log)

    concat_npy(data_path, node_type,
               sample_num, roi_num, can_override, log)
    concat_npy(data_path, edge_type,
               sample_num, roi_num, can_override, log)
    timer.end()

    log.info("Transform from .mat to .npy complete")
    log.info(f"Consuming {timer.last()} s")
