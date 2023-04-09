import os
import re
import pathlib

import numpy as np
from scipy import io as sci_io
from omegaconf import DictConfig


def load_fill_nan(mat_path: str, cfg: DictConfig):
    z_mat = sci_io.loadmat(mat_path)['Z']
    np_arr = np.array(z_mat)
    np.nan_to_num(np_arr, False, cfg.self_effect_val)
    return np_arr


def convert_mat_in_path(path: pathlib.Path, cfg: DictConfig):
    file_list = os.listdir(path)
    file_list.sort()

    pattern = re.compile(
        r'resultsROI_Subject(?P<sub_id>\d+)_Condition(?P<con_id>\d+)\.mat'
    )

    for filename in file_list:
        if pattern.match(filename):
            id_tuple = pattern.match(filename).groups()
            mat_path = str(pathlib.Path(path, filename))

            np_arr = load_fill_nan(mat_path, cfg)
            npy_path = str(pathlib.Path(path, f'sub{id_tuple[0]}_con{id_tuple[1]}.npy'))
            np.save(npy_path, np_arr)

            os.remove(mat_path)


def mat2ny(cfg: DictConfig):
    node_path = pathlib.Path(cfg.train.data_path, cfg.train.node_type)
    edge_path = pathlib.Path(cfg.train.data_path, cfg.train.edge_type)

    convert_mat_in_path(node_path, cfg)
    convert_mat_in_path(edge_path, cfg)
