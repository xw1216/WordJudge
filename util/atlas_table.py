from pathlib import Path

import numpy as np
import pandas as pd
import wandb

import matplotlib.pyplot as plt
from torch import Tensor
from omegaconf import DictConfig


def read_atlas_table(cfg: DictConfig):
    csv_path = Path(cfg.dataset.data_path, 'Atlas_Labels', cfg.dataset.atlas_table_type + '.csv')
    if not csv_path.exists():
        raise RuntimeError('No such atlas table')

    series: pd.Series = \
        pd.read_csv(
            str(csv_path)
        ).loc[:, 'label']

    return series.tolist()


def draw_pool_score(atlas_table: list, score: Tensor, save_path: Path, name: str):
    file_path = Path(save_path, name + '.png')
    plt.figure(figsize=(8, 20))
    plt.yticks(np.arange(len(atlas_table)), labels=atlas_table)
    plt.barh(atlas_table, score.numpy())
    plt.savefig(str(file_path))
    wandb.log({
        "test/fold": 0,
        "test/" + name: wandb.Image(plt.gcf()),
    })
    plt.show()
    plt.close()


def draw_community_heatmap(atlas_table: list, community_factor: Tensor, save_path: Path):
    cluster_num = community_factor.shape[0]
    atlas_roi_num = len(atlas_table)
    name = 'community_factor'

    cluster_labels = []
    for i in range(cluster_num):
        cluster_labels.append('C' + str(i))

    factor_np = community_factor.numpy()
    plt.figure(figsize=(20, 8))
    plt.xticks(
        np.arange(atlas_roi_num), labels=atlas_table,
        rotation=90, rotation_mode='anchor', ha='right'
    )
    plt.yticks(np.arange(cluster_num), labels=cluster_labels)

    plt.imshow(factor_np, cmap='coolwarm')
    plt.title('Membership score of ROI to community')

    for i in range(cluster_num):
        for j in range(atlas_roi_num):
            text = plt.text(j, i, factor_np[i][j],
                            ha="center", va="center", color="black", fontweight="bold")
    plt.colorbar()
    plt.tight_layout()

    plt.savefig(str(Path(save_path, name + '.png')))
    wandb.log({
        "test/fold": 0,
        "test/" + name: wandb.Image(plt.gcf()),
    })
    plt.show()
    plt.close()


def draw_atlas_interpret(
        atlas_table: list, community_factor: Tensor,
        score1_true: Tensor, score1_false: Tensor,
        save_path: Path
):
    save_path = Path(save_path, 'img')
    save_path.mkdir(parents=True, exist_ok=True)

    draw_community_heatmap(atlas_table, community_factor, save_path)
    draw_pool_score(atlas_table, score1_true, save_path, 'score1_true')
    draw_pool_score(atlas_table, score1_false, save_path, 'score1_false')
