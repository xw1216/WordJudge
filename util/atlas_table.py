from pathlib import Path

import numpy as np
import pandas as pd
import wandb

from matplotlib import colors
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
    roi_num = len(atlas_table)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.title('Pool Score Distribution')
    plt.xticks(
        np.arange(len(atlas_table)), labels=atlas_table,
        rotation=60, rotation_mode='anchor', ha='right', fontsize=6
    )
    plt.plot(
        np.arange(roi_num), score.numpy(),
        c='lightcoral', marker='v',
        markeredgecolor='black',
        linewidth=1, markersize=5
    )
    plt.savefig(str(file_path))
    wandb.log({
        "test/fold": 0,
        "test/" + name: wandb.Image(plt.gcf()),
    })
    plt.show()
    plt.close()


def divide_index_by_step(length: int, step: int):
    perm = [0]
    rows = length // step
    for i in range(rows):
        perm.append((i + 1) * step)
    if length % step != 0:
        rows += 1
        perm.append(length)
    return rows, perm


def build_cluster_label(length: int):
    cluster_labels = []
    for i in range(length):
        cluster_labels.append('C' + str(i))
    return cluster_labels


def draw_community_heatmap(atlas_table: list, community_factor: Tensor, save_path: Path):
    cluster_num = community_factor.shape[0]
    atlas_roi_num = len(atlas_table)

    name = 'community_factor'
    rows, perm = divide_index_by_step(atlas_roi_num, 12)
    cluster_labels = build_cluster_label(cluster_num)

    factor_np = community_factor.numpy()
    v_min = np.min(factor_np)
    v_max = np.max(factor_np)
    norm = colors.Normalize(vmin=v_min, vmax=v_max)

    fig = plt.figure(figsize=(8, 0.9 * cluster_num * rows), dpi=300)
    plt.axis('off')
    plt.title('Membership score of ROI to community', fontsize=16)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

    imgs = []
    axes = []

    for i in range(rows):
        ax = fig.add_subplot(rows, 1, i + 1)
        axes.append(ax)
        start_index = perm[i]
        end_index = perm[i + 1]

        plt.xticks(
            np.arange(end_index - start_index), labels=atlas_table[start_index:end_index],
            rotation=45, rotation_mode='anchor', ha='right', fontsize=8
        )
        plt.yticks(np.arange(2), labels=cluster_labels, fontsize=8)

        im = plt.imshow(factor_np[:, start_index:end_index], cmap='coolwarm', norm=norm)
        imgs.append(im)

        for j in range(cluster_num):
            for k in range(start_index, end_index):
                ax.text(k - start_index, j, round(factor_np[j][k], 2),
                        ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(imgs[0], ax=axes, shrink=0.7)
    plt.savefig(str(Path(save_path, name + '.png')))
    wandb.log({
        "test/fold": 0,
        "test/" + name: wandb.Image(plt.gcf()),
    })
    plt.show()
    plt.close()


def draw_atlas_interpret(
        atlas_table: list, community_factor: Tensor,
        score_true: Tensor, score_false: Tensor,
        save_path: Path
):
    save_path = Path(save_path, 'img')
    save_path.mkdir(parents=True, exist_ok=True)

    draw_community_heatmap(atlas_table, community_factor, save_path)
    draw_pool_score(atlas_table, score_true, save_path, 'score1_true')
    draw_pool_score(atlas_table, score_false, save_path, 'score1_false')
