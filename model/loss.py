import torch

from torch import Tensor


def top_k_loss(score: Tensor, pool_ratio: float, eps: float = 1e-10) -> Tensor:
    if pool_ratio > 0.5:
        pool_ratio = 1 - pool_ratio

    score = score.sort(dim=1, descending=True).values
    num_select = int(score.size(1) * pool_ratio)

    res = -(torch.log(score[:, :num_select] + eps).mean() +
            torch.log(1 - score[:, -num_select:] + eps).mean())
    return res


def consist_loss(score: Tensor, labels: Tensor, n_class: int, device: torch.device) -> Tensor:
    loss = Tensor([0])
    for c in range(n_class):
        sub_score = score[labels == c]
        sub_score = torch.sigmoid(sub_score)

        m = sub_score.shape[0]
        w_mat = torch.ones((m, m))
        d_mat = torch.eye(m) * m
        l_mat = d_mat - w_mat
        l_mat = l_mat.to(device)

        res = torch.trace(torch.transpose(sub_score, 0, 1) @ l_mat @ sub_score) / (m ** 2)
        loss += res

    return loss * 2


def unit_loss(weight: Tensor) -> float:
    return (torch.norm(weight, p=2) - 1) ** 2


def cross_entropy_loss(output: Tensor, labels: Tensor) -> Tensor:
    return torch.nn.functional.nll_loss(output, labels)
