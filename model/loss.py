import torch

from torch import Tensor


class LossSelector:
    def __init__(self, conv_len: int):
        self.conv_len = conv_len
        self.loss_all: Tensor = torch.Tensor()
        self.loss_ce: Tensor = torch.Tensor()
        self.loss_consist: Tensor = torch.Tensor()

        self.loss_unit: list[Tensor] = []
        self.loss_top: list[Tensor] = []

        for i in range(self.conv_len):
            self.loss_unit.append(torch.Tensor())
            self.loss_top.append(torch.Tensor())

    def to_num(self):
        self.loss_all = self.loss_all.cpu().item()
        self.loss_ce = self.loss_ce.cpu().item()
        self.loss_consist = self.loss_consist.cpu().item()

        for i in range(self.conv_len):
            self.loss_unit[i] = self.loss_unit[i].cpu().item()
            self.loss_top[i] = self.loss_top[i].cpu().item()


def top_k_loss(score: Tensor, pool_ratio: float, eps: float = 1e-10) -> Tensor:
    # if pool_ratio > 0.5:
    #     pool_ratio = 1 - pool_ratio

    score = score.sort(dim=1, descending=True).values
    num_select = int(score.size(1) * pool_ratio)

    res = -(torch.log(score[:, :num_select] + eps).mean() +
            torch.log(1 - score[:, -num_select:] + eps).mean())
    return res


def consist_loss(score: Tensor, labels: Tensor, n_class: int, device: torch.device) -> Tensor:
    loss = torch.zeros(1).sum().to(device)
    for c in range(n_class):
        sub_score = score[labels == c]
        # sub_score = torch.sigmoid(sub_score)
        m = sub_score.shape[0]

        if m < 1:
            loss += 0
            continue

        w_mat = torch.ones((m, m))
        d_mat = torch.eye(m) * m
        l_mat = d_mat - w_mat
        l_mat = l_mat.to(device)

        res = torch.trace(torch.transpose(sub_score, 0, 1) @ l_mat @ sub_score) / (m ** 2)
        loss += res

    return loss


def unit_loss(weight: Tensor) -> Tensor:
    return (torch.norm(weight, p=2) - 1) ** 2


def cross_entropy_loss(output: Tensor, labels: Tensor) -> Tensor:
    return torch.nn.functional.nll_loss(output, labels)
