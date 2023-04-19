import math
import torch


def uniform(fan_in: int, x: torch.Tensor | None):
    if x is None:
        return
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        x.uniform_(-bound, bound)
