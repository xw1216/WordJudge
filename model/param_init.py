import math
import torch


def uniform(fan_in: int, x: torch.Tensor | None):
    if x is None:
        return
    bound = 1.0 / math.sqrt(fan_in)
    x.uniform_(from_=-bound, to=bound)
