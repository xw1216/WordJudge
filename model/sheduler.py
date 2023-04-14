from torch.optim import Optimizer
from torch.optim import lr_scheduler as sched


def build_scheduler(
        name: str, optim: Optimizer, step_size: int, gamma: float = 0.5,
        lr: float = None, epoch: int = None
):
    assert name in ["Step", "Exp", "CosAnneal", "Reduce", 'OneCycle']

    if name == 'Step':
        return sched.StepLR(optim, step_size, gamma)
    elif name == 'Exp':
        return sched.ExponentialLR(optim, gamma)
    elif name == 'CosAnneal':
        return sched.CosineAnnealingLR(optim, T_max=step_size)
    elif name == 'Reduce':
        return sched.ReduceLROnPlateau(optim, mode='min', factor=gamma)
    else:
        return sched.OneCycleLR(optim, max_lr=lr, epochs=epoch, steps_per_epoch=1)


