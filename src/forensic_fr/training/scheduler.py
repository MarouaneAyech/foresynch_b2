import math


def cosine_lr(epoch: int, warmup_epochs: int, total_epochs: int, base_lr: float) -> float:
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))
