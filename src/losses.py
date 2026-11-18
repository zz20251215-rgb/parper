from torch.nn import functional as F


def loss_components(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none').mean((1, 2, 3))
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * target).sum((1, 2, 3)) + 1) / (
        probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    positive = target.flatten(1).any(1)
    return bce, dice, positive


def combine_loss(bce, dice, positive):
    return bce.mean() + (dice * positive).sum() / positive.sum().clamp_min(1)


def loss_fn(logits, target):
    return combine_loss(*loss_components(logits, target))
