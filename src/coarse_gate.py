"""Low-resolution pixel gate for the first stage of the cascade."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class MobileNetV3SmallCoarseGate(nn.Module):
    """MobileNetV3-Small encoder with a 1/4-resolution crack heatmap head."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.encoder = nn.ModuleList([
            backbone.features[:2],   # 1/4, 16 channels
            backbone.features[2:5],  # 1/16, 40 channels
            backbone.features[5:9],  # 1/16, 48 channels
            backbone.features[9:12], # 1/32, 96 channels
        ])
        self.projections = nn.ModuleList([
            nn.Conv2d(16, 32, 1), nn.Conv2d(40, 32, 1),
            nn.Conv2d(48, 32, 1), nn.Conv2d(96, 32, 1)])
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.encoder_trainable = True

    def set_encoder_trainable(self, trainable):
        self.encoder_trainable = trainable
        self.encoder.requires_grad_(trainable)
        self.encoder.train(self.training and trainable)

    def train(self, mode=True):
        super().train(mode)
        if not self.encoder_trainable:
            self.encoder.eval()
        return self

    def forward(self, x):
        x = (x - self.mean) / self.std
        features = []
        for layer, projection in zip(self.encoder, self.projections):
            x = layer(x)
            features.append(projection(x))
        target_size = features[0].shape[-2:]
        fused = features[0]
        for feature in features[1:]:
            fused = fused + F.interpolate(feature, size=target_size,
                                          mode='bilinear', align_corners=False)
        return self.head(fused)


def coarse_gate_loss(logits, target):
    """BCE on every coarse pixel plus Dice on positive images."""
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target, size=logits.shape[-2:], mode='nearest')
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    positive = target.flatten(1).any(1)
    if positive.any():
        p = probability[positive]
        y = target[positive]
        dice = 1 - (2 * (p * y).sum((1, 2, 3)) + 1) / (
            p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
        return bce + dice.mean()
    return bce
