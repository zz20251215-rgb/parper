"""Fixed-grid first-stage gate: classify each 512x512 tile independently."""

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class MobileNetV3SmallTileGate(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(nn.Linear(1152, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(0.1), nn.Linear(128, 1))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        features = self.features(x)
        pooled = torch.cat([self.avg_pool(features), self.max_pool(features)], dim=1)
        return self.head(pooled.flatten(1)).squeeze(1)
