"""Lightweight first-stage crack recognizer for the cascade.

The first version predicts one image-level crack score and one enclosing box.
It is intentionally a coarse gate; pixel masks remain the responsibility of
the second-stage segmenter.
"""

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def mask_to_box(mask):
    """Return a normalized ``(cx, cy, w, h)`` box from a binary mask.

    Empty masks return ``(0, 0, 0, 0)``. Coordinates are normalized by the
    mask width and height, so labels survive preview resizing.
    """
    if mask.ndim == 3:
        mask = mask[0]
    ys, xs = torch.where(mask > 0)
    if len(xs) == 0:
        return torch.zeros(4, dtype=torch.float32, device=mask.device)
    height, width = mask.shape[-2:]
    x0, x1 = xs.min().float(), xs.max().float() + 1
    y0, y1 = ys.min().float(), ys.max().float() + 1
    return torch.stack(((x0 + x1) / (2 * width), (y0 + y1) / (2 * height),
                        (x1 - x0) / width, (y1 - y0) / height))


class MobileNetV3SmallRecognizer(nn.Module):
    """MobileNetV3-Small gate with coarse multi-scale feature fusion."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.stages = nn.ModuleList([
            backbone.features[:2],   # 1/4, 16 channels
            backbone.features[2:5],  # 1/16, 40 channels
            backbone.features[5:9],  # 1/16, 48 channels
            backbone.features[9:12], # 1/32, 96 channels
        ])
        self.projections = nn.ModuleList([
            nn.Conv2d(16, 32, 1), nn.Conv2d(40, 32, 1),
            nn.Conv2d(48, 32, 1), nn.Conv2d(96, 32, 1)])
        self.fuse = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, 1)
        self.box = nn.Sequential(nn.Linear(32, 32), nn.ReLU(inplace=True),
                                 nn.Linear(32, 4), nn.Sigmoid())
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.encoder_trainable = True

    def set_encoder_trainable(self, trainable):
        self.encoder_trainable = trainable
        self.stages.requires_grad_(trainable)
        self.stages.train(self.training and trainable)

    def train(self, mode=True):
        super().train(mode)
        if not self.encoder_trainable:
            self.stages.eval()
        return self

    def forward(self, x):
        x = (x - self.mean) / self.std
        features = []
        for stage, projection in zip(self.stages, self.projections):
            x = stage(x)
            features.append(projection(x))
        target_size = features[0].shape[-2:]
        fused = features[0]
        for feature in features[1:]:
            fused = fused + nn.functional.interpolate(
                feature, size=target_size, mode='bilinear', align_corners=False)
        pooled = self.fuse(fused).flatten(1)
        return {'logit': self.classifier(pooled).squeeze(1), 'box': self.box(pooled)}


def recognizer_loss(output, target_present, target_box, box_weight=2.0):
    """Joint gate and box loss; box regression is applied to positive images."""
    present_loss = nn.functional.binary_cross_entropy_with_logits(
        output['logit'], target_present.float())
    positive = target_present.bool()
    if positive.any():
        box_loss = nn.functional.smooth_l1_loss(output['box'][positive], target_box[positive])
    else:
        box_loss = output['box'].sum() * 0
    return present_loss + box_weight * box_loss
