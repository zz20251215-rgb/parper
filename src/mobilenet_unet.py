import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class SeparableBlock(nn.Module):
    """Depthwise-separable convolution used in the lightweight decoder."""

    def __init__(self, inputs, outputs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, inputs, 3, padding=1, groups=inputs, bias=False),
            nn.BatchNorm2d(inputs),
            nn.ReLU6(inplace=True),
            nn.Conv2d(inputs, outputs, 1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU6(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, groups=outputs, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU6(inplace=True),
            nn.Conv2d(outputs, outputs, 1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class MobileNetV3LargeUNet(nn.Module):
    """High-resolution U-Net with a MobileNetV3-Large encoder.

    Features are taken before the classifier and retained at 1/2, 1/4, 1/8,
    1/16 and 1/32 input resolution so thin structures can use shallow skips.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        self.encoder = nn.ModuleList([
            backbone.features[:1],
            backbone.features[1:3],
            backbone.features[3:5],
            backbone.features[5:8],
            backbone.features[8:14],
        ])
        self.decoder = nn.ModuleList([
            SeparableBlock(160 + 80, 96),
            SeparableBlock(96 + 40, 64),
            SeparableBlock(64 + 24, 40),
            SeparableBlock(40 + 16, 24),
        ])
        self.head = nn.Conv2d(24, 1, 1)
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
        size = x.shape[-2:]
        x = (x - self.mean) / self.std
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        for layer, skip in zip(self.decoder, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = layer(torch.cat([x, skip], dim=1))
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return self.head(x)


class MobileNetV3LargeRefineUNet(MobileNetV3LargeUNet):
    """MobileNetV3-Large U-Net with a full-resolution refinement block.

    The baseline predicts immediately after its last interpolation. This
    variant adds one depthwise-separable spatial block at the native 512x512
    resolution so thin crack responses can be reconnected locally before the
    final 1x1 prediction. Encoder, skips, loss, and data protocol are shared.
    """

    def __init__(self, pretrained=True):
        super().__init__(pretrained=pretrained)
        self.refine = SeparableBlock(24, 24)

    def forward(self, x):
        size = x.shape[-2:]
        x = (x - self.mean) / self.std
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        for layer, skip in zip(self.decoder, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = layer(torch.cat([x, skip], dim=1))
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return self.head(self.refine(x))
