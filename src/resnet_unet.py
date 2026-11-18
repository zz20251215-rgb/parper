import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet34_Weights, resnet34


def block(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
        nn.BatchNorm2d(outputs), nn.ReLU(inplace=True),
        nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
        nn.BatchNorm2d(outputs), nn.ReLU(inplace=True))


class ResNet34UNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)
        self.encoder = nn.ModuleList([
            nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),
            nn.Sequential(backbone.maxpool, backbone.layer1),
            backbone.layer2, backbone.layer3, backbone.layer4])
        self.decoder = nn.ModuleList([
            block(512 + 256, 256), block(256 + 128, 128),
            block(128 + 64, 64), block(64 + 64, 32), block(32, 16)])
        self.head = nn.Conv2d(16, 1, 1)
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
        for layer, skip in zip(self.decoder[:-1], reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = layer(torch.cat([x, skip], dim=1))
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return self.head(self.decoder[-1](x))
