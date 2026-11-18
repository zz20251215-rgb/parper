import torch
from torch import nn


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [32, 64, 128, 256, 512]
        self.down = nn.ModuleList(block(a, b) for a, b in zip([3] + channels[:-1], channels))
        self.up = nn.ModuleList(nn.ConvTranspose2d(a, b, 2, stride=2)
                                for a, b in zip(channels[:0:-1], channels[-2::-1]))
        self.decode = nn.ModuleList(block(a, b) for a, b in zip(channels[:0:-1], channels[-2::-1]))
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        skips = []
        for layer in self.down[:-1]:
            x = layer(x)
            skips.append(x)
            x = nn.functional.max_pool2d(x, 2)
        x = self.down[-1](x)
        for up, layer, skip in zip(self.up, self.decode, reversed(skips)):
            x = layer(torch.cat([up(x), skip], dim=1))
        return self.out(x)
