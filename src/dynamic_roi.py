"""Dynamic ROI ranking on the confirmed 3x4 fixed tile geometry."""

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class MobileNetV3SmallTileRanker(nn.Module):
    """Score all 12 tiles jointly so neighboring cracks can share context."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.context = nn.Sequential(nn.Linear(1152 * 12, 256), nn.ReLU(inplace=True),
                                     nn.Dropout(0.1), nn.Linear(256, 12))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        batch, tiles = x.shape[:2]
        x = (x.reshape(batch * tiles, *x.shape[2:]) - self.mean) / self.std
        features = self.features(x)
        pooled = torch.cat([self.avg_pool(features), self.max_pool(features)], dim=1).flatten(1)
        return self.context(pooled.reshape(batch, tiles, -1).flatten(1))


class MobileNetV3SmallGridRanker(nn.Module):
    """Rank tiles with explicit 3x4 neighborhood and position context."""

    def __init__(self, pretrained=True, channels=128):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.tile = nn.Sequential(nn.Linear(1152, channels), nn.ReLU(inplace=True), nn.Dropout(0.1))
        self.position = nn.Parameter(torch.zeros(1, channels, 3, 4))
        self.context = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True), nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False), nn.BatchNorm2d(channels // 2), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(channels // 2, 1, 1)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        batch, tiles = x.shape[:2]
        x = (x.reshape(batch * tiles, *x.shape[2:]) - self.mean) / self.std
        features = self.features(x)
        pooled = torch.cat([self.avg_pool(features), self.max_pool(features)], dim=1).flatten(1)
        tile_features = self.tile(pooled).reshape(batch, 3, 4, -1).permute(0, 3, 1, 2)
        return self.head(self.context(tile_features + self.position)).flatten(1)


class MobileNetV3SmallShallowGridRanker(nn.Module):
    """Fixed-grid ranker that preserves shallow high-resolution evidence.

    The baseline pools only the final MobileNet feature map. This variant pools
    four encoder stages (1/8 through 1/32) with average and maximum pooling,
    then uses the same 3x4 context head. It isolates the effect of retaining
    thin local responses from changes to labels, loss, or top-k selection.
    """

    def __init__(self, pretrained=True, channels=128):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.stages = nn.ModuleList([
            backbone.features[:3],
            backbone.features[3:7],
            backbone.features[7:10],
            backbone.features[10:13],
        ])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 24 + 40 + 96 + 576 channels, with average and max pooling.
        self.tile = nn.Sequential(
            nn.Linear(2 * (24 + 40 + 96 + 576), channels),
            nn.ReLU(inplace=True), nn.Dropout(0.1))
        self.position = nn.Parameter(torch.zeros(1, channels, 3, 4))
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(channels // 2, 1, 1)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        batch, tiles = x.shape[:2]
        x = (x.reshape(batch * tiles, *x.shape[2:]) - self.mean) / self.std
        pooled = []
        for stage in self.stages:
            x = stage(x)
            pooled.append(torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1).flatten(1))
        tile_features = self.tile(torch.cat(pooled, dim=1))
        tile_grid = tile_features.reshape(batch, tiles, -1).reshape(batch, 3, 4, -1).permute(0, 3, 1, 2)
        return self.head(self.context(tile_grid + self.position)).flatten(1)


class MobileNetV3SmallEvidenceGridRanker(nn.Module):
    """Rank fixed tiles while retaining local evidence inside each tile.

    The shallow ranker globally pools every encoder stage. This variant also
    keeps a 2x2 average/max pooled view of the 1/8 stage, so a thin response
    near a tile boundary or in a small local region is not reduced to one
    scalar before the 3x4 frame context head sees it.
    """

    def __init__(self, pretrained=True, channels=128):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.stages = nn.ModuleList([
            backbone.features[:3],
            backbone.features[3:7],
            backbone.features[7:10],
            backbone.features[10:13],
        ])
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.global_max = nn.AdaptiveMaxPool2d(1)
        self.local_avg = nn.AdaptiveAvgPool2d(2)
        self.local_max = nn.AdaptiveMaxPool2d(2)
        global_channels = 2 * (24 + 40 + 96 + 576)
        local_channels = 2 * 24 * 2 * 2
        self.tile = nn.Sequential(
            nn.Linear(global_channels + local_channels, channels),
            nn.ReLU(inplace=True), nn.Dropout(0.1))
        self.position = nn.Parameter(torch.zeros(1, channels, 3, 4))
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(channels // 2, 1, 1)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        batch, tiles = x.shape[:2]
        x = (x.reshape(batch * tiles, *x.shape[2:]) - self.mean) / self.std
        pooled = []
        local = None
        for index, stage in enumerate(self.stages):
            x = stage(x)
            pooled.extend((self.global_avg(x), self.global_max(x)))
            if index == 0:
                local = torch.cat((self.local_avg(x), self.local_max(x)), dim=1).flatten(1)
        tile_features = self.tile(torch.cat([*(p.flatten(1) for p in pooled), local], dim=1))
        tile_grid = tile_features.reshape(batch, tiles, -1).reshape(batch, 3, 4, -1).permute(0, 3, 1, 2)
        return self.head(self.context(tile_grid + self.position)).flatten(1)


class MobileNetV3SmallMassRanker(nn.Module):
    """Rank fixed tiles with presence and crack-pixel-mass heads."""

    def __init__(self, pretrained=True, channels=128):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.tile = nn.Sequential(nn.Linear(1152, channels), nn.ReLU(inplace=True), nn.Dropout(0.1))
        self.position = nn.Parameter(torch.zeros(1, channels, 3, 4))
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2), nn.ReLU(inplace=True),
        )
        self.presence_head = nn.Conv2d(channels // 2, 1, 1)
        self.mass_head = nn.Conv2d(channels // 2, 1, 1)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        batch, tiles = x.shape[:2]
        x = (x.reshape(batch * tiles, *x.shape[2:]) - self.mean) / self.std
        features = self.features(x)
        pooled = torch.cat([self.avg_pool(features), self.max_pool(features)], dim=1).flatten(1)
        tile_features = self.tile(pooled).reshape(batch, 3, 4, -1).permute(0, 3, 1, 2)
        context = self.context(tile_features + self.position)
        return {
            'presence': self.presence_head(context).flatten(1),
            'mass': self.mass_head(context).flatten(1),
        }


class MobileNetV3SmallUtilityRanker(MobileNetV3SmallMassRanker):
    """Same encoder as the mass ranker, with a utility-named regression head."""

    def __init__(self, pretrained=True, channels=128):
        super().__init__(pretrained=pretrained, channels=channels)
        self.utility_head = self.mass_head

    def forward(self, x):
        output = super().forward(x)
        return {'presence': output['presence'], 'utility': output['mass']}


class MobileNetV3SmallContextRanker(nn.Module):
    """Fuse low-resolution full-frame context with fixed-grid tile evidence."""

    def __init__(self, pretrained=True, channels=128):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        tile_backbone = mobilenet_v3_small(weights=weights)
        context_backbone = mobilenet_v3_small(weights=weights)
        self.tile_features = tile_backbone.features
        self.context_features = context_backbone.features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.tile_projection = nn.Sequential(
            nn.Linear(1152, channels), nn.ReLU(inplace=True), nn.Dropout(0.1))
        self.context_projection = nn.Sequential(
            nn.Conv2d(576, channels, 1, bias=False), nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
        self.position = nn.Parameter(torch.zeros(1, channels, 3, 4))
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(channels // 2, 1, 1)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, tiles, frame):
        batch, tile_count = tiles.shape[:2]
        tile_input = (tiles.reshape(batch * tile_count, *tiles.shape[2:]) - self.mean) / self.std
        tile_features = self.tile_features(tile_input)
        tile_pooled = torch.cat([self.avg_pool(tile_features), self.max_pool(tile_features)], dim=1).flatten(1)
        tile_grid = self.tile_projection(tile_pooled).reshape(batch, 3, 4, -1).permute(0, 3, 1, 2)
        frame_features = self.context_features((frame - self.mean) / self.std)
        frame_grid = torch.nn.functional.adaptive_avg_pool2d(frame_features, (3, 4))
        fused = tile_grid + self.context_projection(frame_grid) + self.position
        return self.head(self.context(fused)).flatten(1)


def ranking_loss(logits, target, pairwise_weight=0.0):
    """Weighted BCE with optional positive-vs-negative ranking loss."""
    positive = target.sum().clamp_min(1)
    negative = target.numel() - target.sum()
    pos_weight = (negative / positive).clamp(1, 12)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=pos_weight)
    if pairwise_weight <= 0:
        return loss
    pair_losses = []
    for row, labels in zip(logits, target.bool()):
        positives = row[labels]
        negatives = row[~labels]
        if positives.numel() and negatives.numel():
            pair_losses.append(nn.functional.softplus(0.5 - positives[:, None] + negatives[None, :]).mean())
    if pair_losses:
        loss = loss + pairwise_weight * torch.stack(pair_losses).mean()
    return loss
