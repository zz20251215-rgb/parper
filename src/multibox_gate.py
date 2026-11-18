"""Multi-scale, multi-candidate first-stage gate for stitched tunnel frames."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class MobileNetV3SmallMultiBoxGate(nn.Module):
    """Predict image presence plus anchor-free horizontal candidates.

    The three heads operate at strides 4, 8 and 16. Each candidate is encoded
    as normalized ``(cx, cy, w, h)`` at the center cell; objectness selects
    candidate cells. The design keeps a stride-4 path for very small cracks.
    """

    def __init__(self, pretrained=True, channels=48):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.stage4 = backbone.features[:2]      # stride 4, 16 channels
        self.stage8 = backbone.features[2:4]     # stride 8, 24 channels
        self.stage16 = backbone.features[4:9]    # stride 16, 48 channels
        self.stage32 = backbone.features[9:12]   # stride 32, 96 channels
        self.proj4 = nn.Conv2d(16, channels, 1)
        self.proj8 = nn.Conv2d(24, channels, 1)
        self.proj16 = nn.Conv2d(48, channels, 1)
        self.proj32 = nn.Conv2d(96, channels, 1)
        self.fuse4 = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.fuse8 = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.fuse16 = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.head4 = nn.Conv2d(channels, 5, 1)
        self.head8 = nn.Conv2d(channels, 5, 1)
        self.head16 = nn.Conv2d(channels, 5, 1)
        self.image_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Linear(channels, 1))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        f4 = self.stage4(x)
        f8 = self.stage8(f4)
        f16 = self.stage16(f8)
        f32 = self.stage32(f16)
        p4 = self.proj4(f4)
        p8 = self.proj8(f8) + F.interpolate(self.proj16(f16), size=f8.shape[-2:],
                                             mode='bilinear', align_corners=False)
        p16 = self.proj16(f16) + F.interpolate(self.proj32(f32), size=f16.shape[-2:],
                                               mode='bilinear', align_corners=False)
        p4 = self.fuse4(p4 + F.interpolate(p8, size=p4.shape[-2:], mode='bilinear', align_corners=False))
        p8 = self.fuse8(p8)
        p16 = self.fuse16(p16)
        return {'image_logit': self.image_head(p16).squeeze(1),
                'stride4': self.head4(p4), 'stride8': self.head8(p8),
                'stride16': self.head16(p16)}


class MobileNetV3SmallCenterGate(nn.Module):
    """Multi-scale center heatmap gate with context windows decoded later."""

    def __init__(self, pretrained=True, channels=48):
        super().__init__()
        base = MobileNetV3SmallMultiBoxGate(pretrained=pretrained, channels=channels)
        self.stage4, self.stage8, self.stage16, self.stage32 = base.stage4, base.stage8, base.stage16, base.stage32
        self.proj4, self.proj8, self.proj16, self.proj32 = base.proj4, base.proj8, base.proj16, base.proj32
        self.fuse4, self.fuse8, self.fuse16 = base.fuse4, base.fuse8, base.fuse16
        self.head4 = nn.Conv2d(channels, 1, 1)
        self.head8 = nn.Conv2d(channels, 1, 1)
        self.head16 = nn.Conv2d(channels, 1, 1)
        self.image_head = base.image_head
        self.register_buffer('mean', base.mean.detach().clone())
        self.register_buffer('std', base.std.detach().clone())

    def forward(self, x):
        x = (x - self.mean) / self.std
        f4 = self.stage4(x); f8 = self.stage8(f4); f16 = self.stage16(f8); f32 = self.stage32(f16)
        p4 = self.proj4(f4)
        p8 = self.proj8(f8) + F.interpolate(self.proj16(f16), size=f8.shape[-2:], mode='bilinear', align_corners=False)
        p16 = self.proj16(f16) + F.interpolate(self.proj32(f32), size=f16.shape[-2:], mode='bilinear', align_corners=False)
        p4 = self.fuse4(p4 + F.interpolate(p8, size=p4.shape[-2:], mode='bilinear', align_corners=False))
        p8 = self.fuse8(p8); p16 = self.fuse16(p16)
        return {'image_logit': self.image_head(p16).squeeze(1),
                'stride4': self.head4(p4), 'stride8': self.head8(p8), 'stride16': self.head16(p16)}


def center_gate_loss(output, targets, image_present, objectness_weight=2.0):
    total = F.binary_cross_entropy_with_logits(output['image_logit'], image_present.float())
    for name in ('stride4', 'stride8', 'stride16'):
        logits = output[name]
        target = torch.zeros_like(logits[:, 0])
        stride = int(name[6:])
        for batch_index, boxes in enumerate(targets):
            for cx, cy, width, height in boxes:
                max_size = max(float(width), float(height))
                assigned = 4 if max_size <= 0.12 else (8 if max_size <= 0.35 else 16)
                if assigned != stride: continue
                gx = min(logits.shape[-1] - 1, max(0, int(float(cx) * logits.shape[-1])))
                gy = min(logits.shape[-2] - 1, max(0, int(float(cy) * logits.shape[-2])))
                target[batch_index, gy, gx] = 1
        total = total + objectness_weight * F.binary_cross_entropy_with_logits(
            logits[:, 0], target, pos_weight=torch.tensor(12.0, device=logits.device))
    return total


def decode_centers(output, threshold=0.5, max_candidates=32):
    batch = output['image_logit'].shape[0]; decoded = [[] for _ in range(batch)]
    for name in ('stride4', 'stride8', 'stride16'):
        scores = output[name][:, 0].sigmoid(); stride = int(name[6:])
        for index in range(batch):
            ys, xs = torch.where(scores[index] >= threshold)
            for y, x in zip(ys.tolist(), xs.tolist()):
                cx = (x + 0.5) / scores.shape[-1]; cy = (y + 0.5) / scores.shape[-2]
                window = {4: 0.10, 8: 0.22, 16: 0.45}[stride]
                decoded[index].append((float(scores[index, y, x]), torch.tensor([cx, cy, window, window])))
    for index in range(batch):
        decoded[index].sort(key=lambda item: item[0], reverse=True); decoded[index] = decoded[index][:max_candidates]
    return decoded


def multibox_loss(output, targets, image_present, objectness_weight=2.0, box_weight=5.0):
    """Compute image gate, objectness and center-cell box regression losses."""
    total = F.binary_cross_entropy_with_logits(output['image_logit'], image_present.float())
    for name in ('stride4', 'stride8', 'stride16'):
        logits = output[name]
        obj_target = torch.zeros_like(logits[:, 0])
        box_target = torch.zeros((logits.shape[0], 4, *logits.shape[-2:]), device=logits.device)
        box_valid = torch.zeros_like(obj_target)
        stride = int(name[6:])
        for batch_index, boxes in enumerate(targets):
            for cx, cy, width, height in boxes:
                max_size = max(float(width), float(height))
                assigned = 4 if max_size <= 0.12 else (8 if max_size <= 0.35 else 16)
                if assigned != stride:
                    continue
                gx = min(logits.shape[-1] - 1, max(0, int(float(cx) * logits.shape[-1])))
                gy = min(logits.shape[-2] - 1, max(0, int(float(cy) * logits.shape[-2])))
                obj_target[batch_index, gy, gx] = 1
                box_target[batch_index, :, gy, gx] = torch.tensor(
                    [cx, cy, width, height], device=logits.device)
                box_valid[batch_index, gy, gx] = 1
        positive = box_valid.bool()
        pos_weight = torch.tensor(12.0, device=logits.device)
        total = total + objectness_weight * F.binary_cross_entropy_with_logits(
            logits[:, 0], obj_target, pos_weight=pos_weight)
        if positive.any():
            total = total + box_weight * F.smooth_l1_loss(
                logits[:, 1:5].sigmoid().permute(0, 2, 3, 1)[positive],
                box_target.permute(0, 2, 3, 1)[positive])
    return total


def decode_candidates(output, threshold=0.2, max_candidates=64):
    """Decode candidate maps into per-image normalized boxes and scores."""
    batch = output['image_logit'].shape[0]
    decoded = [[] for _ in range(batch)]
    for name in ('stride4', 'stride8', 'stride16'):
        values = output[name]
        scores = values[:, 0].sigmoid()
        boxes = values[:, 1:5].sigmoid()
        for index in range(batch):
            ys, xs = torch.where(scores[index] >= threshold)
            for y, x in zip(ys.tolist(), xs.tolist()):
                decoded[index].append((float(scores[index, y, x]),
                                       boxes[index, :, y, x].detach().cpu()))
    for index in range(batch):
        decoded[index].sort(key=lambda item: item[0], reverse=True)
        decoded[index] = decoded[index][:max_candidates]
    return decoded
