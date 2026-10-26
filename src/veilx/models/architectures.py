"""Inference-only re-implementations of the two backbone architectures.

Module names and ``Sequential`` positions reproduce the authors' code at the pinned revisions
(CVLface ``models/iresnet/model.py`` IR_101 and ``models/iresnet_insightface/model.py``
iresnet100) so that checkpoint keys map one-to-one; ``scripts/verify_models.py`` checks
numerical equality against the authors' own modules. Training-only code (weight init,
flop counting) is omitted: every tensor is overwritten by the strict loader.
"""

from __future__ import annotations

import torch
from torch import nn

# ----------------------------------------------------------------- CVLface IR-101 (AdaFace)


class _Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class _BasicBlockIR(nn.Module):
    def __init__(self, in_channel: int, depth: int, stride: int) -> None:
        super().__init__()
        self.shortcut_layer: nn.Module
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False), nn.BatchNorm2d(depth)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_layer(x) + self.shortcut_layer(x)


# (in_channel, depth, num_units) per stage for IR-101 ("num_layers=100" in the authors' code)
_IR101_STAGES = ((64, 64, 3), (64, 128, 13), (128, 256, 30), (256, 512, 3))


class CVLfaceIR101(nn.Module):
    def __init__(self, output_dim: int = 512) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64)
        )
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            _Flatten(),
            nn.Linear(512 * 7 * 7, output_dim),
            nn.BatchNorm1d(output_dim, affine=False),
        )
        units: list[nn.Module] = []
        for in_c, depth, n in _IR101_STAGES:
            units.append(_BasicBlockIR(in_c, depth, 2))
            units.extend(_BasicBlockIR(depth, depth, 1) for _ in range(n - 1))
        self.body = nn.Sequential(*units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.body(self.input_layer(x)))


# ------------------------------------------------- InsightFace iresnet100 (ArcFace, CVLface copy)


class _IBasicBlock(nn.Module):
    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn3(self.conv2(self.prelu(self.bn2(self.conv1(self.bn1(x))))))
        identity = self.downsample(x) if self.downsample is not None else x
        return out + identity


class InsightfaceIResNet100(nn.Module):
    fc_scale = 7 * 7

    def __init__(self, num_features: int = 512) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 13)
        self.layer3 = self._make_layer(256, 30)
        self.layer4 = self._make_layer(512, 3)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(512 * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)

    def _make_layer(self, planes: int, blocks: int) -> nn.Sequential:
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes, 1, 2, bias=False),
            nn.BatchNorm2d(planes, eps=1e-05),
        )
        layers: list[nn.Module] = [_IBasicBlock(self.inplanes, planes, 2, downsample)]
        self.inplanes = planes
        layers.extend(_IBasicBlock(planes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.bn2(x), 1)
        return self.features(self.fc(self.dropout(x)))


def build_architecture(name: str, embedding_dim: int = 512) -> nn.Module:
    if name == "cvlface_ir101":
        return CVLfaceIR101(embedding_dim)
    if name == "insightface_iresnet100":
        return InsightfaceIResNet100(embedding_dim)
    raise ValueError(f"unknown architecture {name!r}")
