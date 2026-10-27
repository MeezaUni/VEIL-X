"""Lightweight visibility estimator: U-Net-lite encoder-decoder, 112x112 RGB -> 28x28 logits.

Input normalisation: (x / 255 - 0.5) / 0.5, RGB, NCHW. Encoder stages at 112/56/28/14/7 with
32/64/128/256/256 channels; the decoder upsamples 7 -> 14 -> 28 with skip connections. Output:
one logit per 4x4-pixel cell (P(visible) = sigmoid). Trained from scratch (no pretrained weights).
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class VisNet(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        w = width
        self.e1 = _block(3, w)  # 112
        self.e2 = _block(w, 2 * w)  # 56
        self.e3 = _block(2 * w, 4 * w)  # 28
        self.e4 = _block(4 * w, 8 * w)  # 14
        self.e5 = _block(8 * w, 8 * w)  # 7
        self.pool = nn.MaxPool2d(2)
        self.up5 = nn.ConvTranspose2d(8 * w, 8 * w, 2, stride=2)  # 7 -> 14
        self.d4 = _block(16 * w, 4 * w)
        self.up4 = nn.ConvTranspose2d(4 * w, 4 * w, 2, stride=2)  # 14 -> 28
        self.d3 = _block(8 * w, 2 * w)
        self.head = nn.Conv2d(2 * w, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x4 = self.e4(self.pool(x3))
        x5 = self.e5(self.pool(x4))
        y = self.d4(torch.cat([self.up5(x5), x4], 1))
        y = self.d3(torch.cat([self.up4(y), x3], 1))
        logits: torch.Tensor = self.head(y)[:, 0]
        return logits


def to_input(images: NDArray[np.uint8]) -> NDArray[np.float32]:
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (112, 112, 3):
        raise ValueError(f"expected (N, 112, 112, 3) uint8, got {images.shape} {images.dtype}")
    x = (images.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def n_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
