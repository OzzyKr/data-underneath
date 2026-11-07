"""Tier 2: a small frozen pretrained CNN (ImageNet ResNet-18) as a second, independent opinion on patterns.

Frozen and deterministic: no training, no fine-tuning, CPU only, eval mode. The weights (45 MB) are downloaded once
into data/cache/torch and used offline after that. Used only to score how alike a column patch and a legend swatch
look; it never decides alone (see lithology.naming).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[3] / "data" / "cache" / "torch"
INPUT_PX = 160  # every patch is resized to this square before the network
LAYER = 3  # mid-level features: texture-sensitive, less tied to ImageNet object classes


@lru_cache(maxsize=1)
def _model():
    os.environ.setdefault("TORCH_HOME", str(CACHE))
    import torch
    import torchvision

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
    return m


def embed(patches: list[np.ndarray]) -> np.ndarray:
    """Unit-length feature vectors (N, 256) for RGB uint8 patches."""
    import torch

    m = _model()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = torch.stack(
        [(torch.from_numpy(np.ascontiguousarray(p)).permute(2, 0, 1).float() / 255 - mean) / std for p in patches]
    )
    x = torch.nn.functional.interpolate(x, size=(INPUT_PX, INPUT_PX), mode="bilinear", align_corners=False)
    with torch.no_grad():
        h = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        for i, block in enumerate((m.layer1, m.layer2, m.layer3, m.layer4), start=1):
            h = block(h)
            if i == LAYER:
                break
        v = h.mean(dim=(2, 3))
    return torch.nn.functional.normalize(v, dim=1).numpy()


def tiles(a: np.ndarray, size: int, n: int) -> list[np.ndarray]:
    """Up to ``n`` square tiles of side ``size`` spread over an (H, W, 3) patch (the whole patch if smaller)."""
    h, w = a.shape[:2]
    if h < size or w < size:
        return [a]
    out = []
    for k in range(n):
        y = int((h - size) * (k + 0.5) / n)
        x = int((w - size) * (((k * 7) % n) + 0.5) / n)
        out.append(a[y : y + size, x : x + size])
    return out


def similarity(cluster_patches: list[list[np.ndarray]], swatches: list[np.ndarray], tiles_per: int = 6) -> np.ndarray:
    """Cosine similarity (clusters x swatches) between mean tile embeddings."""
    size = max(16, min(min(s.shape[:2]) for s in swatches))
    sw = np.stack([embed(tiles(s, size, 4)).mean(0) for s in swatches])
    sw /= np.linalg.norm(sw, axis=1, keepdims=True)
    cl = []
    for parts in cluster_patches:
        v = np.concatenate([embed(tiles(p, size, tiles_per)) for p in parts]).mean(0)
        cl.append(v / np.linalg.norm(v))
    return np.stack(cl) @ sw.T
