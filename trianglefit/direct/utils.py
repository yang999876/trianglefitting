from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def inverse_sigmoid(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    clamped = values.clamp(eps, 1.0 - eps)
    return torch.log(clamped) - torch.log1p(-clamped)


def inverse_softplus(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    values = values.clamp_min(eps)
    return values + torch.log(-torch.expm1(-values))


def rgb_rmse(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((first - second) ** 2))


def coordinate_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    y = torch.linspace(0.5 / height, 1.0 - (0.5 / height), height, device=device, dtype=dtype)
    x = torch.linspace(0.5 / width, 1.0 - (0.5 / width), width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return grid_x.unsqueeze(0), grid_y.unsqueeze(0)


def serialize_json(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def tensor_to_uint8_image(image: torch.Tensor) -> np.ndarray:
    tensor = image.detach().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = tensor.permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def format_float(value: torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def mean_window_color(image: torch.Tensor, center_x: int, center_y: int, radius: int = 4) -> torch.Tensor:
    _, _, height, width = image.shape
    x0 = max(0, center_x - radius)
    x1 = min(width, center_x + radius + 1)
    y0 = max(0, center_y - radius)
    y1 = min(height, center_y + radius + 1)
    window = image[:, :, y0:y1, x0:x1]
    return window.mean(dim=(0, 2, 3))


def normalized_triangle_area(base: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    return 0.5 * base * height


def flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {key: format_float(value) for key, value in metrics.items()}


def chunked(items: Iterable[Any], size: int) -> Iterable[Tuple[Any, ...]]:
    buffer = []
    for item in items:
        buffer.append(item)
        if len(buffer) == size:
            yield tuple(buffer)
            buffer = []
    if buffer:
        yield tuple(buffer)


