from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .utils import tensor_to_uint8_image


@dataclass(frozen=True)
class LoadedImage:
    path: str
    original: torch.Tensor
    working: torch.Tensor
    original_size: Tuple[int, int]
    working_size: Tuple[int, int]


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _composite_to_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    composed = Image.alpha_composite(white, rgba)
    return composed.convert("RGB")


def resize_for_working(image: torch.Tensor, work_size: int) -> torch.Tensor:
    _, _, height, width = image.shape
    longest = max(height, width)
    if longest <= work_size:
        return image.clone()

    scale = float(work_size) / float(longest)
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    return F.interpolate(
        image,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def load_image(path: Path, work_size: int) -> LoadedImage:
    pil = Image.open(path)
    rgb = _composite_to_white(pil)
    original = _pil_to_tensor(rgb)
    working = resize_for_working(original, work_size)
    _, _, original_height, original_width = original.shape
    _, _, working_height, working_width = working.shape
    return LoadedImage(
        path=str(path),
        original=original,
        working=working,
        original_size=(original_width, original_height),
        working_size=(working_width, working_height),
    )


def save_image(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_image(image)).save(path)


