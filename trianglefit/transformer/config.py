from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class TransformerTriangleConfig:
    num_triangles: int = 300
    work_size: int = 256
    steps: int = 3000
    seed: int = 42
    lr: float = 1e-4
    mask_temperature: float = 0.02
    metrics_every: int = 200
    progress_every: int = 200
    checkpoint_every: int = 500
    base_triangle_fraction: float = 0.20
    texture_triangle_fraction: float = 0.80
    l1_weight: float = 1.0
    lpips_weight: float = 0.3
    d_model: int = 512
    num_heads: int = 8
    num_decoder_layers: int = 6
    dim_feedforward: int = 512
    pretrained_backbone: bool = True
    freeze_backbone: bool = True
    min_size: float = 0.002
    base_max_size: float = 0.34
    texture_max_size: float = 0.16

    def base_count(self) -> int:
        return max(1, min(self.num_triangles, int(round(self.num_triangles * self.base_triangle_fraction))))

    def texture_count(self) -> int:
        remaining = max(0, self.num_triangles - self.base_count())
        if remaining == 0:
            return 0
        texture_count = max(1, int(round(self.num_triangles * self.texture_triangle_fraction)))
        return min(remaining, texture_count)

    def accent_count(self) -> int:
        return max(0, self.num_triangles - self.base_count() - self.texture_count())

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


