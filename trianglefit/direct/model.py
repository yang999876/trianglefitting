from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from .config import StageConfig
from .utils import inverse_sigmoid, inverse_softplus

ROLE_BASE = 0
ROLE_TEXTURE = 1
ROLE_ACCENT = 2
ROLE_NAMES = {
    ROLE_BASE: "base",
    ROLE_TEXTURE: "texture",
    ROLE_ACCENT: "accent",
}
ROLE_NAME_TO_ID = {name: role_id for role_id, name in ROLE_NAMES.items()}


@dataclass(frozen=True)
class EllipseParameters:
    centers: torch.Tensor
    sizes: torch.Tensor
    theta: torch.Tensor
    rgb: torch.Tensor
    alpha: torch.Tensor
    role_ids: torch.Tensor
    background_grid_rgb: Optional[torch.Tensor] = None

    def to_json_list(self) -> List[Dict[str, float]]:
        centers = self.centers.detach().cpu()
        sizes = self.sizes.detach().cpu()
        theta = self.theta.detach().cpu()
        rgb = self.rgb.detach().cpu()
        role_ids = self.role_ids.detach().cpu()
        payload = []
        for index in range(centers.shape[0]):
            role_name = ROLE_NAMES.get(int(role_ids[index].item()), "texture")
            payload.append(
                {
                    "index": index,
                    "kind": role_name,
                    "cx": float(centers[index, 0].item()),
                    "cy": float(centers[index, 1].item()),
                    "base": float(sizes[index, 0].item()),
                    "height": float(sizes[index, 1].item()),
                    "theta": float(theta[index, 0].item()),
                    "r": float(rgb[index, 0].item()),
                    "g": float(rgb[index, 1].item()),
                    "b": float(rgb[index, 2].item()),
                }
            )
        return payload

    def role_mask(self, role_id: int) -> torch.Tensor:
        return (self.role_ids == role_id).to(dtype=self.centers.dtype).view(-1, 1)


class EllipseParameterTable(nn.Module):
    def __init__(
        self,
        center_logits: torch.Tensor,
        size_raw: torch.Tensor,
        theta_raw: torch.Tensor,
        rgb_logits: torch.Tensor,
        background_grid_rgb: torch.Tensor,
        base_count: int,
        texture_count: int,
    ) -> None:
        super().__init__()
        self.center_logits = nn.Parameter(center_logits)
        self.size_raw = nn.Parameter(size_raw)
        self.theta_raw = nn.Parameter(theta_raw)
        self.rgb_logits = nn.Parameter(rgb_logits)
        self.register_buffer("background_grid_rgb", background_grid_rgb)
        self.base_count = int(base_count)
        self.texture_count_value = int(texture_count)

    @classmethod
    def from_decoded(
        cls,
        centers: torch.Tensor,
        sizes: torch.Tensor,
        theta: torch.Tensor,
        rgb: torch.Tensor,
        background_grid_rgb: torch.Tensor,
        base_count: int,
        texture_count: int,
    ) -> "EllipseParameterTable":
        return cls(
            center_logits=inverse_sigmoid(centers),
            size_raw=inverse_softplus(sizes),
            theta_raw=theta,
            rgb_logits=inverse_sigmoid(rgb),
            background_grid_rgb=background_grid_rgb,
            base_count=base_count,
            texture_count=texture_count,
        )

    def texture_count(self) -> int:
        return max(0, min(self.texture_count_value, int(self.center_logits.shape[0]) - self.base_count))

    def accent_count(self) -> int:
        return max(0, int(self.center_logits.shape[0]) - self.base_count - self.texture_count())

    def role_range(self, kind: str, active_count: Optional[int] = None) -> Tuple[int, int]:
        if kind == "base":
            total = self.base_count
            count = total if active_count is None else max(0, min(total, int(active_count)))
            return 0, count
        if kind == "texture":
            total = self.texture_count()
            count = total if active_count is None else max(0, min(total, int(active_count)))
            start = self.base_count
            return start, start + count
        if kind == "accent":
            total = self.accent_count()
            count = total if active_count is None else max(0, min(total, int(active_count)))
            start = self.base_count + self.texture_count()
            return start, start + count
        raise ValueError("Unsupported role kind: %s" % kind)

    def decode(
        self,
        min_size: float = 1e-3,
        active_base_count: Optional[int] = None,
        active_texture_count: Optional[int] = None,
        active_accent_count: Optional[int] = None,
    ) -> EllipseParameters:
        centers = torch.sigmoid(self.center_logits)
        sizes = torch.nn.functional.softplus(self.size_raw) + min_size
        theta = self.theta_raw
        rgb = torch.sigmoid(self.rgb_logits)
        background_grid_rgb = self.background_grid_rgb

        total_count = rgb.shape[0]
        role_ids = torch.full((total_count,), ROLE_ACCENT, device=rgb.device, dtype=torch.long)
        role_ids[: self.base_count] = ROLE_BASE
        role_ids[self.base_count : self.base_count + self.texture_count()] = ROLE_TEXTURE

        if active_base_count is None:
            active_base_count = self.base_count
        if active_texture_count is None:
            active_texture_count = self.texture_count()
        if active_accent_count is None:
            active_accent_count = self.accent_count()

        active_mask = torch.zeros((total_count, 1), device=rgb.device, dtype=rgb.dtype)
        for kind, count in (("base", active_base_count), ("texture", active_texture_count), ("accent", active_accent_count)):
            start, end = self.role_range(kind, count)
            if end > start:
                active_mask[start:end] = 1.0

        alpha = torch.ones((total_count, 1), device=rgb.device, dtype=rgb.dtype) * active_mask

        return EllipseParameters(
            centers=centers,
            sizes=sizes,
            theta=theta,
            rgb=rgb,
            alpha=alpha,
            role_ids=role_ids,
            background_grid_rgb=background_grid_rgb.clamp(0.0, 1.0),
        )

    def apply_stage(self, stage: StageConfig) -> None:
        self.center_logits.requires_grad_(stage.optimize_centers)
        self.size_raw.requires_grad_(stage.optimize_radii)
        self.theta_raw.requires_grad_(stage.optimize_theta)
        self.rgb_logits.requires_grad_(stage.optimize_color)

    def optimizer_groups(self, stage: StageConfig) -> List[Dict[str, object]]:
        groups: List[Dict[str, object]] = []
        geometry_params: List[nn.Parameter] = []
        if self.center_logits.requires_grad:
            geometry_params.append(self.center_logits)
        if self.size_raw.requires_grad:
            geometry_params.append(self.size_raw)
        if self.theta_raw.requires_grad:
            geometry_params.append(self.theta_raw)
        if geometry_params:
            groups.append({"params": geometry_params, "lr": stage.geometry_lr})

        color_params: List[nn.Parameter] = []
        if self.rgb_logits.requires_grad:
            color_params.append(self.rgb_logits)
        if color_params:
            groups.append({"params": color_params, "lr": stage.color_lr})
        return groups

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        for parameter in self.parameters():
            if parameter.requires_grad:
                yield parameter

    def set_decoded_slice(
        self,
        start: int,
        end: int,
        centers: torch.Tensor,
        sizes: torch.Tensor,
        theta: torch.Tensor,
        rgb: torch.Tensor,
    ) -> None:
        if end <= start:
            return
        with torch.no_grad():
            self.center_logits[start:end].copy_(inverse_sigmoid(centers))
            self.size_raw[start:end].copy_(inverse_softplus(sizes))
            self.theta_raw[start:end].copy_(theta)
            self.rgb_logits[start:end].copy_(inverse_sigmoid(rgb))


def ellipse_parameters_from_json(
    entries: List[Dict[str, float]],
    device: Optional[torch.device] = None,
    background_grid_rgb: Optional[torch.Tensor] = None,
) -> EllipseParameters:
    centers = torch.tensor([[item["cx"], item["cy"]] for item in entries], dtype=torch.float32, device=device)
    sizes = torch.tensor([[item["base"], item["height"]] for item in entries], dtype=torch.float32, device=device)
    theta = torch.tensor([[item["theta"]] for item in entries], dtype=torch.float32, device=device)
    rgb = torch.tensor([[item["r"], item["g"], item["b"]] for item in entries], dtype=torch.float32, device=device)
    alpha = torch.ones((len(entries), 1), dtype=torch.float32, device=device)
    role_ids = []
    for item in entries:
        kind = item.get("kind", "texture")
        if kind == "detail":
            kind = "texture"
        role_ids.append(ROLE_NAME_TO_ID.get(kind, ROLE_TEXTURE))
    role_ids_tensor = torch.tensor(role_ids, dtype=torch.long, device=device)
    return EllipseParameters(
        centers=centers,
        sizes=sizes,
        theta=theta,
        rgb=rgb,
        alpha=alpha,
        role_ids=role_ids_tensor,
        background_grid_rgb=background_grid_rgb,
    )


