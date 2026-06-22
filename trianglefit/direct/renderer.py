from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .model import EllipseParameterTable, EllipseParameters
from .utils import coordinate_grid


@dataclass(frozen=True)
class RenderResult:
    image: torch.Tensor
    alpha: torch.Tensor
    masks: torch.Tensor
    decoded: EllipseParameters
    background_image: torch.Tensor
    background_plus_base_image: torch.Tensor
    coverage_map: torch.Tensor
    active_base_count: int
    active_texture_count: int
    active_accent_count: int


def background_image_from_parameters(
    background_grid_rgb: Optional[torch.Tensor],
    height: int,
    width: int,
    background_rgb: Tuple[float, float, float],
) -> torch.Tensor:
    if background_grid_rgb is None:
        device = torch.device("cpu")
        dtype = torch.float32
        return torch.tensor(background_rgb, device=device, dtype=dtype).view(1, 3, 1, 1).expand(1, 3, height, width).clone()
    return F.interpolate(background_grid_rgb, size=(height, width), mode="nearest")


def _triangle_masks(decoded: EllipseParameters, height: int, width: int, mask_temperature: float, hard_edges: bool) -> torch.Tensor:
    device = decoded.centers.device
    dtype = decoded.centers.dtype
    grid_x, grid_y = coordinate_grid(height, width, device=device, dtype=dtype)

    centers_x = decoded.centers[:, 0].view(-1, 1, 1)
    centers_y = decoded.centers[:, 1].view(-1, 1, 1)
    base = decoded.sizes[:, 0].view(-1, 1, 1).clamp_min(1e-6)
    tri_height = decoded.sizes[:, 1].view(-1, 1, 1).clamp_min(1e-6)
    theta = decoded.theta[:, 0].view(-1, 1, 1)

    dx = grid_x - centers_x
    dy = grid_y - centers_y
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    x_rot = cos_theta * dx + sin_theta * dy
    y_rot = -sin_theta * dx + cos_theta * dy

    y_from_top = y_rot + (tri_height * 0.5)
    half_base_at_y = y_from_top * (base / (2.0 * tri_height))
    margins = torch.stack(
        (
            y_from_top,
            (tri_height * 0.5) - y_rot,
            x_rot + half_base_at_y,
            half_base_at_y - x_rot,
        ),
        dim=0,
    )
    signed_margin = torch.amin(margins, dim=0)
    if hard_edges:
        return (signed_margin >= 0.0).to(dtype=dtype)

    softness = max(float(mask_temperature), 1e-4)
    normalized_margin = signed_margin / softness
    return torch.sigmoid(normalized_margin)


def _composite_subset(
    base_image: torch.Tensor,
    decoded: EllipseParameters,
    masks: torch.Tensor,
    start: int,
    end: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if end <= start:
        coverage = torch.zeros((1, 1, base_image.shape[-2], base_image.shape[-1]), device=base_image.device, dtype=base_image.dtype)
        return base_image.clone(), coverage

    layer_alpha = (masks[start:end] * decoded.alpha[start:end].view(-1, 1, 1)).unsqueeze(1)
    color_layers = decoded.rgb[start:end].view(-1, 3, 1, 1)
    transparency = (1.0 - layer_alpha).clamp(0.0, 1.0)

    reversed_transparency = torch.flip(transparency, dims=(0,))
    reversed_cumprod = torch.cumprod(reversed_transparency, dim=0)
    reversed_exclusive = torch.cat([torch.ones_like(reversed_transparency[:1]), reversed_cumprod[:-1]], dim=0)
    trans_after = torch.flip(reversed_exclusive, dims=(0,))

    background_trans = torch.prod(transparency, dim=0)
    weighted_colors = (layer_alpha * trans_after * color_layers).sum(dim=0, keepdim=True)
    image = (base_image * background_trans) + weighted_colors
    coverage = 1.0 - background_trans
    return image, coverage


def _hard_triangle_mask_single(
    decoded: EllipseParameters,
    index: int,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
) -> torch.Tensor:
    center_x = decoded.centers[index, 0]
    center_y = decoded.centers[index, 1]
    base = decoded.sizes[index, 0].clamp_min(1e-6)
    tri_height = decoded.sizes[index, 1].clamp_min(1e-6)
    theta = decoded.theta[index, 0]

    dx = grid_x - center_x
    dy = grid_y - center_y
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    x_rot = cos_theta * dx + sin_theta * dy
    y_rot = -sin_theta * dx + cos_theta * dy

    y_from_top = y_rot + (tri_height * 0.5)
    half_base_at_y = y_from_top * (base / (2.0 * tri_height))
    inside = (
        (y_from_top >= 0.0)
        & (y_rot <= (tri_height * 0.5))
        & (x_rot >= -half_base_at_y)
        & (x_rot <= half_base_at_y)
    )
    return inside.to(dtype=decoded.centers.dtype)


def render_parameters_image_only(
    decoded: EllipseParameters,
    height: int,
    width: int,
    background_rgb: Tuple[float, float, float],
    active_base_count: Optional[int] = None,
    active_texture_count: Optional[int] = None,
    active_accent_count: Optional[int] = None,
) -> torch.Tensor:
    total_count = decoded.centers.shape[0]
    role_ids = decoded.role_ids
    base_total = int((role_ids == 0).sum().item())
    texture_total = int((role_ids == 1).sum().item())
    accent_total = total_count - base_total - texture_total
    active_base = base_total if active_base_count is None else max(0, min(base_total, int(active_base_count)))
    active_texture = texture_total if active_texture_count is None else max(0, min(texture_total, int(active_texture_count)))
    active_accent = accent_total if active_accent_count is None else max(0, min(accent_total, int(active_accent_count)))

    background_image = background_image_from_parameters(decoded.background_grid_rgb, height, width, background_rgb)
    if background_image.device != decoded.centers.device:
        background_image = background_image.to(device=decoded.centers.device, dtype=decoded.centers.dtype)
    image = background_image.clone()
    grid_x, grid_y = coordinate_grid(height, width, device=decoded.centers.device, dtype=decoded.centers.dtype)

    draw_order = list(range(0, active_base))
    draw_order.extend(range(base_total, base_total + active_texture))
    draw_order.extend(range(base_total + texture_total, base_total + texture_total + active_accent))
    for index in draw_order:
        mask = _hard_triangle_mask_single(decoded=decoded, index=index, grid_x=grid_x, grid_y=grid_y).unsqueeze(1)
        color = decoded.rgb[index].view(1, 3, 1, 1)
        image = image * (1.0 - mask) + (color * mask)
    return image.clamp(0.0, 1.0)


def render_parameters(
    decoded: EllipseParameters,
    height: int,
    width: int,
    mask_temperature: float,
    background_rgb: Tuple[float, float, float],
    hard_edges: bool = False,
    active_base_count: Optional[int] = None,
    active_texture_count: Optional[int] = None,
    active_accent_count: Optional[int] = None,
) -> RenderResult:
    total_count = decoded.centers.shape[0]
    role_ids = decoded.role_ids
    base_total = int((role_ids == 0).sum().item())
    texture_total = int((role_ids == 1).sum().item())
    accent_total = total_count - base_total - texture_total
    active_base = base_total if active_base_count is None else max(0, min(base_total, int(active_base_count)))
    active_texture = texture_total if active_texture_count is None else max(0, min(texture_total, int(active_texture_count)))
    active_accent = accent_total if active_accent_count is None else max(0, min(accent_total, int(active_accent_count)))

    masks = _triangle_masks(decoded=decoded, height=height, width=width, mask_temperature=mask_temperature, hard_edges=hard_edges)

    background_image = background_image_from_parameters(decoded.background_grid_rgb, height, width, background_rgb)
    if background_image.device != decoded.centers.device:
        background_image = background_image.to(device=decoded.centers.device, dtype=decoded.centers.dtype)

    base_start = 0
    base_end = active_base
    texture_start = base_total
    texture_end = base_total + active_texture
    accent_start = base_total + texture_total
    accent_end = accent_start + active_accent

    background_plus_base_image, base_coverage = _composite_subset(background_image, decoded, masks, base_start, base_end)
    texture_image, texture_coverage = _composite_subset(background_plus_base_image, decoded, masks, texture_start, texture_end)
    final_image, accent_coverage = _composite_subset(texture_image, decoded, masks, accent_start, accent_end)
    coverage_map = base_coverage + texture_coverage + accent_coverage
    final_alpha = torch.clamp(coverage_map, 0.0, 1.0)

    return RenderResult(
        image=final_image.clamp(0.0, 1.0),
        alpha=final_alpha,
        masks=masks,
        decoded=decoded,
        background_image=background_image.clamp(0.0, 1.0),
        background_plus_base_image=background_plus_base_image.clamp(0.0, 1.0),
        coverage_map=coverage_map,
        active_base_count=active_base,
        active_texture_count=active_texture,
        active_accent_count=active_accent,
    )


def render_table(
    table: EllipseParameterTable,
    height: int,
    width: int,
    mask_temperature: float,
    background_rgb: Tuple[float, float, float],
    hard_edges: bool = False,
    active_base_count: Optional[int] = None,
    active_texture_count: Optional[int] = None,
    active_accent_count: Optional[int] = None,
) -> RenderResult:
    min_size = 1.0 / float(max(height, width))
    decoded = table.decode(
        min_size=min_size,
        active_base_count=active_base_count,
        active_texture_count=active_texture_count,
        active_accent_count=active_accent_count,
    )
    return render_parameters(
        decoded=decoded,
        height=height,
        width=width,
        mask_temperature=mask_temperature,
        background_rgb=background_rgb,
        hard_edges=hard_edges,
        active_base_count=active_base_count,
        active_texture_count=active_texture_count,
        active_accent_count=active_accent_count,
    )


