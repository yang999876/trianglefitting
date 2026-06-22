from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .config import FitConfig
from .model import EllipseParameterTable
from .utils import mean_window_color


@dataclass(frozen=True)
class TargetAnalysis:
    edges: torch.Tensor
    coherence: torch.Tensor
    tangent_theta: torch.Tensor


def initialize_background_grid(target: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
    return F.adaptive_avg_pool2d(target, output_size=grid_size).clamp(0.0, 1.0)


def _sample_from_distribution(
    distribution: torch.Tensor,
    num_samples: int,
    width: int,
    height: int,
    seed: int,
) -> torch.Tensor:
    if num_samples <= 0:
        return torch.empty((0, 2), device=distribution.device, dtype=distribution.dtype)
    generator = torch.Generator(device=distribution.device)
    generator.manual_seed(seed)
    flat = distribution.reshape(-1)
    flat = flat / flat.sum().clamp_min(1e-6)
    indices = torch.multinomial(flat, num_samples=num_samples, replacement=True, generator=generator)
    y = torch.div(indices, width, rounding_mode="floor")
    x = indices % width
    centers_x = (x.to(dtype=distribution.dtype) + 0.5) / float(width)
    centers_y = (y.to(dtype=distribution.dtype) + 0.5) / float(height)
    return torch.stack([centers_x, centers_y], dim=1)


def _blur_map(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size <= 1:
        return image
    padding = kernel_size // 2
    padded = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)


def _sobel_gradients(grayscale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    device = grayscale.device
    dtype = grayscale.dtype
    kernel_x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=dtype, device=device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], dtype=dtype, device=device).view(1, 1, 3, 3)
    gx = F.conv2d(grayscale, kernel_x, padding=1)
    gy = F.conv2d(grayscale, kernel_y, padding=1)
    return gx, gy


def analyze_target_structure(target: torch.Tensor) -> TargetAnalysis:
    grayscale = (0.299 * target[:, 0:1]) + (0.587 * target[:, 1:2]) + (0.114 * target[:, 2:3])
    gx, gy = _sobel_gradients(grayscale)
    edges = torch.sqrt(gx**2 + gy**2 + 1e-6)
    jxx = _blur_map(gx * gx, kernel_size=5)
    jyy = _blur_map(gy * gy, kernel_size=5)
    jxy = _blur_map(gx * gy, kernel_size=5)
    coherence = torch.sqrt((jxx - jyy) ** 2 + (4.0 * jxy**2)) / (jxx + jyy + 1e-6)
    coherence = coherence.clamp(0.0, 1.0)
    orientation = 0.5 * torch.atan2(2.0 * jxy, (jxx - jyy))
    tangent_theta = orientation + (math.pi / 2.0)
    return TargetAnalysis(edges=edges, coherence=coherence, tangent_theta=tangent_theta)


def _normalized_map(value: torch.Tensor) -> torch.Tensor:
    return value / value.max().clamp_min(1e-6)


def build_residual_components(
    target: torch.Tensor,
    current_image: torch.Tensor,
    coverage_map: torch.Tensor,
    target_analysis: TargetAnalysis,
    coverage_penalty_strength: float,
) -> Dict[str, torch.Tensor]:
    residual = torch.mean(torch.abs(target - current_image), dim=1, keepdim=True)
    low = _blur_map(residual, kernel_size=max(5, ((min(target.shape[-2:]) // 6) | 1)))
    high = (residual - low).clamp_min(0.0)
    coverage_penalty = 1.0 / (1.0 + (coverage_penalty_strength * coverage_map))
    base_map = (low * coverage_penalty).clamp_min(1e-6)
    texture_map = ((high + (0.4 * target_analysis.edges)) * coverage_penalty).clamp_min(1e-6)
    accent_map = ((target_analysis.edges + (0.5 * high)) * coverage_penalty).clamp_min(1e-6)
    return {
        "residual": residual,
        "low": low,
        "high": high,
        "base_distribution": _normalized_map(base_map).squeeze(0).squeeze(0),
        "texture_distribution": _normalized_map(texture_map).squeeze(0).squeeze(0),
        "accent_distribution": _normalized_map(accent_map).squeeze(0).squeeze(0),
        "base_residual_map": base_map,
        "texture_residual_map": texture_map,
        "accent_residual_map": accent_map,
    }


def _theta_from_analysis(
    target_analysis: TargetAnalysis,
    center_x: int,
    center_y: int,
    generator: torch.Generator,
    jitter_scale: float,
) -> float:
    theta = float(target_analysis.tangent_theta[0, 0, center_y, center_x].item())
    jitter = torch.empty((1,), device=target_analysis.tangent_theta.device, dtype=target_analysis.tangent_theta.dtype).uniform_(-jitter_scale, jitter_scale, generator=generator)
    return theta + float(jitter.item())


def _base_sizes_from_patch(
    residual_map: torch.Tensor,
    center_x: int,
    center_y: int,
    width: int,
    height: int,
    generator: torch.Generator,
) -> Tuple[float, float]:
    min_size, max_size = 0.08, 0.34
    half_window = max(10, min(width, height) // 8)
    x0 = max(0, center_x - half_window)
    x1 = min(width, center_x + half_window + 1)
    y0 = max(0, center_y - half_window)
    y1 = min(height, center_y + half_window + 1)
    patch = residual_map.squeeze(0).squeeze(0)[y0:y1, x0:x1]
    if float(patch.sum().item()) <= 1e-6:
        base = torch.empty((1,), device=residual_map.device, dtype=residual_map.dtype).uniform_(min_size, max_size, generator=generator).item()
        tri_height = torch.empty((1,), device=residual_map.device, dtype=residual_map.dtype).uniform_(min_size, max_size, generator=generator).item()
        return base, tri_height
    ys, xs = torch.meshgrid(
        torch.arange(y0, y1, device=residual_map.device, dtype=residual_map.dtype),
        torch.arange(x0, x1, device=residual_map.device, dtype=residual_map.dtype),
        indexing="ij",
    )
    weights = patch / patch.sum().clamp_min(1e-6)
    cx = torch.tensor(float(center_x), device=residual_map.device, dtype=residual_map.dtype)
    cy = torch.tensor(float(center_y), device=residual_map.device, dtype=residual_map.dtype)
    var_x = torch.sum(weights * (xs - cx) ** 2)
    var_y = torch.sum(weights * (ys - cy) ** 2)
    tri_base = float(2.5 * torch.sqrt(var_x + 1.0).item()) / float(width)
    tri_height = float(2.5 * torch.sqrt(var_y + 1.0).item()) / float(height)
    tri_base = max(min_size, min(max_size, tri_base))
    tri_height = max(min_size, min(max_size, tri_height))
    return tri_base, tri_height


def _role_shape_from_analysis(
    kind: str,
    residual_map: torch.Tensor,
    target_analysis: TargetAnalysis,
    center_x: int,
    center_y: int,
    width: int,
    height: int,
    generator: torch.Generator,
) -> Tuple[float, float, float]:
    if kind == "base":
        tri_base, tri_height = _base_sizes_from_patch(residual_map, center_x, center_y, width, height, generator)
        theta = float(torch.empty((1,), device=residual_map.device, dtype=residual_map.dtype).uniform_(0.0, 2.0 * math.pi, generator=generator).item())
        return tri_base, tri_height, theta

    local_residual = float(residual_map[0, 0, center_y, center_x].item())
    residual_scale = local_residual / float(residual_map.max().item() + 1e-6)
    coherence = float(target_analysis.coherence[0, 0, center_y, center_x].item())

    if kind == "texture":
        base_min, base_max = 0.03, 0.12
        height_min, height_max = 0.025, 0.10
        base_jitter = float(torch.empty((1,), device=residual_map.device, dtype=residual_map.dtype).uniform_(0.90, 1.10, generator=generator).item())
        height_jitter = float(torch.empty((1,), device=residual_map.device, dtype=residual_map.dtype).uniform_(0.85, 1.15, generator=generator).item())
        tri_base = (base_min + ((base_max - base_min) * (0.30 + (0.70 * residual_scale)))) * base_jitter
        tri_height = (height_min + ((height_max - height_min) * (0.20 + (0.80 * coherence)))) * height_jitter
        theta = _theta_from_analysis(target_analysis, center_x, center_y, generator, jitter_scale=0.45)
        return tri_base, tri_height, theta

    base_min, base_max = 0.02, 0.08
    height_min, height_max = 0.03, 0.14
    base_jitter = float(torch.empty((1,), device=residual_map.device, dtype=residual_map.dtype).uniform_(0.85, 1.10, generator=generator).item())
    height_jitter = float(torch.empty((1,), device=residual_map.device, dtype=residual_map.dtype).uniform_(0.95, 1.15, generator=generator).item())
    tri_base = (base_min + ((base_max - base_min) * (0.20 + (0.80 * residual_scale)))) * base_jitter
    tri_height = (height_min + ((height_max - height_min) * (0.25 + (0.75 * coherence)))) * height_jitter
    theta = _theta_from_analysis(target_analysis, center_x, center_y, generator, jitter_scale=0.20)
    return tri_base, tri_height, theta


def _build_rgb_from_centers(target: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    _, _, height, width = target.shape
    rgb_values = []
    for index in range(centers.shape[0]):
        center_x = int(torch.clamp(torch.round(centers[index, 0] * (width - 1)), 0, width - 1).item())
        center_y = int(torch.clamp(torch.round(centers[index, 1] * (height - 1)), 0, height - 1).item())
        rgb_values.append(mean_window_color(target, center_x=center_x, center_y=center_y))
    if not rgb_values:
        return torch.empty((0, 3), device=target.device, dtype=target.dtype)
    return torch.stack(rgb_values, dim=0)


def _role_parameters_from_centers(
    kind: str,
    target: torch.Tensor,
    centers: torch.Tensor,
    residual_map: torch.Tensor,
    target_analysis: TargetAnalysis,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, height, width = target.shape
    generator = torch.Generator(device=target.device)
    generator.manual_seed(seed)
    theta_values = []
    size_values = []
    for index in range(centers.shape[0]):
        center_x = int(torch.clamp(torch.round(centers[index, 0] * (width - 1)), 0, width - 1).item())
        center_y = int(torch.clamp(torch.round(centers[index, 1] * (height - 1)), 0, height - 1).item())
        tri_base, tri_height, theta = _role_shape_from_analysis(kind, residual_map, target_analysis, center_x, center_y, width, height, generator)
        size_values.append([tri_base, tri_height])
        theta_values.append([theta])
    sizes = torch.tensor(size_values, device=target.device, dtype=target.dtype) if size_values else torch.empty((0, 2), device=target.device, dtype=target.dtype)
    theta = torch.tensor(theta_values, device=target.device, dtype=target.dtype) if theta_values else torch.empty((0, 1), device=target.device, dtype=target.dtype)
    rgb = _build_rgb_from_centers(target, centers)
    return sizes, theta, rgb


def initialize_ellipse_table(target: torch.Tensor, config: FitConfig) -> Tuple[EllipseParameterTable, TargetAnalysis]:
    _, _, height, width = target.shape
    background_grid_rgb = initialize_background_grid(target, config.background_grid_size).to(device=target.device, dtype=target.dtype)
    background_image = F.interpolate(background_grid_rgb, size=(height, width), mode="nearest")
    target_analysis = analyze_target_structure(target)
    zero_coverage = torch.zeros((1, 1, height, width), device=target.device, dtype=target.dtype)
    residuals = build_residual_components(
        target=target,
        current_image=background_image,
        coverage_map=zero_coverage,
        target_analysis=target_analysis,
        coverage_penalty_strength=config.coverage_penalty_strength,
    )

    base_count = config.base_ellipse_count()
    texture_count = config.texture_ellipse_count()
    accent_count = config.accent_ellipse_count()

    base_centers = _sample_from_distribution(residuals["base_distribution"], base_count, width, height, config.seed + 101).to(device=target.device, dtype=target.dtype)
    texture_centers = _sample_from_distribution(residuals["texture_distribution"], texture_count, width, height, config.seed + 211).to(device=target.device, dtype=target.dtype)
    accent_centers = _sample_from_distribution(residuals["accent_distribution"], accent_count, width, height, config.seed + 307).to(device=target.device, dtype=target.dtype)
    centers = torch.cat([base_centers, texture_centers, accent_centers], dim=0)

    base_sizes, base_theta, base_rgb = _role_parameters_from_centers("base", target, base_centers, residuals["base_residual_map"], target_analysis, config.seed + 401)
    texture_sizes, texture_theta, texture_rgb = _role_parameters_from_centers("texture", target, texture_centers, residuals["texture_residual_map"], target_analysis, config.seed + 503)
    accent_sizes, accent_theta, accent_rgb = _role_parameters_from_centers("accent", target, accent_centers, residuals["accent_residual_map"], target_analysis, config.seed + 607)
    sizes = torch.cat([base_sizes, texture_sizes, accent_sizes], dim=0)
    theta = torch.cat([base_theta, texture_theta, accent_theta], dim=0)
    rgb = torch.cat([base_rgb, texture_rgb, accent_rgb], dim=0)

    table = EllipseParameterTable.from_decoded(
        centers=centers,
        sizes=sizes,
        theta=theta,
        rgb=rgb,
        background_grid_rgb=background_grid_rgb,
        base_count=base_count,
        texture_count=texture_count,
    )
    return table, target_analysis


def reinitialize_table_slice_from_residual(
    table: EllipseParameterTable,
    start: int,
    end: int,
    target: torch.Tensor,
    distribution: torch.Tensor,
    residual_map: torch.Tensor,
    target_analysis: TargetAnalysis,
    seed: int,
    kind: str,
) -> None:
    if end <= start:
        return
    _, _, height, width = target.shape
    count = end - start
    device = target.device
    dtype = target.dtype
    centers = _sample_from_distribution(distribution, count, width, height, seed).to(device=device, dtype=dtype)
    sizes, theta, rgb = _role_parameters_from_centers(
        kind=kind,
        target=target,
        centers=centers,
        residual_map=residual_map,
        target_analysis=target_analysis,
        seed=seed + 701,
    )
    table.set_decoded_slice(start=start, end=end, centers=centers, sizes=sizes, theta=theta, rgb=rgb)


