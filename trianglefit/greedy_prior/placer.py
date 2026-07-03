from __future__ import annotations

from dataclasses import dataclass
import math
import random
from pathlib import Path
from typing import Callable, List, Protocol, Sequence, Tuple

import torch

from ..direct.fit_geometrize_json import GEOMETRIZE_ISOSCELES_TRIANGLE, GEOMETRIZE_RECTANGLE
from ..direct.utils import serialize_json
from . import cuda_scoring


ProgressCallback = Callable[[int, int, "GreedyPlacementResult"], None]


@dataclass(frozen=True)
class ShapeBounds:
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 1.0
    y_max: float = 1.0

    @classmethod
    def from_values(cls, values: Sequence[float] | None) -> "ShapeBounds":
        if values is None:
            return cls()
        if len(values) != 4:
            raise ValueError("shape_bounds must contain four values: x_min y_min x_max y_max.")
        numbers = [float(value) for value in values]
        if any(abs(value) > 1.0 for value in numbers):
            numbers = [value / 100.0 for value in numbers]
        x_min, y_min, x_max, y_max = numbers
        x_min = max(0.0, min(1.0, x_min))
        y_min = max(0.0, min(1.0, y_min))
        x_max = max(0.0, min(1.0, x_max))
        y_max = max(0.0, min(1.0, y_max))
        if x_max <= x_min or y_max <= y_min:
            return cls()
        return cls(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    def center_limits_px(self, width: int, height: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.x_min * float(width)
        y0 = self.y_min * float(height)
        x1 = self.x_max * float(width)
        y1 = self.y_max * float(height)
        min_xy = torch.tensor([x0 + 0.5, y0 + 0.5], device=device, dtype=dtype)
        max_xy = torch.tensor([max(x0 + 0.5, x1 - 0.5), max(y0 + 0.5, y1 - 0.5)], device=device, dtype=dtype)
        return min_xy, max_xy

    def center_limits_px_values(self, width: int, height: int) -> Tuple[float, float, float, float]:
        x0 = self.x_min * float(width)
        y0 = self.y_min * float(height)
        x1 = self.x_max * float(width)
        y1 = self.y_max * float(height)
        return (
            x0 + 0.5,
            y0 + 0.5,
            max(x0 + 0.5, x1 - 0.5),
            max(y0 + 0.5, y1 - 0.5),
        )

    def to_list(self) -> List[float]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]


@dataclass(frozen=True)
class TrianglePlacementConfig:
    num_triangles: int = 300
    candidate_count: int = 2048
    max_shape_mutations: int = 2000
    candidate_chunk_size: int = 256
    seed: int = -1
    shape_bounds: ShapeBounds = ShapeBounds()
    min_half_base_fraction: float = 1.0 / 256.0
    max_half_base_fraction: float = 32.0 / 256.0
    min_height_fraction: float = 1.0 / 256.0
    max_height_fraction: float = 64.0 / 256.0
    center_mutation_fraction: float = 32.0 / 256.0
    size_mutation_fraction: float = 16.0 / 256.0
    angle_mutation_degrees: float = 16.0
    min_improvement: float = 1e-9
    background_rgb: Tuple[float, float, float] | None = None

    def validate(self) -> None:
        if self.num_triangles < 0:
            raise ValueError("num_triangles must be non-negative.")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive.")
        if self.max_shape_mutations < 0:
            raise ValueError("max_shape_mutations must be non-negative.")
        if self.candidate_chunk_size <= 0:
            raise ValueError("candidate_chunk_size must be positive.")
        if self.max_half_base_fraction < self.min_half_base_fraction:
            raise ValueError("max_half_base_fraction must be >= min_half_base_fraction.")
        if self.max_height_fraction < self.min_height_fraction:
            raise ValueError("max_height_fraction must be >= min_height_fraction.")
        if self.background_rgb is not None:
            normalize_rgb(self.background_rgb)


def normalize_rgb(values: Sequence[float] | None) -> Tuple[float, float, float] | None:
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError("background_rgb must contain three values: red green blue.")
    rgb = [float(value) for value in values]
    if any(abs(value) > 1.0 for value in rgb):
        rgb = [value / 255.0 for value in rgb]
    return tuple(max(0.0, min(1.0, value)) for value in rgb)


@dataclass(frozen=True)
class PlacedTriangle:
    cx: float
    cy: float
    half_base: float
    height: float
    theta: float
    rgb: Tuple[float, float, float]
    score_sse: float
    improvement_sse: float

    def scaled(self, scale_x: float, scale_y: float) -> "PlacedTriangle":
        return PlacedTriangle(
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            half_base=self.half_base * scale_x,
            height=self.height * scale_y,
            theta=self.theta,
            rgb=self.rgb,
            score_sse=self.score_sse,
            improvement_sse=self.improvement_sse,
        )


@dataclass(frozen=True)
class GreedyPlacementResult:
    image: torch.Tensor
    background_rgb: Tuple[float, float, float]
    triangles: List[PlacedTriangle]
    history: List[dict]
    seed: int
    initial_sse: float
    final_sse: float
    width: int
    height: int

    @property
    def triangle_count(self) -> int:
        return len(self.triangles)

    @property
    def final_mse(self) -> float:
        return self.final_sse / float(max(1, self.width * self.height * 3))

    @property
    def final_rmse(self) -> float:
        return math.sqrt(max(self.final_mse, 0.0))


@dataclass(frozen=True)
class CandidateScores:
    scores: torch.Tensor
    colors: torch.Tensor
    counts: torch.Tensor


@dataclass
class TriangleBatch:
    centers: torch.Tensor
    half_base: torch.Tensor
    height: torch.Tensor
    theta: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.centers.shape[0])

    def clone(self) -> "TriangleBatch":
        return TriangleBatch(
            centers=self.centers.clone(),
            half_base=self.half_base.clone(),
            height=self.height.clone(),
            theta=self.theta.clone(),
        )

    def index_select(self, indices: torch.Tensor) -> "TriangleBatch":
        return TriangleBatch(
            centers=self.centers.index_select(0, indices),
            half_base=self.half_base.index_select(0, indices),
            height=self.height.index_select(0, indices),
            theta=self.theta.index_select(0, indices),
        )

    def where(self, mask: torch.Tensor, other: "TriangleBatch") -> "TriangleBatch":
        mask_2d = mask.view(-1, 1)
        return TriangleBatch(
            centers=torch.where(mask_2d, other.centers, self.centers),
            half_base=torch.where(mask_2d, other.half_base, self.half_base),
            height=torch.where(mask_2d, other.height, self.height),
            theta=torch.where(mask_2d, other.theta, self.theta),
        )


class TrianglePlacer(Protocol):
    def fit(
        self,
        target: torch.Tensor,
        config: TrianglePlacementConfig,
        initial_image: torch.Tensor | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> GreedyPlacementResult:
        ...


def resolve_seed(seed: int) -> int:
    if int(seed) >= 0:
        return int(seed)
    return random.SystemRandom().randrange(0, 2**31)


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(resolve_seed(seed))
    return generator


def _pixel_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(height, device=device, dtype=dtype) + 0.5
    x = torch.arange(width, device=device, dtype=dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return grid_x, grid_y


def _triangle_masks_px(batch: TriangleBatch, grid_x: torch.Tensor, grid_y: torch.Tensor) -> torch.Tensor:
    centers_x = batch.centers[:, 0].view(-1, 1, 1)
    centers_y = batch.centers[:, 1].view(-1, 1, 1)
    half_base = batch.half_base.view(-1, 1, 1).clamp_min(1e-6)
    tri_height = batch.height.view(-1, 1, 1).clamp_min(1e-6)
    theta = batch.theta.view(-1, 1, 1)

    dx = grid_x.unsqueeze(0) - centers_x
    dy = grid_y.unsqueeze(0) - centers_y
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    x_rot = cos_theta * dx + sin_theta * dy
    y_rot = -sin_theta * dx + cos_theta * dy

    y_from_top = y_rot + (tri_height * 0.5)
    half_base_at_y = y_from_top * (half_base / tri_height)
    return (
        (y_from_top >= 0.0)
        & (y_rot <= (tri_height * 0.5))
        & (x_rot >= -half_base_at_y)
        & (x_rot <= half_base_at_y)
    ).to(dtype=batch.centers.dtype)


def _image_sse(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    return torch.sum((target - current) ** 2)


def _rmse_from_sse(sse: float, width: int, height: int) -> float:
    return math.sqrt(max(sse, 0.0) / float(max(1, width * height * 3)))


@torch.no_grad()
def render_triangles(
    triangles: Sequence[PlacedTriangle],
    background_rgb: Tuple[float, float, float],
    width: int,
    height: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    render_device = torch.device(device)
    dtype = torch.float32
    image = torch.tensor(background_rgb, device=render_device, dtype=dtype).view(1, 3, 1, 1).expand(1, 3, height, width).clone()
    grid_x, grid_y = _pixel_grid(height=height, width=width, device=render_device, dtype=dtype)
    for triangle in triangles:
        batch = TriangleBatch(
            centers=torch.tensor([[triangle.cx, triangle.cy]], device=render_device, dtype=dtype),
            half_base=torch.tensor([[triangle.half_base]], device=render_device, dtype=dtype),
            height=torch.tensor([[triangle.height]], device=render_device, dtype=dtype),
            theta=torch.tensor([[triangle.theta]], device=render_device, dtype=dtype),
        )
        mask = _triangle_masks_px(batch, grid_x=grid_x, grid_y=grid_y).view(1, 1, height, width)
        color = torch.tensor(triangle.rgb, device=render_device, dtype=dtype).view(1, 3, 1, 1)
        image = image * (1.0 - mask) + color * mask
    return image.clamp(0.0, 1.0)


def export_geometrize_json(
    triangles: Sequence[PlacedTriangle],
    background_rgb: Tuple[float, float, float],
    path: Path,
    width: int,
    height: int,
) -> None:
    shapes: List[dict] = [
        {
            "type": GEOMETRIZE_RECTANGLE,
            "data": [0.0, 0.0, float(width), float(height)],
            "color": [int(round(max(0.0, min(1.0, value)) * 255.0)) for value in background_rgb] + [255],
            "score": 0.0,
        }
    ]
    for triangle in triangles:
        shapes.append(
            {
                "type": GEOMETRIZE_ISOSCELES_TRIANGLE,
                "data": [
                    float(triangle.cx),
                    float(triangle.cy),
                    float(max(triangle.half_base, 1e-6)),
                    float(max(triangle.height, 1e-6)),
                    float(math.degrees(triangle.theta) % 360.0),
                ],
                "color": [int(round(max(0.0, min(1.0, value)) * 255.0)) for value in triangle.rgb] + [255],
                "score": float(triangle.score_sse),
            }
        )
    serialize_json({"shapes": shapes}, path)


class HillClimbTrianglePlacer:
    def fit(
        self,
        target: torch.Tensor,
        config: TrianglePlacementConfig,
        initial_image: torch.Tensor | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> GreedyPlacementResult:
        config.validate()
        if target.ndim != 4 or target.shape[0] != 1 or target.shape[1] != 3:
            raise ValueError("target must have shape [1, 3, H, W].")

        target = target.detach().to(dtype=torch.float32)
        device = target.device
        dtype = target.dtype
        _, _, height, width = target.shape
        if device.type != "cuda":
            raise ValueError("HillClimbTrianglePlacer now requires a CUDA target. Use --device cuda or --device auto on a CUDA machine.")
        seed = resolve_seed(config.seed)

        if initial_image is None:
            configured_background = normalize_rgb(config.background_rgb)
            if configured_background is None:
                background = target.mean(dim=(0, 2, 3)).clamp(0.0, 1.0)
            else:
                background = torch.tensor(configured_background, device=device, dtype=dtype)
            current = background.view(1, 3, 1, 1).expand_as(target).clone()
        else:
            current = initial_image.detach().to(device=device, dtype=dtype).clone()
            if current.shape != target.shape:
                raise ValueError("initial_image must have the same shape as target.")
            background = current[:, :, 0, 0].view(3).clamp(0.0, 1.0)

        background_rgb = tuple(float(value) for value in background.detach().cpu())
        target_chw = target[0]
        current_chw = current[0].contiguous()
        triangles: List[PlacedTriangle] = []
        history: List[dict] = []
        current_sse = float(_image_sse(target, current).detach().cpu().item())
        initial_sse = current_sse

        bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y = config.shape_bounds.center_limits_px_values(width=width, height=height)
        min_half_base = max(config.min_half_base_fraction * float(width), 1e-3)
        max_half_base = max(min_half_base, config.max_half_base_fraction * float(width))
        min_height = max(config.min_height_fraction * float(height), 1e-3)
        max_height = max(min_height, config.max_height_fraction * float(height))
        center_step_x = config.center_mutation_fraction * float(width)
        center_step_y = config.center_mutation_fraction * float(height)
        half_base_step = config.size_mutation_fraction * float(width)
        height_step = config.size_mutation_fraction * float(height)
        angle_step = math.radians(config.angle_mutation_degrees)

        for index in range(config.num_triangles):
            params, color, score_tensor = cuda_scoring.search_and_apply(
                target_chw=target_chw,
                current_chw=current_chw,
                current_sse=current_sse,
                candidate_count=config.candidate_count,
                mutation_count=config.max_shape_mutations,
                bounds_min_x=bounds_min_x,
                bounds_min_y=bounds_min_y,
                bounds_max_x=bounds_max_x,
                bounds_max_y=bounds_max_y,
                min_half_base=min_half_base,
                max_half_base=max_half_base,
                min_height=min_height,
                max_height=max_height,
                center_step_x=center_step_x,
                center_step_y=center_step_y,
                half_base_step=half_base_step,
                height_step=height_step,
                angle_step=angle_step,
                seed=seed,
                round_index=index,
            )
            score = float(score_tensor.detach().cpu().item())
            improvement = current_sse - score
            if not math.isfinite(score) or improvement <= config.min_improvement:
                break

            previous_sse = current_sse
            current_sse = score
            params_cpu = params.detach().cpu()
            color_cpu = color.detach().cpu().view(3)
            triangle = PlacedTriangle(
                cx=float(params_cpu[0].item()),
                cy=float(params_cpu[1].item()),
                half_base=float(params_cpu[2].item()),
                height=float(params_cpu[3].item()),
                theta=float(params_cpu[4].item()),
                rgb=tuple(float(value) for value in color_cpu),
                score_sse=current_sse,
                improvement_sse=previous_sse - current_sse,
            )
            triangles.append(triangle)
            history.append(
                {
                    "triangle": len(triangles),
                    "sse": triangle.score_sse,
                    "mse": triangle.score_sse / float(max(1, width * height * 3)),
                    "rmse": _rmse_from_sse(triangle.score_sse, width=width, height=height),
                    "improvement_sse": triangle.improvement_sse,
                    "cx": triangle.cx,
                    "cy": triangle.cy,
                    "half_base": triangle.half_base,
                    "height": triangle.height,
                    "theta_degrees": math.degrees(triangle.theta) % 360.0,
                }
            )

            if progress_callback is not None:
                partial = GreedyPlacementResult(
                    image=current.detach().clone(),
                    background_rgb=background_rgb,
                    triangles=list(triangles),
                    history=list(history),
                    seed=seed,
                    initial_sse=initial_sse,
                    final_sse=current_sse,
                    width=width,
                    height=height,
                )
                progress_callback(index + 1, config.num_triangles, partial)

        final_sse = current_sse
        return GreedyPlacementResult(
            image=current_chw.view(1, 3, height, width).detach(),
            background_rgb=background_rgb,
            triangles=triangles,
            history=history,
            seed=seed,
            initial_sse=initial_sse,
            final_sse=final_sse,
            width=width,
            height=height,
        )

    def _search_one_triangle(
        self,
        target_chw: torch.Tensor,
        target_sq_chw: torch.Tensor,
        current: torch.Tensor,
        current_sse: torch.Tensor,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
        width: int,
        height: int,
        config: TrianglePlacementConfig,
        generator: torch.Generator,
    ) -> Tuple[TriangleBatch, torch.Tensor, torch.Tensor]:
        best_states = self._random_candidates(
            count=int(config.candidate_count),
            width=width,
            height=height,
            config=config,
            generator=generator,
            device=target_chw.device,
            dtype=target_chw.dtype,
        )
        initial_scores = self._score_candidates(
            candidates=best_states,
            target_chw=target_chw,
            target_sq_chw=target_sq_chw,
            current=current,
            current_sse=current_sse,
            grid_x=grid_x,
            grid_y=grid_y,
            chunk_size=config.candidate_chunk_size,
        )
        best_colors = initial_scores.colors
        best_scores = initial_scores.scores
        for _ in range(config.max_shape_mutations):
            mutated = self._mutate(best_states, width=width, height=height, config=config, generator=generator)
            mutated_scores = self._score_candidates(
                candidates=mutated,
                target_chw=target_chw,
                target_sq_chw=target_sq_chw,
                current=current,
                current_sse=current_sse,
                grid_x=grid_x,
                grid_y=grid_y,
                chunk_size=config.candidate_chunk_size,
            )
            improved = mutated_scores.scores < best_scores
            best_states = best_states.where(improved, mutated)
            best_colors = torch.where(improved.view(-1, 1), mutated_scores.colors, best_colors)
            best_scores = torch.where(improved, mutated_scores.scores, best_scores)

        best_index = torch.argmin(best_scores).view(1)
        return best_states.index_select(best_index), best_colors.index_select(0, best_index)[0], best_scores.index_select(0, best_index)[0]

    def _random_candidates(
        self,
        count: int,
        width: int,
        height: int,
        config: TrianglePlacementConfig,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TriangleBatch:
        bounds_min, bounds_max = config.shape_bounds.center_limits_px(width=width, height=height, device=device, dtype=dtype)
        centers_unit = torch.rand((count, 2), device=device, dtype=dtype, generator=generator)
        centers = bounds_min.view(1, 2) + centers_unit * (bounds_max - bounds_min).view(1, 2)

        min_half_base = max(config.min_half_base_fraction * float(width), 1e-3)
        max_half_base = max(min_half_base, config.max_half_base_fraction * float(width))
        min_height = max(config.min_height_fraction * float(height), 1e-3)
        max_height = max(min_height, config.max_height_fraction * float(height))
        half_base = torch.rand((count, 1), device=device, dtype=dtype, generator=generator) * (max_half_base - min_half_base) + min_half_base
        tri_height = torch.rand((count, 1), device=device, dtype=dtype, generator=generator) * (max_height - min_height) + min_height
        theta = torch.rand((count, 1), device=device, dtype=dtype, generator=generator) * (2.0 * math.pi)
        return TriangleBatch(centers=centers, half_base=half_base, height=tri_height, theta=theta)

    def _mutate(
        self,
        batch: TriangleBatch,
        width: int,
        height: int,
        config: TrianglePlacementConfig,
        generator: torch.Generator,
    ) -> TriangleBatch:
        device = batch.centers.device
        dtype = batch.centers.dtype
        count = batch.count
        mutated = batch.clone()
        choices = torch.randint(0, 4, (count,), device=device, generator=generator)

        center_step = torch.tensor(
            [config.center_mutation_fraction * float(width), config.center_mutation_fraction * float(height)],
            device=device,
            dtype=dtype,
        )
        center_delta = (torch.rand((count, 2), device=device, dtype=dtype, generator=generator) * 2.0 - 1.0) * center_step.view(1, 2)
        center_mask = (choices == 0).view(-1, 1)
        mutated.centers = torch.where(center_mask, mutated.centers + center_delta, mutated.centers)
        bounds_min, bounds_max = config.shape_bounds.center_limits_px(width=width, height=height, device=device, dtype=dtype)
        mutated.centers = torch.max(torch.min(mutated.centers, bounds_max.view(1, 2)), bounds_min.view(1, 2))

        half_base_step = config.size_mutation_fraction * float(width)
        height_step = config.size_mutation_fraction * float(height)
        half_base_delta = (torch.rand((count, 1), device=device, dtype=dtype, generator=generator) * 2.0 - 1.0) * half_base_step
        height_delta = (torch.rand((count, 1), device=device, dtype=dtype, generator=generator) * 2.0 - 1.0) * height_step
        half_mask = (choices == 1).view(-1, 1)
        height_mask = (choices == 2).view(-1, 1)
        mutated.half_base = torch.where(half_mask, mutated.half_base + half_base_delta, mutated.half_base)
        mutated.height = torch.where(height_mask, mutated.height + height_delta, mutated.height)

        min_half_base = max(config.min_half_base_fraction * float(width), 1e-3)
        max_half_base = max(min_half_base, config.max_half_base_fraction * float(width))
        min_height = max(config.min_height_fraction * float(height), 1e-3)
        max_height = max(min_height, config.max_height_fraction * float(height))
        mutated.half_base = mutated.half_base.clamp(min=min_half_base, max=max_half_base)
        mutated.height = mutated.height.clamp(min=min_height, max=max_height)

        angle_step = math.radians(config.angle_mutation_degrees)
        angle_delta = (torch.rand((count, 1), device=device, dtype=dtype, generator=generator) * 2.0 - 1.0) * angle_step
        angle_mask = (choices == 3).view(-1, 1)
        mutated.theta = torch.where(angle_mask, mutated.theta + angle_delta, mutated.theta)
        mutated.theta = torch.remainder(mutated.theta, 2.0 * math.pi)
        return mutated

    def _score_candidates(
        self,
        candidates: TriangleBatch,
        target_chw: torch.Tensor,
        target_sq_chw: torch.Tensor,
        current: torch.Tensor,
        current_sse: torch.Tensor,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
        chunk_size: int,
    ) -> CandidateScores:
        if target_chw.is_cuda:
            try:
                return self._score_candidates_cuda(
                    candidates=candidates,
                    target_chw=target_chw,
                    current=current,
                    current_sse=current_sse,
                    chunk_size=chunk_size,
                )
            except Exception:
                pass
        return self._score_candidates_torch(
            candidates=candidates,
            target_chw=target_chw,
            target_sq_chw=target_sq_chw,
            current=current,
            current_sse=current_sse,
            grid_x=grid_x,
            grid_y=grid_y,
            chunk_size=chunk_size,
        )

    def _score_candidates_cuda(
        self,
        candidates: TriangleBatch,
        target_chw: torch.Tensor,
        current: torch.Tensor,
        current_sse: torch.Tensor,
        chunk_size: int,
    ) -> CandidateScores:
        scores: List[torch.Tensor] = []
        colors: List[torch.Tensor] = []
        counts: List[torch.Tensor] = []
        for start in range(0, candidates.count, int(chunk_size)):
            end = min(candidates.count, start + int(chunk_size))
            chunk = TriangleBatch(
                centers=candidates.centers[start:end],
                half_base=candidates.half_base[start:end],
                height=candidates.height[start:end],
                theta=candidates.theta[start:end],
            )
            chunk_scores, chunk_colors, chunk_counts = cuda_scoring.score_triangles(
                target_chw=target_chw,
                current_chw=current,
                centers=chunk.centers,
                half_base=chunk.half_base,
                height=chunk.height,
                theta=chunk.theta,
                current_sse=current_sse,
            )
            scores.append(chunk_scores)
            colors.append(chunk_colors)
            counts.append(chunk_counts)
        return CandidateScores(scores=torch.cat(scores, dim=0), colors=torch.cat(colors, dim=0), counts=torch.cat(counts, dim=0))

    def _score_candidates_torch(
        self,
        candidates: TriangleBatch,
        target_chw: torch.Tensor,
        target_sq_chw: torch.Tensor,
        current: torch.Tensor,
        current_sse: torch.Tensor,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
        chunk_size: int,
    ) -> CandidateScores:
        old_error = (target_chw - current).square().sum(dim=0)
        scores: List[torch.Tensor] = []
        colors: List[torch.Tensor] = []
        counts: List[torch.Tensor] = []
        for start in range(0, candidates.count, int(chunk_size)):
            end = min(candidates.count, start + int(chunk_size))
            chunk = TriangleBatch(
                centers=candidates.centers[start:end],
                half_base=candidates.half_base[start:end],
                height=candidates.height[start:end],
                theta=candidates.theta[start:end],
            )
            mask = _triangle_masks_px(chunk, grid_x=grid_x, grid_y=grid_y)
            count = mask.sum(dim=(1, 2))
            safe_count = count.clamp_min(1.0)
            target_sum = torch.einsum("bhw,chw->bc", mask, target_chw)
            target_sq_sum = torch.einsum("bhw,chw->bc", mask, target_sq_chw)
            color = (target_sum / safe_count.view(-1, 1)).clamp(0.0, 1.0)
            old_sse_inside = torch.einsum("bhw,hw->b", mask, old_error)
            new_sse_inside = target_sq_sum.sum(dim=1)
            new_sse_inside = new_sse_inside - 2.0 * (color * target_sum).sum(dim=1)
            new_sse_inside = new_sse_inside + color.square().sum(dim=1) * count
            score = current_sse - old_sse_inside + new_sse_inside
            score = torch.where(count > 0.0, score, torch.full_like(score, float("inf")))
            scores.append(score)
            colors.append(color)
            counts.append(count)
        return CandidateScores(scores=torch.cat(scores, dim=0), colors=torch.cat(colors, dim=0), counts=torch.cat(counts, dim=0))
