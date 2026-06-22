from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .io import load_image, save_image
from .utils import ensure_dir, inverse_sigmoid, inverse_softplus, serialize_json, tensor_to_uint8_image

GEOMETRIZE_RECTANGLE = 1
GEOMETRIZE_ISOSCELES_TRIANGLE = 512


@dataclass(frozen=True)
class GeometrizeShape:
    shape_type: int
    data: List[float]
    color: List[float]
    score: float | None = None


@dataclass(frozen=True)
class GeometrizePayload:
    shapes: List[GeometrizeShape]
    canvas_width: float
    canvas_height: float
    background_rgb: Tuple[float, float, float]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _read_geometrize_json(path: Path) -> GeometrizePayload:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_shapes = payload.get("shapes")
    if not isinstance(raw_shapes, list):
        raise ValueError("Expected a Geometrize JSON object with a 'shapes' list.")

    shapes: List[GeometrizeShape] = []
    for item in raw_shapes:
        shape_type = int(item["type"])
        data = [float(value) for value in item.get("data", [])]
        color = [float(value) for value in item.get("color", [255.0, 255.0, 255.0, 255.0])]
        score = None if "score" not in item else float(item["score"])
        shapes.append(GeometrizeShape(shape_type=shape_type, data=data, color=color, score=score))

    rectangle = next((shape for shape in shapes if shape.shape_type == GEOMETRIZE_RECTANGLE and len(shape.data) >= 4), None)
    if rectangle is not None:
        canvas_width = max(float(rectangle.data[0]), float(rectangle.data[2]))
        canvas_height = max(float(rectangle.data[1]), float(rectangle.data[3]))
        background_rgb = tuple(float(value) / 255.0 for value in rectangle.color[:3])
    else:
        triangle_shapes = [shape for shape in shapes if shape.shape_type == GEOMETRIZE_ISOSCELES_TRIANGLE and len(shape.data) >= 4]
        if not triangle_shapes:
            raise ValueError("No isosceles triangle shapes found in %s." % path)
        canvas_width = max(shape.data[0] + shape.data[2] for shape in triangle_shapes)
        canvas_height = max(shape.data[1] + shape.data[3] for shape in triangle_shapes)
        background_rgb = (1.0, 1.0, 1.0)

    return GeometrizePayload(
        shapes=shapes,
        canvas_width=max(canvas_width, 1.0),
        canvas_height=max(canvas_height, 1.0),
        background_rgb=background_rgb,
    )


class PixelTriangleTable(nn.Module):
    def __init__(
        self,
        centers_px: torch.Tensor,
        half_base_px: torch.Tensor,
        height_px: torch.Tensor,
        theta_rad: torch.Tensor,
        rgb: torch.Tensor,
        alpha: torch.Tensor,
        background_rgb: torch.Tensor,
    ) -> None:
        super().__init__()
        self.centers_px = nn.Parameter(centers_px)
        self.half_base_raw = nn.Parameter(inverse_softplus(half_base_px.clamp_min(1e-4)))
        self.height_raw = nn.Parameter(inverse_softplus(height_px.clamp_min(1e-4)))
        self.theta_rad = nn.Parameter(theta_rad)
        self.rgb_logits = nn.Parameter(inverse_sigmoid(rgb.clamp(1e-6, 1.0 - 1e-6)))
        self.register_buffer("alpha", alpha.clamp(0.0, 1.0))
        self.register_buffer("background_rgb", background_rgb.clamp(0.0, 1.0))

    @property
    def count(self) -> int:
        return int(self.centers_px.shape[0])

    def decoded(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        half_base = F.softplus(self.half_base_raw) + 1e-4
        height = F.softplus(self.height_raw) + 1e-4
        rgb = torch.sigmoid(self.rgb_logits)
        return self.centers_px, half_base, height, self.theta_rad, rgb, self.alpha

    def optimizer_groups(self, geometry_lr: float, color_lr: float) -> List[Dict[str, object]]:
        return [
            {"params": [self.centers_px, self.half_base_raw, self.height_raw, self.theta_rad], "lr": geometry_lr},
            {"params": [self.rgb_logits], "lr": color_lr},
        ]


def _table_from_payload(payload: GeometrizePayload, work_width: int, work_height: int, device: torch.device) -> PixelTriangleTable:
    scale_x = float(work_width) / float(payload.canvas_width)
    scale_y = float(work_height) / float(payload.canvas_height)
    triangles = [shape for shape in payload.shapes if shape.shape_type == GEOMETRIZE_ISOSCELES_TRIANGLE and len(shape.data) >= 5]
    if not triangles:
        raise ValueError("No type=%d isosceles triangle entries found." % GEOMETRIZE_ISOSCELES_TRIANGLE)

    centers = []
    half_bases = []
    heights = []
    theta = []
    rgb = []
    alpha = []
    for shape in triangles:
        x, y, half_base, tri_height, angle_degrees = shape.data[:5]
        centers.append([x * scale_x, y * scale_y])
        half_bases.append([max(half_base * scale_x, 1e-4)])
        heights.append([max(tri_height * scale_y, 1e-4)])
        theta.append([math.radians(angle_degrees)])
        rgba = shape.color + [255.0] * max(0, 4 - len(shape.color))
        rgb.append([rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0])
        alpha.append([rgba[3] / 255.0])

    background_rgb = torch.tensor(payload.background_rgb, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return PixelTriangleTable(
        centers_px=torch.tensor(centers, dtype=torch.float32, device=device),
        half_base_px=torch.tensor(half_bases, dtype=torch.float32, device=device),
        height_px=torch.tensor(heights, dtype=torch.float32, device=device),
        theta_rad=torch.tensor(theta, dtype=torch.float32, device=device),
        rgb=torch.tensor(rgb, dtype=torch.float32, device=device).clamp(0.0, 1.0),
        alpha=torch.tensor(alpha, dtype=torch.float32, device=device).clamp(0.0, 1.0),
        background_rgb=background_rgb,
    )


def _pixel_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(height, device=device, dtype=dtype) + 0.5
    x = torch.arange(width, device=device, dtype=dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return grid_x.unsqueeze(0), grid_y.unsqueeze(0)


def _signed_triangle_margin(table: PixelTriangleTable, height: int, width: int) -> torch.Tensor:
    centers, half_base, tri_height, theta, _, _ = table.decoded()
    device = centers.device
    dtype = centers.dtype
    grid_x, grid_y = _pixel_grid(height, width, device=device, dtype=dtype)

    dx = grid_x - centers[:, 0].view(-1, 1, 1)
    dy = grid_y - centers[:, 1].view(-1, 1, 1)
    cos_theta = torch.cos(theta[:, 0]).view(-1, 1, 1)
    sin_theta = torch.sin(theta[:, 0]).view(-1, 1, 1)
    x_rot = cos_theta * dx + sin_theta * dy
    y_rot = -sin_theta * dx + cos_theta * dy

    tri_height = tri_height[:, 0].view(-1, 1, 1).clamp_min(1e-4)
    half_base = half_base[:, 0].view(-1, 1, 1).clamp_min(1e-4)
    y_from_top = y_rot + (tri_height * 0.5)
    half_base_at_y = y_from_top * (half_base / tri_height)
    margins = torch.stack(
        (
            y_from_top,
            (tri_height * 0.5) - y_rot,
            x_rot + half_base_at_y,
            half_base_at_y - x_rot,
        ),
        dim=0,
    )
    return torch.amin(margins, dim=0)


def render_soft(table: PixelTriangleTable, height: int, width: int, temperature_px: float) -> torch.Tensor:
    _, _, _, _, rgb, alpha = table.decoded()
    signed_margin = _signed_triangle_margin(table, height=height, width=width)
    masks = torch.sigmoid(signed_margin / max(float(temperature_px), 1e-4))
    layer_alpha = (masks * alpha.view(-1, 1, 1)).unsqueeze(1)
    color_layers = rgb.view(-1, 3, 1, 1)
    transparency = (1.0 - layer_alpha).clamp(0.0, 1.0)

    reversed_transparency = torch.flip(transparency, dims=(0,))
    reversed_cumprod = torch.cumprod(reversed_transparency, dim=0)
    reversed_exclusive = torch.cat([torch.ones_like(reversed_transparency[:1]), reversed_cumprod[:-1]], dim=0)
    trans_after = torch.flip(reversed_exclusive, dims=(0,))

    background = table.background_rgb.expand(1, 3, height, width)
    background_trans = torch.prod(transparency, dim=0)
    weighted_colors = (layer_alpha * trans_after * color_layers).sum(dim=0, keepdim=True)
    return (background * background_trans + weighted_colors).clamp(0.0, 1.0)


@torch.no_grad()
def render_hard_loop(table: PixelTriangleTable, height: int, width: int, device: torch.device | None = None) -> torch.Tensor:
    centers, half_base, tri_height, theta, rgb, alpha = table.decoded()
    render_device = centers.device if device is None else device
    centers = centers.to(render_device)
    half_base = half_base.to(render_device)
    tri_height = tri_height.to(render_device)
    theta = theta.to(render_device)
    rgb = rgb.to(render_device)
    alpha = alpha.to(render_device)
    background = table.background_rgb.to(render_device).expand(1, 3, height, width).clone()
    grid_x, grid_y = _pixel_grid(height, width, device=render_device, dtype=centers.dtype)
    image = background
    for index in range(int(centers.shape[0])):
        dx = grid_x - centers[index, 0]
        dy = grid_y - centers[index, 1]
        cos_theta = torch.cos(theta[index, 0])
        sin_theta = torch.sin(theta[index, 0])
        x_rot = cos_theta * dx + sin_theta * dy
        y_rot = -sin_theta * dx + cos_theta * dy
        h = tri_height[index, 0].clamp_min(1e-4)
        hb = half_base[index, 0].clamp_min(1e-4)
        y_from_top = y_rot + (h * 0.5)
        half_base_at_y = y_from_top * (hb / h)
        mask = (
            (y_from_top >= 0.0)
            & (y_rot <= (h * 0.5))
            & (x_rot >= -half_base_at_y)
            & (x_rot <= half_base_at_y)
        ).to(dtype=centers.dtype).unsqueeze(1)
        layer_alpha = (mask * alpha[index].view(1, 1, 1, 1)).clamp(0.0, 1.0)
        color = rgb[index].view(1, 3, 1, 1)
        image = image * (1.0 - layer_alpha) + color * layer_alpha
    return image.clamp(0.0, 1.0).cpu()


def _l1(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(first.detach().cpu() - second.detach().cpu())).item())


def _rmse(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((first.detach().cpu() - second.detach().cpu()) ** 2)).item())


def _export_geometrize_json(table: PixelTriangleTable, path: Path, width: int, height: int) -> None:
    centers, half_base, tri_height, theta, rgb, alpha = table.decoded()
    centers = centers.detach().cpu()
    half_base = half_base.detach().cpu()
    tri_height = tri_height.detach().cpu()
    theta = theta.detach().cpu()
    rgb = rgb.detach().cpu()
    alpha = alpha.detach().cpu()
    background = table.background_rgb.detach().cpu().view(3)
    shapes: List[Dict[str, object]] = [
        {
            "type": GEOMETRIZE_RECTANGLE,
            "data": [0.0, 0.0, float(width), float(height)],
            "color": [int(round(float(value) * 255.0)) for value in background] + [255],
            "score": 0.0,
        }
    ]
    for index in range(int(centers.shape[0])):
        shapes.append(
            {
                "type": GEOMETRIZE_ISOSCELES_TRIANGLE,
                "data": [
                    float(centers[index, 0].item()),
                    float(centers[index, 1].item()),
                    float(half_base[index, 0].item()),
                    float(tri_height[index, 0].item()),
                    float(math.degrees(theta[index, 0].item()) % 360.0),
                ],
                "color": [
                    int(round(float(rgb[index, 0].item()) * 255.0)),
                    int(round(float(rgb[index, 1].item()) * 255.0)),
                    int(round(float(rgb[index, 2].item()) * 255.0)),
                    int(round(float(alpha[index, 0].item()) * 255.0)),
                ],
                "score": 0.0,
            }
        )
    serialize_json({"shapes": shapes}, path)


def resolve_seed(seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    return random.SystemRandom().randrange(0, 2**31)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Geometrize isosceles triangles with direct differentiable parameter optimization.")
    parser.add_argument("--input", required=True, help="Target image path.")
    parser.add_argument("--init-json", required=True, help="Geometrize JSON containing type=512 isosceles triangles.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--steps", type=int, default=1000, help="Optimization steps.")
    parser.add_argument("--work-size", type=int, default=512, help="Longest side used during optimization.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--device", default="auto", help="Torch device, for example auto, cpu or cuda.")
    parser.add_argument("--geometry-lr", type=float, default=2e-4, help="Learning rate for centers, size and angle.")
    parser.add_argument("--color-lr", type=float, default=2e-3, help="Learning rate for RGB colors.")
    parser.add_argument("--temperature", type=float, default=1.25, help="Initial soft mask temperature in pixels.")
    parser.add_argument("--final-temperature", type=float, default=0.35, help="Final soft mask temperature in pixels.")
    parser.add_argument("--metrics-every", type=int, default=50, help="Print metrics every N steps.")
    parser.add_argument("--progress-every", type=int, default=200, help="Save progress image every N steps.")
    parser.add_argument("--max-triangles", type=int, default=None, help="Optionally keep only the first N triangles from the JSON.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    seed = resolve_seed(args.seed)
    seed_all(seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output))
    progress_dir = ensure_dir(output_dir / "progress")

    loaded = load_image(Path(args.input), args.work_size)
    target_work = loaded.working.to(device=device, dtype=torch.float32)
    _, _, work_height, work_width = target_work.shape
    save_image(loaded.working, output_dir / "target_work.png")

    payload = _read_geometrize_json(Path(args.init_json))
    if args.max_triangles is not None:
        kept_shapes = []
        kept_triangles = 0
        for shape in payload.shapes:
            if shape.shape_type != GEOMETRIZE_ISOSCELES_TRIANGLE:
                kept_shapes.append(shape)
                continue
            if kept_triangles < int(args.max_triangles):
                kept_shapes.append(shape)
                kept_triangles += 1
        payload = GeometrizePayload(
            shapes=kept_shapes,
            canvas_width=payload.canvas_width,
            canvas_height=payload.canvas_height,
            background_rgb=payload.background_rgb,
        )

    table = _table_from_payload(payload, work_width=work_width, work_height=work_height, device=device)
    optimizer = torch.optim.Adam(table.optimizer_groups(geometry_lr=args.geometry_lr, color_lr=args.color_lr))

    initial_hard = render_hard_loop(table, height=work_height, width=work_width)
    save_image(initial_hard, output_dir / "initial.png")
    initial_l1 = _l1(initial_hard, loaded.working)
    initial_rmse = _rmse(initial_hard, loaded.working)

    best_l1 = initial_l1
    best_step = 0
    best_image = initial_hard
    history: List[Dict[str, float | int]] = [
        {
            "step": 0,
            "hard_l1": initial_l1,
            "hard_rmse": initial_rmse,
            "temperature": float(args.temperature),
        }
    ]
    print(
        "Starting direct optimization: triangles=%d work_size=%dx%d device=%s initial_hard_l1=%.6f initial_hard_rmse=%.6f"
        % (table.count, work_width, work_height, device, initial_l1, initial_rmse),
        flush=True,
    )

    total_steps = max(0, int(args.steps))
    for step in range(total_steps):
        one_based = step + 1
        progress = float(one_based - 1) / float(max(1, total_steps - 1))
        temperature = float(args.temperature) + (float(args.final_temperature) - float(args.temperature)) * progress

        prediction = render_soft(table, height=work_height, width=work_width, temperature_px=temperature)
        loss = F.l1_loss(prediction, target_work)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(table.parameters(), max_norm=5.0)
        optimizer.step()

        should_report = one_based == 1 or one_based % int(args.metrics_every) == 0 or one_based == total_steps
        should_progress = one_based % int(args.progress_every) == 0 or one_based == total_steps
        if should_report or should_progress:
            hard_image = render_hard_loop(table, height=work_height, width=work_width)
            hard_l1 = _l1(hard_image, loaded.working)
            hard_rmse = _rmse(hard_image, loaded.working)
            if hard_l1 < best_l1:
                best_l1 = hard_l1
                best_step = one_based
                best_image = hard_image.clone()
            if should_report:
                history.append(
                    {
                        "step": one_based,
                        "train_l1": float(loss.detach().cpu().item()),
                        "hard_l1": hard_l1,
                        "hard_rmse": hard_rmse,
                        "best_hard_l1": best_l1,
                        "temperature": temperature,
                    }
                )
                print(
                    "[step %d/%d] train_l1=%.6f hard_l1=%.6f hard_rmse=%.6f best_l1=%.6f temp=%.3f"
                    % (one_based, total_steps, float(loss.detach().cpu().item()), hard_l1, hard_rmse, best_l1, temperature),
                    flush=True,
                )
            if should_progress:
                save_image(hard_image, progress_dir / ("step_%04d.png" % one_based))

    final_hard = render_hard_loop(table, height=work_height, width=work_width)
    save_image(final_hard, output_dir / "final.png")
    save_image(best_image, output_dir / "best.png")

    full_height = int(loaded.original.shape[-2])
    full_width = int(loaded.original.shape[-1])
    scale_x = float(full_width) / float(work_width)
    scale_y = float(full_height) / float(work_height)
    with torch.no_grad():
        table.centers_px[:, 0].mul_(scale_x)
        table.centers_px[:, 1].mul_(scale_y)
        table.half_base_raw.copy_(inverse_softplus((F.softplus(table.half_base_raw) + 1e-4) * scale_x))
        table.height_raw.copy_(inverse_softplus((F.softplus(table.height_raw) + 1e-4) * scale_y))
    final_fullres = render_hard_loop(table, height=full_height, width=full_width, device=torch.device("cpu"))
    save_image(final_fullres, output_dir / "final_fullres.png")
    _export_geometrize_json(table, output_dir / "optimized_geometrize_fullres.json", width=full_width, height=full_height)

    serialize_json(
        {
            "input": str(args.input),
            "init_json": str(args.init_json),
            "seed": seed,
            "device": str(device),
            "triangle_count": table.count,
            "work_size": {"width": work_width, "height": work_height},
            "image_size": {"width": full_width, "height": full_height},
            "initial_hard_l1": initial_l1,
            "initial_hard_rmse": initial_rmse,
            "best_step": best_step,
            "best_hard_l1": best_l1,
            "final_hard_l1": _l1(final_hard, loaded.working),
            "final_hard_rmse": _rmse(final_hard, loaded.working),
            "history": history,
        },
        output_dir / "metrics.json",
    )
    print("Saved initial image to %s" % (output_dir / "initial.png"), flush=True)
    print("Saved best image to %s" % (output_dir / "best.png"), flush=True)
    print("Saved final image to %s" % (output_dir / "final.png"), flush=True)
    print("Saved metrics to %s" % (output_dir / "metrics.json"), flush=True)
    print("Saved optimized fullres Geometrize JSON to %s" % (output_dir / "optimized_geometrize_fullres.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


