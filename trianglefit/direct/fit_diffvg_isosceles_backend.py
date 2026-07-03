from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .fit_geometrize_json import (
    GEOMETRIZE_ISOSCELES_TRIANGLE,
    GeometrizePayload,
    PixelTriangleTable,
    _export_geometrize_json,
    _read_geometrize_json,
    _table_from_payload,
    render_hard_loop,
    resolve_device,
    resolve_seed,
)
from .io import load_image, save_image
from .utils import ensure_dir, inverse_sigmoid, inverse_softplus, serialize_json
from ..greedy_prior.placer import HillClimbTrianglePlacer, ShapeBounds, TrianglePlacementConfig


def _import_diffvg():
    try:
        import pydiffvg  # type: ignore

        return pydiffvg
    except ImportError as exc:
        raise RuntimeError(
            "pydiffvg is not installed in this Python environment. "
            "Install the local third-party/diffvg package first, then rerun this command."
        ) from exc


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _triangle_vertices_from_table(table: PixelTriangleTable) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centers, half_base, tri_height, theta, rgb, alpha = table.decoded()
    cos_theta = torch.cos(theta[:, 0])
    sin_theta = torch.sin(theta[:, 0])

    local_x = torch.stack(
        (
            torch.zeros_like(half_base[:, 0]),
            -half_base[:, 0],
            half_base[:, 0],
        ),
        dim=1,
    )
    local_y = torch.stack(
        (
            -tri_height[:, 0] * 0.5,
            tri_height[:, 0] * 0.5,
            tri_height[:, 0] * 0.5,
        ),
        dim=1,
    )
    x = centers[:, 0:1] + local_x * cos_theta[:, None] - local_y * sin_theta[:, None]
    y = centers[:, 1:2] + local_x * sin_theta[:, None] + local_y * cos_theta[:, None]
    vertices = torch.stack((x, y), dim=2)
    rgba = torch.cat((rgb, alpha), dim=1).clamp(0.0, 1.0)
    return vertices, rgba, table.background_rgb.view(3)


def _render_diffvg_isosceles(
    pydiffvg,
    table: PixelTriangleTable,
    width: int,
    height: int,
    seed: int,
    samples: int,
) -> torch.Tensor:
    vertices, rgba, background_rgb = _triangle_vertices_from_table(table)
    device = vertices.device
    shapes = []
    shape_groups = []
    for index in range(int(vertices.shape[0])):
        path = pydiffvg.Path(
            num_control_points=torch.zeros(3, dtype=torch.int32, device=device),
            points=vertices[index].contiguous(),
            stroke_width=torch.tensor(0.0, dtype=torch.float32, device=device),
            is_closed=True,
        )
        shapes.append(path)
        shape_groups.append(
            pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([index], dtype=torch.int32, device=device),
                fill_color=rgba[index].contiguous(),
            )
        )

    scene_args = pydiffvg.RenderFunction.serialize_scene(width, height, shapes, shape_groups)
    image_rgba = pydiffvg.RenderFunction.apply(width, height, samples, samples, seed, None, *scene_args)
    rgb = image_rgba[:, :, :3]
    alpha = image_rgba[:, :, 3:4]
    background = background_rgb.to(dtype=rgb.dtype, device=rgb.device).view(1, 1, 3)
    image = (alpha * rgb) + ((1.0 - alpha) * background)
    return image.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)


@torch.no_grad()
def _render_hard_isosceles(
    table: PixelTriangleTable,
    width: int,
    height: int,
    exclude_indices: set[int] | None = None,
) -> torch.Tensor:
    centers, half_base, tri_height, theta, rgb, alpha = table.decoded()
    device = centers.device
    dtype = centers.dtype
    image = table.background_rgb.to(device=device, dtype=dtype).expand(1, 3, height, width).clone()
    y = torch.arange(height, device=device, dtype=dtype) + 0.5
    x = torch.arange(width, device=device, dtype=dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    exclude_indices = exclude_indices or set()
    for index in range(int(centers.shape[0])):
        if index in exclude_indices:
            continue
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
        ).to(dtype=dtype).view(1, 1, height, width)
        layer_alpha = (mask * alpha[index].view(1, 1, 1, 1)).clamp(0.0, 1.0)
        color = rgb[index].view(1, 3, 1, 1)
        image = image * (1.0 - layer_alpha) + color * layer_alpha
    return image.clamp(0.0, 1.0)


def _image_l1_tensor(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(first - second))


@torch.no_grad()
def _lowest_contribution_indices(
    table: PixelTriangleTable,
    target: torch.Tensor,
    width: int,
    height: int,
    count: int,
) -> Tuple[List[int], torch.Tensor, List[float]]:
    count = min(max(0, int(count)), table.count)
    if count == 0:
        return [], _render_hard_isosceles(table, width=width, height=height), []

    current = _render_hard_isosceles(table, width=width, height=height)
    base_l1 = _image_l1_tensor(current, target)
    contributions: List[float] = []
    for index in range(table.count):
        without = _render_hard_isosceles(table, width=width, height=height, exclude_indices={index})
        removed_l1 = _image_l1_tensor(without, target)
        contributions.append(float((removed_l1 - base_l1).detach().cpu().item()))

    ranked = sorted(range(table.count), key=lambda item: contributions[item])
    selected = ranked[:count]
    without_selected = _render_hard_isosceles(table, width=width, height=height, exclude_indices=set(selected))
    return selected, without_selected, contributions


@torch.no_grad()
def _sample_residual_bounds(
    target: torch.Tensor,
    current: torch.Tensor,
    bounds_fraction: float,
    generator: torch.Generator,
) -> ShapeBounds:
    _, _, height, width = target.shape
    residual = torch.mean(torch.abs(target - current), dim=1).view(-1)
    if float(residual.sum().detach().cpu().item()) <= 1e-12:
        flat_index = torch.randint(0, residual.numel(), (1,), device=target.device, generator=generator)
    else:
        flat_index = torch.multinomial(residual.clamp_min(0.0), 1, replacement=True, generator=generator)
    point = int(flat_index.detach().cpu().item())
    y = point // width
    x = point % width
    fraction = max(float(bounds_fraction), 1.0 / float(max(width, height)))
    half_w = max(2.0, float(width) * fraction * 0.5)
    half_h = max(2.0, float(height) * fraction * 0.5)
    x_min = max(0.0, (float(x) - half_w) / float(width))
    y_min = max(0.0, (float(y) - half_h) / float(height))
    x_max = min(1.0, (float(x) + half_w) / float(width))
    y_max = min(1.0, (float(y) + half_h) / float(height))
    return ShapeBounds.from_values((x_min, y_min, x_max, y_max))


@torch.no_grad()
def _write_placed_triangle_to_slot(table: PixelTriangleTable, index: int, triangle) -> None:
    device = table.centers_px.device
    dtype = table.centers_px.dtype
    table.centers_px[index].copy_(torch.tensor([triangle.cx, triangle.cy], device=device, dtype=dtype))
    table.half_base_raw[index].copy_(inverse_softplus(torch.tensor([triangle.half_base], device=device, dtype=dtype)))
    table.height_raw[index].copy_(inverse_softplus(torch.tensor([triangle.height], device=device, dtype=dtype)))
    table.theta_rad[index].copy_(torch.tensor([triangle.theta], device=device, dtype=dtype))
    table.rgb_logits[index].copy_(inverse_sigmoid(torch.tensor(triangle.rgb, device=device, dtype=dtype)))
    table.alpha[index].fill_(1.0)


def _rebirth_low_contribution_triangles(
    table: PixelTriangleTable,
    target: torch.Tensor,
    width: int,
    height: int,
    count: int,
    candidate_count: int,
    max_shape_mutations: int,
    bounds_fraction: float,
    seed: int,
    round_index: int,
) -> dict:
    selected, current, contributions = _lowest_contribution_indices(
        table=table,
        target=target,
        width=width,
        height=height,
        count=count,
    )
    if not selected:
        return {"indices": [], "before_l1": _l1(current, target), "after_l1": _l1(current, target)}

    before_l1 = _l1(current, target)
    generator = torch.Generator(device=target.device)
    generator.manual_seed(int(seed) + 1_000_003 + int(round_index) * 65_537)
    placer = HillClimbTrianglePlacer()
    reborn = []
    for offset, slot_index in enumerate(selected):
        bounds = _sample_residual_bounds(
            target=target,
            current=current,
            bounds_fraction=bounds_fraction,
            generator=generator,
        )
        config = TrianglePlacementConfig(
            num_triangles=1,
            candidate_count=int(candidate_count),
            max_shape_mutations=int(max_shape_mutations),
            candidate_chunk_size=max(1, min(int(candidate_count), 256)),
            seed=int(seed) + int(round_index) * 10_000 + offset,
            shape_bounds=bounds,
            background_rgb=tuple(float(value) for value in table.background_rgb.detach().cpu().view(3)),
        )
        result = placer.fit(target=target, config=config, initial_image=current)
        if not result.triangles:
            continue
        triangle = result.triangles[0]
        _write_placed_triangle_to_slot(table, slot_index, triangle)
        current = result.image.to(device=target.device, dtype=target.dtype)
        reborn.append(
            {
                "index": int(slot_index),
                "old_contribution_l1": float(contributions[slot_index]),
                "bounds": bounds.to_list(),
                "improvement_sse": float(triangle.improvement_sse),
            }
        )

    after_l1 = _l1(current, target)
    return {
        "indices": [int(index) for index in selected],
        "before_l1": before_l1,
        "after_l1": after_l1,
        "reborn": reborn,
    }


def _l1(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(first.detach().cpu() - second.detach().cpu())).item())


def _rmse(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((first.detach().cpu() - second.detach().cpu()) ** 2)).item())


def _trim_payload(payload: GeometrizePayload, max_triangles: int | None) -> GeometrizePayload:
    if max_triangles is None:
        return payload
    kept_shapes = []
    kept_triangles = 0
    for shape in payload.shapes:
        if shape.shape_type != GEOMETRIZE_ISOSCELES_TRIANGLE:
            kept_shapes.append(shape)
            continue
        if kept_triangles < int(max_triangles):
            kept_shapes.append(shape)
            kept_triangles += 1
    return GeometrizePayload(
        shapes=kept_shapes,
        canvas_width=payload.canvas_width,
        canvas_height=payload.canvas_height,
        background_rgb=payload.background_rgb,
    )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="Optional JSON config file. CLI arguments override config values.")
    config_args, _ = config_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(description="Fine-tune Geometrize isosceles triangles with diffvg while preserving isosceles parameterization.")
    parser.add_argument("--config", default=None, help="Optional JSON config file. CLI arguments override config values.")
    parser.add_argument("--input", default=None, help="Target image path.")
    parser.add_argument("--init-json", default=None, help="Geometrize JSON containing type=512 isosceles triangles.")
    parser.add_argument("--output", default=None, help="Output directory.")
    parser.add_argument("--steps", type=int, default=200, help="Optimization steps.")
    parser.add_argument("--work-size", type=int, default=512, help="Longest side used during optimization.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--device", default="auto", help="Torch device, for example auto, cpu or cuda.")
    parser.add_argument("--geometry-lr", type=float, default=2e-2, help="Learning rate for center, size and angle parameters.")
    parser.add_argument("--color-lr", type=float, default=1e-2, help="Learning rate for RGB colors.")
    parser.add_argument("--samples", type=int, default=2, help="diffvg samples per axis.")
    parser.add_argument("--metrics-every", type=int, default=25, help="Print metrics every N steps.")
    parser.add_argument("--progress-every", type=int, default=100, help="Save progress image every N steps.")
    parser.add_argument("--max-triangles", type=int, default=None, help="Optionally keep only the first N triangles from the JSON.")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Gradient clipping max norm. 0 disables clipping.")
    parser.add_argument("--rebirth-count", type=int, default=0, help="Number of low-contribution triangles to replace at each rebirth. 0 disables rebirth.")
    parser.add_argument("--rebirth-every", type=int, default=100, help="Run rebirth before this many gradient steps have elapsed.")
    parser.add_argument("--rebirth-initial", action=argparse.BooleanOptionalAction, default=False, help="Run one rebirth pass before gradient optimization.")
    parser.add_argument("--rebirth-candidate-count", type=int, default=256, help="CUDA greedy candidates used for each reborn triangle.")
    parser.add_argument("--rebirth-max-shape-mutations", type=int, default=512, help="CUDA greedy hill-climb mutations used for each reborn triangle.")
    parser.add_argument("--rebirth-bounds-fraction", type=float, default=0.2, help="Local search box side length as a fraction of image size around a residual-sampled point.")

    if config_args.config is not None:
        with Path(config_args.config).open("r", encoding="utf-8") as handle:
            raw_config = json.load(handle)
        if not isinstance(raw_config, dict):
            raise ValueError("Expected config file to contain a JSON object.")
        valid_dests = {action.dest for action in parser._actions}
        config_defaults = {str(key).replace("-", "_"): value for key, value in raw_config.items()}
        unknown = sorted(set(config_defaults) - valid_dests)
        if unknown:
            raise ValueError("Unknown config keys: %s" % ", ".join(unknown))
        parser.set_defaults(**config_defaults)

    args = parser.parse_args(argv)
    missing = [name for name in ("input", "init_json", "output") if getattr(args, name) is None]
    if missing:
        parser.error("Missing required arguments or config keys: %s" % ", ".join(missing))
    return args


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    seed = resolve_seed(args.seed)
    seed_all(seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output))
    progress_dir = ensure_dir(output_dir / "progress")

    pydiffvg = _import_diffvg()
    pydiffvg.set_use_gpu(device.type == "cuda")
    if device.type == "cuda":
        pydiffvg.set_device(device)

    loaded = load_image(Path(args.input), args.work_size)
    target_work = loaded.working.to(device=device, dtype=torch.float32)
    _, _, work_height, work_width = target_work.shape
    save_image(loaded.working, output_dir / "target_work.png")

    payload = _trim_payload(_read_geometrize_json(Path(args.init_json)), args.max_triangles)
    table = _table_from_payload(payload, work_width=work_width, work_height=work_height, device=device)
    rebirth_history: List[dict] = []

    if int(args.rebirth_count) > 0 and bool(args.rebirth_initial):
        event = _rebirth_low_contribution_triangles(
            table=table,
            target=target_work,
            width=work_width,
            height=work_height,
            count=int(args.rebirth_count),
            candidate_count=int(args.rebirth_candidate_count),
            max_shape_mutations=int(args.rebirth_max_shape_mutations),
            bounds_fraction=float(args.rebirth_bounds_fraction),
            seed=seed,
            round_index=0,
        )
        event["step"] = 0
        event["kind"] = "initial"
        rebirth_history.append(event)
        print(
            "[rebirth initial] replaced=%d l1 %.6f -> %.6f"
            % (len(event.get("reborn", [])), float(event["before_l1"]), float(event["after_l1"])),
            flush=True,
        )

    optimizer = torch.optim.Adam(table.optimizer_groups(geometry_lr=args.geometry_lr, color_lr=args.color_lr))

    with torch.no_grad():
        initial = _render_diffvg_isosceles(pydiffvg, table, work_width, work_height, seed=seed, samples=args.samples)
    save_image(initial.cpu(), output_dir / "initial.png")
    initial_l1 = _l1(initial, loaded.working)
    initial_rmse = _rmse(initial, loaded.working)
    best_l1 = initial_l1
    best_step = 0
    best_image = initial.detach().cpu()
    history = [{"step": 0, "l1": initial_l1, "rmse": initial_rmse}]

    print(
        "Starting diffvg isosceles fit: triangles=%d work_size=%dx%d device=%s initial_l1=%.6f initial_rmse=%.6f"
        % (table.count, work_width, work_height, device, initial_l1, initial_rmse),
        flush=True,
    )

    for step in range(int(args.steps)):
        one_based = step + 1
        if (
            int(args.rebirth_count) > 0
            and int(args.rebirth_every) > 0
            and step > 0
            and step % int(args.rebirth_every) == 0
        ):
            event = _rebirth_low_contribution_triangles(
                table=table,
                target=target_work,
                width=work_width,
                height=work_height,
                count=int(args.rebirth_count),
                candidate_count=int(args.rebirth_candidate_count),
                max_shape_mutations=int(args.rebirth_max_shape_mutations),
                bounds_fraction=float(args.rebirth_bounds_fraction),
                seed=seed,
                round_index=step,
            )
            event["step"] = step
            event["kind"] = "periodic"
            rebirth_history.append(event)
            optimizer = torch.optim.Adam(table.optimizer_groups(geometry_lr=args.geometry_lr, color_lr=args.color_lr))
            print(
                "[rebirth step %d] replaced=%d l1 %.6f -> %.6f"
                % (step, len(event.get("reborn", [])), float(event["before_l1"]), float(event["after_l1"])),
                flush=True,
            )

        optimizer.zero_grad(set_to_none=True)
        prediction = _render_diffvg_isosceles(pydiffvg, table, work_width, work_height, seed=seed + one_based, samples=args.samples)
        loss = F.l1_loss(prediction, target_work)
        loss.backward()
        if float(args.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(table.parameters(), max_norm=float(args.grad_clip))
        optimizer.step()

        should_report = one_based == 1 or one_based % int(args.metrics_every) == 0 or one_based == int(args.steps)
        should_progress = one_based % int(args.progress_every) == 0 or one_based == int(args.steps)
        if should_report or should_progress:
            with torch.no_grad():
                eval_image = _render_diffvg_isosceles(pydiffvg, table, work_width, work_height, seed=seed, samples=args.samples)
            current_l1 = _l1(eval_image, loaded.working)
            current_rmse = _rmse(eval_image, loaded.working)
            if current_l1 < best_l1:
                best_l1 = current_l1
                best_step = one_based
                best_image = eval_image.detach().cpu()
            if should_report:
                history.append(
                    {
                        "step": one_based,
                        "train_l1": float(loss.detach().cpu().item()),
                        "eval_l1": current_l1,
                        "eval_rmse": current_rmse,
                        "best_l1": best_l1,
                    }
                )
                print(
                    "[step %d/%d] train_l1=%.6f eval_l1=%.6f eval_rmse=%.6f best_l1=%.6f"
                    % (one_based, int(args.steps), float(loss.detach().cpu().item()), current_l1, current_rmse, best_l1),
                    flush=True,
                )
            if should_progress:
                save_image(eval_image.cpu(), progress_dir / ("step_%04d.png" % one_based))

    with torch.no_grad():
        final = _render_diffvg_isosceles(pydiffvg, table, work_width, work_height, seed=seed, samples=args.samples)
    save_image(final.cpu(), output_dir / "final.png")
    save_image(best_image, output_dir / "best.png")

    _export_geometrize_json(table, output_dir / "optimized_geometrize_work.json", width=work_width, height=work_height)
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
            "initial_l1": initial_l1,
            "initial_rmse": initial_rmse,
            "best_step": best_step,
            "best_l1": best_l1,
            "final_l1": _l1(final, loaded.working),
            "final_rmse": _rmse(final, loaded.working),
            "rebirth": {
                "count": int(args.rebirth_count),
                "every": int(args.rebirth_every),
                "initial": bool(args.rebirth_initial),
                "candidate_count": int(args.rebirth_candidate_count),
                "max_shape_mutations": int(args.rebirth_max_shape_mutations),
                "bounds_fraction": float(args.rebirth_bounds_fraction),
                "history": rebirth_history,
            },
            "history": history,
        },
        output_dir / "metrics.json",
    )
    print("Saved initial image to %s" % (output_dir / "initial.png"), flush=True)
    print("Saved best image to %s" % (output_dir / "best.png"), flush=True)
    print("Saved final image to %s" % (output_dir / "final.png"), flush=True)
    print("Saved fullres image to %s" % (output_dir / "final_fullres.png"), flush=True)
    print("Saved work JSON to %s" % (output_dir / "optimized_geometrize_work.json"), flush=True)
    print("Saved fullres JSON to %s" % (output_dir / "optimized_geometrize_fullres.json"), flush=True)
    print("Saved metrics to %s" % (output_dir / "metrics.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
