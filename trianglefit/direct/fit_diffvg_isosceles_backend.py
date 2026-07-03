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
from .utils import ensure_dir, inverse_softplus, serialize_json


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
