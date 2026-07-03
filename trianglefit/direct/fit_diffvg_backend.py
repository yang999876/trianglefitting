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

from .fit_geometrize_json import GEOMETRIZE_ISOSCELES_TRIANGLE, GeometrizePayload, _read_geometrize_json, resolve_device, resolve_seed
from .io import load_image, save_image
from .utils import ensure_dir, serialize_json


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return image
    kernel_size = 2 * int(math.ceil(3.0 * sigma)) + 1
    coords = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - (kernel_size // 2)
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    channels = image.shape[1]
    weight = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(channels, 1, -1, -1)
    padding = kernel_size // 2
    padded = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    return F.conv2d(padded, weight, groups=channels)


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


def _rgba_to_tensor(color: List[float], device: torch.device) -> torch.Tensor:
    rgba = color + [255.0] * max(0, 4 - len(color))
    return torch.tensor([rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0, rgba[3] / 255.0], dtype=torch.float32, device=device).clamp(0.0, 1.0)


def _triangle_vertices_px(
    x: float,
    y: float,
    half_base: float,
    height: float,
    angle_degrees: float,
    scale_x: float,
    scale_y: float,
    device: torch.device,
) -> torch.Tensor:
    theta = math.radians(angle_degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    local = [(0.0, -height * 0.5), (-half_base, height * 0.5), (half_base, height * 0.5)]
    points = []
    for lx, ly in local:
        px = x + (lx * cos_theta) - (ly * sin_theta)
        py = y + (lx * sin_theta) + (ly * cos_theta)
        points.append([px * scale_x, py * scale_y])
    return torch.tensor(points, dtype=torch.float32, device=device)


def _build_diffvg_scene(
    payload: GeometrizePayload,
    work_width: int,
    work_height: int,
    device: torch.device,
    max_triangles: int | None,
):
    pydiffvg = _import_diffvg()
    scale_x = float(work_width) / float(payload.canvas_width)
    scale_y = float(work_height) / float(payload.canvas_height)
    shapes = []
    shape_groups = []
    point_vars = []
    color_vars = []

    triangles = [shape for shape in payload.shapes if shape.shape_type == GEOMETRIZE_ISOSCELES_TRIANGLE and len(shape.data) >= 5]
    if max_triangles is not None:
        triangles = triangles[: int(max_triangles)]
    if not triangles:
        raise ValueError("No type=%d isosceles triangles found." % GEOMETRIZE_ISOSCELES_TRIANGLE)

    for shape in triangles:
        vertices = _triangle_vertices_px(
            x=shape.data[0],
            y=shape.data[1],
            half_base=shape.data[2],
            height=shape.data[3],
            angle_degrees=shape.data[4],
            scale_x=scale_x,
            scale_y=scale_y,
            device=device,
        )
        vertices.requires_grad_(True)
        color = _rgba_to_tensor(shape.color, device=device)
        color.requires_grad_(True)

        path = pydiffvg.Path(
            num_control_points=torch.zeros(3, dtype=torch.int32, device=device),
            points=vertices,
            stroke_width=torch.tensor(0.0, dtype=torch.float32, device=device),
            is_closed=True,
        )
        shapes.append(path)
        shape_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(shapes) - 1], dtype=torch.int32, device=device),
            fill_color=color,
        )
        shape_groups.append(shape_group)
        point_vars.append(vertices)
        color_vars.append(color)
    return pydiffvg, shapes, shape_groups, point_vars, color_vars


def _append_triangle_path(
    pydiffvg,
    shapes,
    shape_groups,
    point_vars,
    color_vars,
    vertices: torch.Tensor,
    color: torch.Tensor,
    device: torch.device,
) -> None:
    vertices.requires_grad_(True)
    color.requires_grad_(True)
    path = pydiffvg.Path(
        num_control_points=torch.zeros(3, dtype=torch.int32, device=device),
        points=vertices,
        stroke_width=torch.tensor(0.0, dtype=torch.float32, device=device),
        is_closed=True,
    )
    shapes.append(path)
    shape_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([len(shapes) - 1], dtype=torch.int32, device=device),
        fill_color=color,
    )
    shape_groups.append(shape_group)
    point_vars.append(vertices)
    color_vars.append(color)


def _random_triangle_vertices_px(
    center_x: float,
    center_y: float,
    half_base: float,
    height: float,
    angle_radians: float,
    device: torch.device,
) -> torch.Tensor:
    cos_theta = math.cos(angle_radians)
    sin_theta = math.sin(angle_radians)
    local = [(0.0, -height * 0.5), (-half_base, height * 0.5), (half_base, height * 0.5)]
    points = []
    for lx, ly in local:
        px = center_x + (lx * cos_theta) - (ly * sin_theta)
        py = center_y + (lx * sin_theta) + (ly * cos_theta)
        points.append([px, py])
    return torch.tensor(points, dtype=torch.float32, device=device)


def _build_random_diffvg_scene(
    payload: GeometrizePayload | None,
    target_work: torch.Tensor,
    work_width: int,
    work_height: int,
    device: torch.device,
    max_triangles: int | None,
    random_alpha: float,
    random_min_size: float,
    random_max_size: float,
    random_color_mode: str,
):
    pydiffvg = _import_diffvg()
    shapes = []
    shape_groups = []
    point_vars = []
    color_vars = []
    if payload is None:
        if max_triangles is None:
            raise ValueError("Random initialization without init_json needs --max-triangles.")
        triangle_count = int(max_triangles)
    else:
        json_triangles = [shape for shape in payload.shapes if shape.shape_type == GEOMETRIZE_ISOSCELES_TRIANGLE and len(shape.data) >= 5]
        triangle_count = len(json_triangles) if max_triangles is None else min(int(max_triangles), len(json_triangles))
    if triangle_count <= 0:
        raise ValueError("Random initialization needs a positive triangle count.")

    min_dim = float(min(work_width, work_height))
    min_size = max(1.0, float(random_min_size) * min_dim)
    max_size = max(min_size, float(random_max_size) * min_dim)
    target = target_work.detach()[0]

    for _ in range(triangle_count):
        center_x = random.random() * float(work_width)
        center_y = random.random() * float(work_height)
        half_base = min_size + random.random() * (max_size - min_size)
        height = min_size + random.random() * (max_size - min_size)
        angle = random.random() * math.tau
        vertices = _random_triangle_vertices_px(center_x, center_y, half_base, height, angle, device=device)

        if random_color_mode == "target":
            sample_x = int(max(0, min(work_width - 1, round(center_x))))
            sample_y = int(max(0, min(work_height - 1, round(center_y))))
            rgb = target[:, sample_y, sample_x].clone().detach().to(device=device, dtype=torch.float32)
        elif random_color_mode == "uniform":
            rgb = torch.rand(3, dtype=torch.float32, device=device)
        else:
            raise ValueError("Unknown random color mode: %s" % random_color_mode)
        alpha = torch.tensor([float(random_alpha)], dtype=torch.float32, device=device).clamp(0.0, 1.0)
        color = torch.cat([rgb.clamp(0.0, 1.0), alpha], dim=0).contiguous()
        _append_triangle_path(pydiffvg, shapes, shape_groups, point_vars, color_vars, vertices, color, device)

    return pydiffvg, shapes, shape_groups, point_vars, color_vars


def _build_jittered_diffvg_scene(
    payload: GeometrizePayload,
    target_work: torch.Tensor,
    work_width: int,
    work_height: int,
    device: torch.device,
    max_triangles: int | None,
    jitter_std: float,
    jitter_color_std: float,
    reinit_fraction: float,
    random_alpha: float,
    random_min_size: float,
    random_max_size: float,
    random_color_mode: str,
):
    pydiffvg = _import_diffvg()
    scale_x = float(work_width) / float(payload.canvas_width)
    scale_y = float(work_height) / float(payload.canvas_height)
    shapes = []
    shape_groups = []
    point_vars = []
    color_vars = []

    triangles = [shape for shape in payload.shapes if shape.shape_type == GEOMETRIZE_ISOSCELES_TRIANGLE and len(shape.data) >= 5]
    if max_triangles is not None:
        triangles = triangles[: int(max_triangles)]
    if not triangles:
        raise ValueError("No type=%d isosceles triangles found." % GEOMETRIZE_ISOSCELES_TRIANGLE)

    target = target_work.detach()[0]
    min_dim = float(min(work_width, work_height))
    jitter_px = max(0.0, float(jitter_std)) * min_dim
    color_jitter = max(0.0, float(jitter_color_std))
    reinit_count = int(round(max(0.0, min(1.0, float(reinit_fraction))) * len(triangles)))
    reinit_indices = set(random.sample(range(len(triangles)), reinit_count)) if reinit_count > 0 else set()
    min_size = max(1.0, float(random_min_size) * min_dim)
    max_size = max(min_size, float(random_max_size) * min_dim)

    for index, shape in enumerate(triangles):
        if index in reinit_indices:
            center_x = random.random() * float(work_width)
            center_y = random.random() * float(work_height)
            half_base = min_size + random.random() * (max_size - min_size)
            height = min_size + random.random() * (max_size - min_size)
            angle = random.random() * math.tau
            vertices = _random_triangle_vertices_px(center_x, center_y, half_base, height, angle, device=device)
            if random_color_mode == "target":
                sample_x = int(max(0, min(work_width - 1, round(center_x))))
                sample_y = int(max(0, min(work_height - 1, round(center_y))))
                rgb = target[:, sample_y, sample_x].clone().detach().to(device=device, dtype=torch.float32)
            elif random_color_mode == "uniform":
                rgb = torch.rand(3, dtype=torch.float32, device=device)
            else:
                raise ValueError("Unknown random color mode: %s" % random_color_mode)
            alpha = torch.tensor([float(random_alpha)], dtype=torch.float32, device=device).clamp(0.0, 1.0)
            color = torch.cat([rgb.clamp(0.0, 1.0), alpha], dim=0).contiguous()
        else:
            vertices = _triangle_vertices_px(
                x=shape.data[0],
                y=shape.data[1],
                half_base=shape.data[2],
                height=shape.data[3],
                angle_degrees=shape.data[4],
                scale_x=scale_x,
                scale_y=scale_y,
                device=device,
            )
            if jitter_px > 0.0:
                vertices = vertices + torch.randn_like(vertices) * jitter_px
            color = _rgba_to_tensor(shape.color, device=device)
            if color_jitter > 0.0:
                color = color + torch.randn_like(color) * color_jitter
                color = color.clamp(0.0, 1.0)
        _append_triangle_path(pydiffvg, shapes, shape_groups, point_vars, color_vars, vertices, color, device)

    return pydiffvg, shapes, shape_groups, point_vars, color_vars


def _render_diffvg(
    pydiffvg,
    shapes,
    shape_groups,
    width: int,
    height: int,
    background_rgb: Tuple[float, float, float],
    seed: int,
    samples: int,
) -> torch.Tensor:
    scene_args = pydiffvg.RenderFunction.serialize_scene(width, height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    image_rgba = render(width, height, samples, samples, seed, None, *scene_args)
    rgb = image_rgba[:, :, :3]
    alpha = image_rgba[:, :, 3:4]
    background = torch.tensor(background_rgb, dtype=rgb.dtype, device=rgb.device).view(1, 1, 3)
    image = (alpha * rgb) + ((1.0 - alpha) * background)
    return image.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)


def _l1(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(first.detach().cpu() - second.detach().cpu())).item())


def _rmse(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((first.detach().cpu() - second.detach().cpu()) ** 2)).item())


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="Optional JSON config file. CLI arguments override config values.")
    config_args, _ = config_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(description="Fine-tune Geometrize triangles through the diffvg backend.")
    parser.add_argument("--config", default=None, help="Optional JSON config file. CLI arguments override config values.")
    parser.add_argument("--input", default=None, help="Target image path.")
    parser.add_argument("--init-json", default=None, help="Geometrize JSON containing type=512 isosceles triangles.")
    parser.add_argument("--output", default=None, help="Output directory.")
    parser.add_argument("--steps", type=int, default=200, help="Optimization steps.")
    parser.add_argument("--work-size", type=int, default=256, help="Longest side used during optimization.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--device", default="auto", help="Torch device, for example auto, cpu or cuda.")
    parser.add_argument("--geometry-lr", type=float, default=0.5, help="Learning rate for diffvg path points.")
    parser.add_argument("--color-lr", type=float, default=1e-2, help="Learning rate for RGBA colors.")
    parser.add_argument("--samples", type=int, default=2, help="diffvg samples per axis.")
    parser.add_argument("--metrics-every", type=int, default=25, help="Print metrics every N steps.")
    parser.add_argument("--progress-every", type=int, default=100, help="Save progress image every N steps.")
    parser.add_argument("--max-triangles", type=int, default=None, help="Optionally keep only the first N triangles from the JSON.")
    parser.add_argument("--init-mode", choices=("json", "random", "jitter", "partial_random"), default="json", help="Initialize from Geometrize JSON, random triangles, jittered JSON, or partially randomized JSON.")
    parser.add_argument("--random-alpha", type=float, default=0.5, help="Alpha used by random initialization.")
    parser.add_argument("--random-min-size", type=float, default=0.02, help="Minimum random half-base/height as a fraction of the short side.")
    parser.add_argument("--random-max-size", type=float, default=0.15, help="Maximum random half-base/height as a fraction of the short side.")
    parser.add_argument("--random-color-mode", choices=("target", "uniform"), default="target", help="How random initialization chooses fill colors.")
    parser.add_argument("--jitter-std", type=float, default=0.03, help="Geometry jitter std as a fraction of the short side for jitter/partial_random init.")
    parser.add_argument("--jitter-color-std", type=float, default=0.03, help="Color jitter std in normalized RGBA units for jitter/partial_random init.")
    parser.add_argument("--reinit-fraction", type=float, default=0.25, help="Fraction of triangles to randomize for partial_random init.")
    parser.add_argument("--blur-sigma-start", type=float, default=0.0, help="Initial Gaussian blur sigma for coarse-to-fine annealing. 0 disables blur.")
    parser.add_argument("--blur-sigma-end", type=float, default=0.0, help="Final Gaussian blur sigma. Linearly interpolated from start to end over training.")
    parser.add_argument("--lpips-weight", type=float, default=0.0, help="Weight for LPIPS perceptual loss. 0 disables. Loss = (1-w)*L1 + w*LPIPS.")
    parser.add_argument("--lpips-cadence", type=int, default=1, help="Compute LPIPS every N steps (expensive). L1-only on other steps.")
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
    missing = [name for name in ("input", "output") if getattr(args, name) is None]
    if args.init_mode != "random" and args.init_json is None:
        missing.append("init_json")
    if args.init_mode == "random" and args.init_json is None and args.max_triangles is None:
        missing.append("max_triangles")
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

    payload = _read_geometrize_json(Path(args.init_json)) if args.init_json is not None else None
    if payload is None:
        background_rgb = tuple(float(value) for value in target_work.mean(dim=(0, 2, 3)).detach().cpu().tolist())
    else:
        background_rgb = payload.background_rgb
    if args.init_mode == "json":
        assert payload is not None
        pydiffvg, shapes, shape_groups, point_vars, color_vars = _build_diffvg_scene(
            payload=payload,
            work_width=work_width,
            work_height=work_height,
            device=device,
            max_triangles=args.max_triangles,
        )
    elif args.init_mode == "random":
        pydiffvg, shapes, shape_groups, point_vars, color_vars = _build_random_diffvg_scene(
            payload=payload,
            target_work=target_work,
            work_width=work_width,
            work_height=work_height,
            device=device,
            max_triangles=args.max_triangles,
            random_alpha=args.random_alpha,
            random_min_size=args.random_min_size,
            random_max_size=args.random_max_size,
            random_color_mode=args.random_color_mode,
        )
    else:
        assert payload is not None
        pydiffvg, shapes, shape_groups, point_vars, color_vars = _build_jittered_diffvg_scene(
            payload=payload,
            target_work=target_work,
            work_width=work_width,
            work_height=work_height,
            device=device,
            max_triangles=args.max_triangles,
            jitter_std=args.jitter_std,
            jitter_color_std=args.jitter_color_std,
            reinit_fraction=args.reinit_fraction if args.init_mode == "partial_random" else 0.0,
            random_alpha=args.random_alpha,
            random_min_size=args.random_min_size,
            random_max_size=args.random_max_size,
            random_color_mode=args.random_color_mode,
        )
    points_optimizer = torch.optim.Adam(point_vars, lr=args.geometry_lr)
    color_optimizer = torch.optim.Adam(color_vars, lr=args.color_lr)

    lpips_model = None
    if args.lpips_weight > 0.0:
        try:
            import lpips as _lpips_lib
            lpips_model = _lpips_lib.LPIPS(net="vgg").eval().to(device)
            for p in lpips_model.parameters():
                p.requires_grad = False
            print("LPIPS (vgg) loaded, weight=%.2f cadence=%d" % (args.lpips_weight, args.lpips_cadence), flush=True)
        except ImportError:
            from .losses import VGGFeatureDistance
            lpips_model = VGGFeatureDistance().eval().to(device)
            for p in lpips_model.parameters():
                p.requires_grad = False
            print("LPIPS not installed, using VGGFeatureDistance fallback, weight=%.2f" % args.lpips_weight, flush=True)

    with torch.no_grad():
        initial = _render_diffvg(pydiffvg, shapes, shape_groups, work_width, work_height, background_rgb, seed=seed, samples=args.samples)
    save_image(initial.cpu(), output_dir / "initial.png")
    initial_l1 = _l1(initial, loaded.working)
    initial_rmse = _rmse(initial, loaded.working)
    best_l1 = initial_l1
    best_step = 0
    best_image = initial.detach().cpu()
    history = [{"step": 0, "l1": initial_l1, "rmse": initial_rmse}]

    print(
        "Starting diffvg backend fit: triangles=%d work_size=%dx%d device=%s initial_l1=%.6f initial_rmse=%.6f blur_sigma=%.1f->%.1f"
        % (len(shapes), work_width, work_height, device, initial_l1, initial_rmse, args.blur_sigma_start, args.blur_sigma_end),
        flush=True,
    )

    for step in range(int(args.steps)):
        one_based = step + 1
        progress = float(step) / float(max(1, int(args.steps) - 1))
        blur_sigma = float(args.blur_sigma_start) + (float(args.blur_sigma_end) - float(args.blur_sigma_start)) * progress

        points_optimizer.zero_grad(set_to_none=True)
        color_optimizer.zero_grad(set_to_none=True)

        prediction = _render_diffvg(pydiffvg, shapes, shape_groups, work_width, work_height, background_rgb, seed=seed + one_based, samples=args.samples)
        blurred_prediction = _gaussian_blur(prediction, blur_sigma)
        blurred_target = _gaussian_blur(target_work, blur_sigma)
        l1_loss = F.l1_loss(blurred_prediction, blurred_target)
        if lpips_model is not None and (args.lpips_cadence <= 1 or one_based % args.lpips_cadence == 0):
            pred_lpips = blurred_prediction * 2.0 - 1.0
            tgt_lpips = blurred_target * 2.0 - 1.0
            lpips_loss = lpips_model(pred_lpips, tgt_lpips).mean()
            loss = (1.0 - args.lpips_weight) * l1_loss + args.lpips_weight * lpips_loss
        else:
            loss = l1_loss
        loss.backward()
        points_optimizer.step()
        color_optimizer.step()

        for color in color_vars:
            color.data.clamp_(0.0, 1.0)

        if one_based == 1 or one_based % int(args.metrics_every) == 0 or one_based == int(args.steps):
            with torch.no_grad():
                eval_image = _render_diffvg(pydiffvg, shapes, shape_groups, work_width, work_height, background_rgb, seed=seed, samples=args.samples)
            current_l1 = _l1(eval_image, loaded.working)
            current_rmse = _rmse(eval_image, loaded.working)
            if current_l1 < best_l1:
                best_l1 = current_l1
                best_step = one_based
                best_image = eval_image.detach().cpu()
            history.append(
                {
                    "step": one_based,
                    "train_l1": float(loss.detach().cpu().item()),
                    "eval_l1": current_l1,
                    "eval_rmse": current_rmse,
                    "best_l1": best_l1,
                    "blur_sigma": blur_sigma,
                }
            )
            print(
                "[step %d/%d] train_l1=%.6f eval_l1=%.6f eval_rmse=%.6f best_l1=%.6f blur_sigma=%.2f"
                % (one_based, int(args.steps), float(loss.detach().cpu().item()), current_l1, current_rmse, best_l1, blur_sigma),
                flush=True,
            )

        if one_based % int(args.progress_every) == 0 or one_based == int(args.steps):
            with torch.no_grad():
                progress_image = _render_diffvg(pydiffvg, shapes, shape_groups, work_width, work_height, background_rgb, seed=seed, samples=args.samples)
            save_image(progress_image.cpu(), progress_dir / ("step_%04d.png" % one_based))

    with torch.no_grad():
        final = _render_diffvg(pydiffvg, shapes, shape_groups, work_width, work_height, background_rgb, seed=seed, samples=args.samples)
    save_image(final.cpu(), output_dir / "final.png")
    save_image(best_image, output_dir / "best.png")
    serialize_json(
        {
            "input": str(args.input),
            "init_json": str(args.init_json),
            "seed": seed,
            "device": str(device),
            "init_mode": args.init_mode,
            "blur_sigma_start": args.blur_sigma_start,
            "blur_sigma_end": args.blur_sigma_end,
            "random_alpha": args.random_alpha if args.init_mode in ("random", "partial_random") else None,
            "random_min_size": args.random_min_size if args.init_mode in ("random", "partial_random") else None,
            "random_max_size": args.random_max_size if args.init_mode in ("random", "partial_random") else None,
            "random_color_mode": args.random_color_mode if args.init_mode in ("random", "partial_random") else None,
            "jitter_std": args.jitter_std if args.init_mode in ("jitter", "partial_random") else None,
            "jitter_color_std": args.jitter_color_std if args.init_mode in ("jitter", "partial_random") else None,
            "reinit_fraction": args.reinit_fraction if args.init_mode == "partial_random" else None,
            "triangle_count": len(shapes),
            "work_size": {"width": work_width, "height": work_height},
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
    print("Saved metrics to %s" % (output_dir / "metrics.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
