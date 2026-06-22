from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time
from typing import Dict, List

import torch
import torch.nn.functional as F

from .fit_diffvg_backend import (
    _build_diffvg_scene,
    _import_diffvg,
    _read_geometrize_json,
    _render_diffvg,
    resolve_device,
    resolve_seed,
    seed_all,
)
from .io import load_image


def _record(bucket: Dict[str, List[float]], name: str, elapsed: float) -> None:
    bucket.setdefault(name, []).append(float(elapsed))


def _summarize(bucket: Dict[str, List[float]]) -> None:
    total = sum(sum(values) for values in bucket.values())
    print("\nTiming summary", flush=True)
    for name, values in sorted(bucket.items(), key=lambda item: sum(item[1]), reverse=True):
        summed = sum(values)
        mean = statistics.mean(values)
        print(
            "%-18s total=%8.3fs mean=%8.3fs calls=%3d share=%5.1f%%"
            % (name, summed, mean, len(values), 100.0 * summed / max(total, 1e-9)),
            flush=True,
        )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the diffvg backend triangle fitter.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--init-json", required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--work-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--geometry-lr", type=float, default=0.1)
    parser.add_argument("--color-lr", type=float, default=0.01)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-triangles", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--diffvg-timing", action="store_true", help="Also print pydiffvg internal timing.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    seed = resolve_seed(args.seed)
    seed_all(seed)
    device = resolve_device(args.device)
    timings: Dict[str, List[float]] = {}

    started = time.perf_counter()
    pydiffvg = _import_diffvg()
    pydiffvg.set_use_gpu(device.type == "cuda")
    if device.type == "cuda":
        pydiffvg.set_device(device)
    if args.diffvg_timing and hasattr(pydiffvg, "set_print_timing"):
        pydiffvg.set_print_timing(True)
    _record(timings, "import_setup", time.perf_counter() - started)

    started = time.perf_counter()
    loaded = load_image(Path(args.input), args.work_size)
    target_work = loaded.working.to(device=device, dtype=torch.float32)
    _, _, work_height, work_width = target_work.shape
    payload = _read_geometrize_json(Path(args.init_json))
    _record(timings, "load_inputs", time.perf_counter() - started)

    started = time.perf_counter()
    pydiffvg, shapes, shape_groups, point_vars, color_vars = _build_diffvg_scene(
        payload=payload,
        work_width=work_width,
        work_height=work_height,
        device=device,
        max_triangles=args.max_triangles,
    )
    points_optimizer = torch.optim.Adam(point_vars, lr=args.geometry_lr)
    color_optimizer = torch.optim.Adam(color_vars, lr=args.color_lr)
    _record(timings, "build_scene", time.perf_counter() - started)

    print(
        "Profiling diffvg backend: triangles=%d work_size=%dx%d device=%s steps=%d samples=%d"
        % (len(shapes), work_width, work_height, device, int(args.steps), int(args.samples)),
        flush=True,
    )

    started = time.perf_counter()
    with torch.no_grad():
        initial = _render_diffvg(
            pydiffvg,
            shapes,
            shape_groups,
            work_width,
            work_height,
            payload.background_rgb,
            seed=seed,
            samples=args.samples,
        )
    _record(timings, "initial_render", time.perf_counter() - started)
    print("initial_l1=%.6f" % float(F.l1_loss(initial, target_work).detach().cpu().item()), flush=True)

    for step in range(int(args.steps)):
        one_based = step + 1
        started = time.perf_counter()
        points_optimizer.zero_grad(set_to_none=True)
        color_optimizer.zero_grad(set_to_none=True)
        _record(timings, "zero_grad", time.perf_counter() - started)

        started = time.perf_counter()
        prediction = _render_diffvg(
            pydiffvg,
            shapes,
            shape_groups,
            work_width,
            work_height,
            payload.background_rgb,
            seed=seed + one_based,
            samples=args.samples,
        )
        _record(timings, "train_render", time.perf_counter() - started)

        started = time.perf_counter()
        loss = F.l1_loss(prediction, target_work)
        _record(timings, "loss", time.perf_counter() - started)

        started = time.perf_counter()
        loss.backward()
        _record(timings, "backward", time.perf_counter() - started)

        started = time.perf_counter()
        points_optimizer.step()
        color_optimizer.step()
        for color in color_vars:
            color.data.clamp_(0.0, 1.0)
        _record(timings, "optimizer", time.perf_counter() - started)

        if int(args.eval_every) > 0 and (one_based % int(args.eval_every) == 0 or one_based == int(args.steps)):
            started = time.perf_counter()
            with torch.no_grad():
                eval_image = _render_diffvg(
                    pydiffvg,
                    shapes,
                    shape_groups,
                    work_width,
                    work_height,
                    payload.background_rgb,
                    seed=seed,
                    samples=args.samples,
                )
            eval_l1 = float(F.l1_loss(eval_image, target_work).detach().cpu().item())
            _record(timings, "eval_render", time.perf_counter() - started)
            print(
                "[step %d/%d] train_l1=%.6f eval_l1=%.6f"
                % (one_based, int(args.steps), float(loss.detach().cpu().item()), eval_l1),
                flush=True,
            )

    _summarize(timings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
