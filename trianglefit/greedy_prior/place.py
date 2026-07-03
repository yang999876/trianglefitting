from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List

import torch

from .placer import (
    HillClimbTrianglePlacer,
    ShapeBounds,
    TrianglePlacementConfig,
    export_geometrize_json,
    render_triangles,
)
from ..direct.io import load_image, save_image
from ..direct.utils import ensure_dir, serialize_json


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="Optional JSON config file. CLI arguments override config values.")
    config_args, _ = config_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(description="Greedy Geometrize-style placement of opaque isosceles triangles.")
    parser.add_argument("--config", default=None, help="Optional JSON config file. CLI arguments override config values.")
    parser.add_argument("--input", default=None, help="Target image path.")
    parser.add_argument("--output", default=None, help="Output directory.")
    parser.add_argument("--placer", choices=("hill_climb",), default="hill_climb", help="Greedy placement strategy.")
    parser.add_argument("--num-triangles", type=int, default=300, help="Maximum number of accepted triangles.")
    parser.add_argument("--candidate-count", type=int, default=2048, help="Random candidates hill-climbed in parallel each step.")
    parser.add_argument("--max-shape-mutations", type=int, default=2000, help="Mutation attempts for every candidate each step.")
    parser.add_argument("--candidate-chunk-size", type=int, default=256, help="Candidate batch size used during scoring.")
    parser.add_argument("--seed", type=int, default=-1, help="-1 chooses a fresh random seed; otherwise the run is reproducible.")
    parser.add_argument("--shape-bounds", type=float, nargs=4, default=(0.0, 0.0, 1.0, 1.0), metavar=("X0", "Y0", "X1", "Y1"), help="Bounds for triangle centers. Fractions 0..1 or percentages 0..100 are accepted.")
    parser.add_argument("--work-size", type=int, default=256, help="Longest side used during greedy placement.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--min-half-base-fraction", type=float, default=1.0 / 256.0)
    parser.add_argument("--max-half-base-fraction", type=float, default=32.0 / 256.0)
    parser.add_argument("--min-height-fraction", type=float, default=1.0 / 256.0)
    parser.add_argument("--max-height-fraction", type=float, default=64.0 / 256.0)
    parser.add_argument("--center-mutation-fraction", type=float, default=32.0 / 256.0)
    parser.add_argument("--size-mutation-fraction", type=float, default=16.0 / 256.0)
    parser.add_argument("--angle-mutation-degrees", type=float, default=16.0)
    parser.add_argument("--metrics-every", type=int, default=10, help="Print metrics every N accepted triangles; 0 disables periodic logs.")
    parser.add_argument("--progress-every", type=int, default=25, help="Save progress image every N accepted triangles; 0 disables progress images.")
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
    if missing:
        parser.error("Missing required arguments or config keys: %s" % ", ".join(missing))
    return args


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = ensure_dir(Path(args.output))
    progress_dir = ensure_dir(output_dir / "progress")
    device = resolve_device(args.device)

    loaded = load_image(Path(args.input), args.work_size)
    target_work = loaded.working.to(device=device, dtype=torch.float32)
    _, _, work_height, work_width = target_work.shape
    full_width, full_height = loaded.original_size
    save_image(loaded.working, output_dir / "target_work.png")

    background = target_work.mean(dim=(0, 2, 3)).clamp(0.0, 1.0)
    initial = background.view(1, 3, 1, 1).expand_as(target_work).clone()
    save_image(initial.cpu(), output_dir / "initial.png")

    config = TrianglePlacementConfig(
        num_triangles=int(args.num_triangles),
        candidate_count=int(args.candidate_count),
        max_shape_mutations=int(args.max_shape_mutations),
        candidate_chunk_size=int(args.candidate_chunk_size),
        seed=int(args.seed),
        shape_bounds=ShapeBounds.from_values(args.shape_bounds),
        min_half_base_fraction=float(args.min_half_base_fraction),
        max_half_base_fraction=float(args.max_half_base_fraction),
        min_height_fraction=float(args.min_height_fraction),
        max_height_fraction=float(args.max_height_fraction),
        center_mutation_fraction=float(args.center_mutation_fraction),
        size_mutation_fraction=float(args.size_mutation_fraction),
        angle_mutation_degrees=float(args.angle_mutation_degrees),
    )

    placer = HillClimbTrianglePlacer()

    def progress_callback(done: int, total: int, result) -> None:
        latest = result.history[-1] if result.history else None
        if latest is not None and int(args.metrics_every) > 0 and (done == 1 or done % int(args.metrics_every) == 0 or done == total):
            print(
                "[triangle %d/%d] rmse=%.6f improvement_sse=%.3f"
                % (done, total, float(latest["rmse"]), float(latest["improvement_sse"])),
                flush=True,
            )
        if int(args.progress_every) > 0 and (done % int(args.progress_every) == 0 or done == total):
            save_image(result.image.cpu(), progress_dir / ("step_%04d.png" % done))

    print(
        "Starting greedy placement: placer=%s triangles=%d candidates=%d mutations=%d work=%dx%d device=%s"
        % (args.placer, config.num_triangles, config.candidate_count, config.max_shape_mutations, work_width, work_height, device),
        flush=True,
    )
    result = placer.fit(target=target_work, config=config, initial_image=initial, progress_callback=progress_callback)
    save_image(result.image.cpu(), output_dir / "final.png")

    export_geometrize_json(
        triangles=result.triangles,
        background_rgb=result.background_rgb,
        path=output_dir / "greedy_geometrize_work.json",
        width=work_width,
        height=work_height,
    )

    scale_x = float(full_width) / float(work_width)
    scale_y = float(full_height) / float(work_height)
    fullres_triangles = [triangle.scaled(scale_x=scale_x, scale_y=scale_y) for triangle in result.triangles]
    export_geometrize_json(
        triangles=fullres_triangles,
        background_rgb=result.background_rgb,
        path=output_dir / "greedy_geometrize.json",
        width=full_width,
        height=full_height,
    )
    final_fullres = render_triangles(
        triangles=fullres_triangles,
        background_rgb=result.background_rgb,
        width=full_width,
        height=full_height,
        device=torch.device("cpu"),
    )
    save_image(final_fullres, output_dir / "final_fullres.png")

    metrics = {
        "input": str(args.input),
        "seed": result.seed,
        "device": str(device),
        "placer": args.placer,
        "triangle_count": result.triangle_count,
        "requested_triangle_count": config.num_triangles,
        "candidate_count": config.candidate_count,
        "max_shape_mutations": config.max_shape_mutations,
        "candidate_chunk_size": config.candidate_chunk_size,
        "shape_bounds": config.shape_bounds.to_list(),
        "work_size": {"width": work_width, "height": work_height},
        "image_size": {"width": full_width, "height": full_height},
        "initial_sse": result.initial_sse,
        "initial_rmse": (result.initial_sse / float(max(1, work_width * work_height * 3))) ** 0.5,
        "final_sse": result.final_sse,
        "final_mse": result.final_mse,
        "final_rmse": result.final_rmse,
        "history": result.history,
    }
    serialize_json(metrics, output_dir / "metrics.json")

    print("Saved final image to %s" % (output_dir / "final.png"), flush=True)
    print("Saved fullres image to %s" % (output_dir / "final_fullres.png"), flush=True)
    print("Saved work JSON to %s" % (output_dir / "greedy_geometrize_work.json"), flush=True)
    print("Saved fullres JSON to %s" % (output_dir / "greedy_geometrize.json"), flush=True)
    print("Saved metrics to %s" % (output_dir / "metrics.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
