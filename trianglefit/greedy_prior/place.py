from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List

import torch
import torch.nn.functional as F

from .placer import (
    HillClimbTrianglePlacer,
    GreedyPlacementResult,
    PlacedTriangle,
    ShapeBounds,
    TrianglePlacementConfig,
    export_geometrize_json,
    normalize_rgb,
    render_triangles,
)
from ..direct.io import load_image, save_image
from ..direct.utils import ensure_dir, serialize_json


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_attention_mask(path: str | None, height: int, width: int, device: torch.device) -> torch.Tensor | None:
    if path is None:
        return None
    mask_path = Path(path)
    if mask_path.suffix.lower() == ".pt":
        mask = torch.load(mask_path, map_location="cpu", weights_only=False)
        if not isinstance(mask, torch.Tensor):
            raise ValueError("attention_mask .pt file must contain a torch.Tensor.")
    elif mask_path.suffix.lower() == ".npy":
        import numpy as np

        mask = torch.from_numpy(np.load(mask_path))
    else:
        from PIL import Image
        import numpy as np

        image = Image.open(mask_path).convert("F")
        mask = torch.from_numpy(np.asarray(image, dtype="float32"))
        if float(mask.max().item()) > 1.0:
            mask = mask / 255.0
        mask = 1.0 + mask

    mask = mask.to(dtype=torch.float32)
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim == 3:
        if mask.shape[0] in (1, 3):
            mask = mask.mean(dim=0)
        elif mask.shape[-1] in (1, 3):
            mask = mask.mean(dim=-1)
        else:
            raise ValueError("attention_mask tensor must be [H,W], [1,H,W], [1,1,H,W], or image-like.")
    if mask.ndim != 2:
        raise ValueError("attention_mask tensor must resolve to shape [H, W].")
    mask = mask.clamp_min(0.0).view(1, 1, int(mask.shape[-2]), int(mask.shape[-1]))
    if mask.shape[-2:] != (height, width):
        mask = F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
    return mask.to(device=device, dtype=torch.float32).contiguous()


def _normalize_resolution_schedule(raw_schedule) -> List[dict]:
    if raw_schedule is None:
        return []
    if isinstance(raw_schedule, str):
        raw_schedule = json.loads(raw_schedule)
    if not isinstance(raw_schedule, list):
        raise ValueError("resolution_schedule must be a list of stage objects.")
    stages = []
    previous_fraction = 0.0
    for index, item in enumerate(raw_schedule):
        if not isinstance(item, dict):
            raise ValueError("resolution_schedule entries must be JSON objects.")
        until_fraction = float(item.get("until_fraction", item.get("until", 0.0)))
        work_size = int(item["work_size"])
        use_attention = bool(item.get("attention", item.get("use_attention", True)))
        if work_size <= 0:
            raise ValueError("resolution_schedule[%d].work_size must be positive." % index)
        if until_fraction <= previous_fraction or until_fraction > 1.0:
            raise ValueError("resolution_schedule until_fraction values must increase up to 1.0.")
        stages.append({"until_fraction": until_fraction, "work_size": work_size, "attention": use_attention})
        previous_fraction = until_fraction
    if stages and stages[-1]["until_fraction"] < 1.0:
        stages.append({"until_fraction": 1.0, "work_size": int(stages[-1]["work_size"]), "attention": bool(stages[-1]["attention"])})
    return stages


def _placement_config_from_args(args, num_triangles: int, attention_mask: torch.Tensor | None) -> TrianglePlacementConfig:
    return TrianglePlacementConfig(
        num_triangles=int(num_triangles),
        candidate_count=int(args.candidate_count),
        max_shape_mutations=int(args.max_shape_mutations),
        seed=int(args.seed),
        shape_bounds=ShapeBounds.from_values(args.shape_bounds),
        min_half_base_fraction=float(args.min_half_base_fraction),
        max_half_base_fraction=float(args.max_half_base_fraction),
        min_height_fraction=float(args.min_height_fraction),
        max_height_fraction=float(args.max_height_fraction),
        center_mutation_fraction=float(args.center_mutation_fraction),
        size_mutation_fraction=float(args.size_mutation_fraction),
        angle_mutation_degrees=float(args.angle_mutation_degrees),
        background_rgb=args.background_rgb,
        attention_mask=attention_mask,
    )


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
    parser.add_argument("--seed", type=int, default=-1, help="-1 chooses a fresh random seed; otherwise the run is reproducible.")
    parser.add_argument("--shape-bounds", type=float, nargs=4, default=(0.0, 0.0, 1.0, 1.0), metavar=("X0", "Y0", "X1", "Y1"), help="Bounds for triangle centers. Fractions 0..1 or percentages 0..100 are accepted.")
    parser.add_argument("--background-rgb", type=float, nargs=3, default=None, metavar=("R", "G", "B"), help="Optional background color. Accepts 0..1 floats or 0..255 values.")
    parser.add_argument("--attention-mask", default=None, help="Optional attention mask path (.pt, .npy, or image). Higher values make greedy scoring care more about those pixels.")
    parser.add_argument("--resolution-schedule", default=None, help="Optional JSON/list schedule for staged dynamic resolution.")
    parser.add_argument("--work-size", type=int, default=256, help="Longest side used during greedy placement.")
    parser.add_argument("--device", default="auto", help="Device: auto or cuda. The greedy search core requires CUDA.")
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


def _run_dynamic_resolution(args: argparse.Namespace, stages: List[dict], output_dir: Path, progress_dir: Path, device: torch.device) -> int:
    total_triangles = int(args.num_triangles)
    full_loaded = load_image(Path(args.input), max(int(stage["work_size"]) for stage in stages))
    full_width, full_height = full_loaded.original_size
    placer = HillClimbTrianglePlacer()
    configured_background = normalize_rgb(args.background_rgb)

    triangles: List[PlacedTriangle] = []
    history: List[dict] = []
    stage_summaries: List[dict] = []
    background_rgb = (1.0, 1.0, 1.0)
    previous_width: int | None = None
    previous_height: int | None = None
    placed = 0
    seed = int(args.seed)
    final_image: torch.Tensor | None = None
    final_width = 0
    final_height = 0
    initial_sse = 0.0
    final_sse = 0.0

    for stage_index, stage in enumerate(stages):
        target_count = int(round(total_triangles * float(stage["until_fraction"])))
        target_count = min(total_triangles, max(placed, target_count))
        stage_count = target_count - placed
        if stage_count <= 0:
            continue

        loaded = load_image(Path(args.input), int(stage["work_size"]))
        target_work = loaded.working.to(device=device, dtype=torch.float32)
        _, _, work_height, work_width = target_work.shape
        stage_dir = ensure_dir(output_dir / ("stage_%02d_%d" % (stage_index + 1, int(stage["work_size"]))))
        save_image(loaded.working, stage_dir / "target_work.png")

        if previous_width is None or previous_height is None:
            if configured_background is None:
                background = target_work.mean(dim=(0, 2, 3)).clamp(0.0, 1.0)
            else:
                background = torch.tensor(configured_background, device=device, dtype=torch.float32)
            background_rgb = tuple(float(value) for value in background.detach().cpu())
            initial = background.view(1, 3, 1, 1).expand_as(target_work).clone()
        else:
            scale_x = float(work_width) / float(previous_width)
            scale_y = float(work_height) / float(previous_height)
            triangles = [triangle.scaled(scale_x=scale_x, scale_y=scale_y) for triangle in triangles]
            initial = render_triangles(
                triangles=triangles,
                background_rgb=background_rgb,
                width=work_width,
                height=work_height,
                device=device,
            ).to(device=device, dtype=torch.float32)
        save_image(initial.cpu(), stage_dir / "initial.png")

        use_attention = bool(stage["attention"]) and args.attention_mask is not None
        attention_mask = load_attention_mask(args.attention_mask, height=work_height, width=work_width, device=device) if use_attention else None
        config = _placement_config_from_args(args, num_triangles=stage_count, attention_mask=attention_mask)

        def progress_callback(done: int, total: int, result: GreedyPlacementResult) -> None:
            global_done = placed + done
            latest = result.history[-1] if result.history else None
            if latest is not None and int(args.metrics_every) > 0 and (
                done == 1 or global_done % int(args.metrics_every) == 0 or global_done == total_triangles
            ):
                print(
                    "[triangle %d/%d stage=%d work=%dx%d attention=%s] rmse=%.6f improvement_sse=%.3f"
                    % (
                        global_done,
                        total_triangles,
                        stage_index + 1,
                        work_width,
                        work_height,
                        "on" if attention_mask is not None else "off",
                        float(latest["rmse"]),
                        float(latest["improvement_sse"]),
                    ),
                    flush=True,
                )
            if int(args.progress_every) > 0 and (global_done % int(args.progress_every) == 0 or global_done == total_triangles):
                save_image(result.image.cpu(), progress_dir / ("step_%04d.png" % global_done))

        print(
            "Starting greedy stage %d/%d: add=%d cumulative=%d work=%dx%d attention=%s"
            % (stage_index + 1, len(stages), stage_count, target_count, work_width, work_height, "on" if attention_mask is not None else "off"),
            flush=True,
        )
        result = placer.fit(target=target_work, config=config, initial_image=initial, progress_callback=progress_callback)
        save_image(result.image.cpu(), stage_dir / "final.png")
        triangles.extend(result.triangles)
        for item in result.history:
            entry = dict(item)
            entry["stage"] = stage_index + 1
            entry["global_triangle"] = placed + int(entry["triangle"])
            entry["work_size"] = {"width": work_width, "height": work_height}
            entry["attention"] = attention_mask is not None
            history.append(entry)

        if stage_index == 0:
            initial_sse = result.initial_sse
        final_sse = result.final_sse
        final_image = result.image
        final_width = work_width
        final_height = work_height
        previous_width = work_width
        previous_height = work_height
        placed += len(result.triangles)
        stage_summaries.append(
            {
                "stage": stage_index + 1,
                "requested_add": stage_count,
                "added": len(result.triangles),
                "cumulative": placed,
                "work_size": {"width": work_width, "height": work_height},
                "attention": attention_mask is not None,
                "final_sse": result.final_sse,
                "final_rmse": result.final_rmse,
            }
        )
        if placed >= total_triangles:
            break

    if final_image is None or previous_width is None or previous_height is None:
        raise RuntimeError("Dynamic resolution greedy did not place any stages.")

    save_image(final_image.cpu(), output_dir / "final.png")
    export_geometrize_json(
        triangles=triangles,
        background_rgb=background_rgb,
        path=output_dir / "greedy_geometrize_work.json",
        width=final_width,
        height=final_height,
    )

    scale_x = float(full_width) / float(final_width)
    scale_y = float(full_height) / float(final_height)
    fullres_triangles = [triangle.scaled(scale_x=scale_x, scale_y=scale_y) for triangle in triangles]
    export_geometrize_json(
        triangles=fullres_triangles,
        background_rgb=background_rgb,
        path=output_dir / "greedy_geometrize.json",
        width=full_width,
        height=full_height,
    )
    final_fullres = render_triangles(
        triangles=fullres_triangles,
        background_rgb=background_rgb,
        width=full_width,
        height=full_height,
        device=torch.device("cpu"),
    )
    save_image(final_fullres, output_dir / "final_fullres.png")

    final_mse = final_sse / float(max(1, final_width * final_height * 3))
    metrics = {
        "input": str(args.input),
        "seed": seed,
        "device": str(device),
        "placer": args.placer,
        "triangle_count": len(triangles),
        "requested_triangle_count": total_triangles,
        "candidate_count": int(args.candidate_count),
        "max_shape_mutations": int(args.max_shape_mutations),
        "shape_bounds": ShapeBounds.from_values(args.shape_bounds).to_list(),
        "attention_mask": None if args.attention_mask is None else str(args.attention_mask),
        "background_rgb": list(background_rgb),
        "work_size": {"width": final_width, "height": final_height},
        "image_size": {"width": full_width, "height": full_height},
        "initial_sse": initial_sse,
        "final_sse": final_sse,
        "final_mse": final_mse,
        "final_rmse": final_mse**0.5,
        "resolution_schedule": stages,
        "stages": stage_summaries,
        "history": history,
    }
    serialize_json(metrics, output_dir / "metrics.json")

    print("Saved final image to %s" % (output_dir / "final.png"), flush=True)
    print("Saved fullres image to %s" % (output_dir / "final_fullres.png"), flush=True)
    print("Saved work JSON to %s" % (output_dir / "greedy_geometrize_work.json"), flush=True)
    print("Saved fullres JSON to %s" % (output_dir / "greedy_geometrize.json"), flush=True)
    print("Saved metrics to %s" % (output_dir / "metrics.json"), flush=True)
    return 0


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = ensure_dir(Path(args.output))
    progress_dir = ensure_dir(output_dir / "progress")
    device = resolve_device(args.device)
    stages = _normalize_resolution_schedule(args.resolution_schedule)
    if stages:
        return _run_dynamic_resolution(args=args, stages=stages, output_dir=output_dir, progress_dir=progress_dir, device=device)

    loaded = load_image(Path(args.input), args.work_size)
    target_work = loaded.working.to(device=device, dtype=torch.float32)
    _, _, work_height, work_width = target_work.shape
    full_width, full_height = loaded.original_size
    save_image(loaded.working, output_dir / "target_work.png")
    attention_mask = load_attention_mask(args.attention_mask, height=work_height, width=work_width, device=device)

    config = TrianglePlacementConfig(
        num_triangles=int(args.num_triangles),
        candidate_count=int(args.candidate_count),
        max_shape_mutations=int(args.max_shape_mutations),
        seed=int(args.seed),
        shape_bounds=ShapeBounds.from_values(args.shape_bounds),
        min_half_base_fraction=float(args.min_half_base_fraction),
        max_half_base_fraction=float(args.max_half_base_fraction),
        min_height_fraction=float(args.min_height_fraction),
        max_height_fraction=float(args.max_height_fraction),
        center_mutation_fraction=float(args.center_mutation_fraction),
        size_mutation_fraction=float(args.size_mutation_fraction),
        angle_mutation_degrees=float(args.angle_mutation_degrees),
        background_rgb=args.background_rgb,
        attention_mask=attention_mask,
    )
    configured_background = normalize_rgb(config.background_rgb)
    if configured_background is None:
        background = target_work.mean(dim=(0, 2, 3)).clamp(0.0, 1.0)
    else:
        background = torch.tensor(configured_background, device=device, dtype=torch.float32)
    initial = background.view(1, 3, 1, 1).expand_as(target_work).clone()
    save_image(initial.cpu(), output_dir / "initial.png")

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
        "Starting greedy placement: placer=%s triangles=%d candidates=%d mutations=%d work=%dx%d device=%s attention=%s"
        % (
            args.placer,
            config.num_triangles,
            config.candidate_count,
            config.max_shape_mutations,
            work_width,
            work_height,
            device,
            "on" if attention_mask is not None else "off",
        ),
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
        "shape_bounds": config.shape_bounds.to_list(),
        "attention_mask": None if args.attention_mask is None else str(args.attention_mask),
        "background_rgb": list(result.background_rgb),
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
