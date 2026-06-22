from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import torch

from .config import FitConfig
from .fit_core import fit_image_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a single image with differentiable opaque isosceles triangles.")
    parser.add_argument("--input", required=True, help="Path to the target image.")
    parser.add_argument("--output", required=True, help="Directory to store final.png, ellipses.json and metrics.json.")
    parser.add_argument("--num-ellipses", type=int, default=200, help="Number of ellipses to optimize.")
    parser.add_argument("--steps", type=int, default=8000, help="Number of optimization steps.")
    parser.add_argument("--work-size", type=int, default=256, help="Longest side to optimize at before rerendering full size.")
    parser.add_argument("--seed", type=int, default=None, help="Optional global random seed. If omitted, a random seed is generated and recorded.")
    parser.add_argument("--device", default="auto", help="Torch device, for example auto, cpu or cuda.")
    parser.add_argument("--fast", action="store_true", help="Use a lighter-weight loss preset that disables LPIPS for faster optimization.")
    parser.add_argument("--profile", action="store_true", help="Collect and print coarse timing breakdowns for training and export.")
    return parser.parse_args(argv)


def resolve_seed(seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    return random.SystemRandom().randrange(0, 2**31)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    optimization_mode = "fast" if args.fast else "quality"
    seed = resolve_seed(args.seed)
    device = resolve_device(args.device)
    config = FitConfig.from_args(
        num_ellipses=args.num_ellipses,
        work_size=args.work_size,
        steps=args.steps,
        seed=seed,
        optimization_mode=optimization_mode,
        profile_enabled=args.profile,
    )
    print("Using seed=%d device=%s" % (seed, device))
    artifacts = fit_image_file(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        config=config,
        device=device,
    )
    print("Saved final image to %s" % artifacts.final_image_path)
    print("Saved triangle parameters to %s" % artifacts.ellipses_path)
    print("Saved metrics to %s" % artifacts.metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


