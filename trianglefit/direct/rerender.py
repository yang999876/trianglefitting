from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from .io import save_image
from .model import ROLE_ACCENT, ROLE_BASE, ROLE_TEXTURE, ellipse_parameters_from_json
from .renderer import render_parameters


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerender a triangle parameter JSON export.")
    parser.add_argument("--params", required=True, help="Path to ellipses.json.")
    parser.add_argument("--output", required=True, help="Path to the output PNG.")
    parser.add_argument("--width", type=int, default=None, help="Optional override width.")
    parser.add_argument("--height", type=int, default=None, help="Optional override height.")
    parser.add_argument("--temperature", type=float, default=None, help="Optional mask temperature override.")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda.")
    parser.add_argument("--hard-edges", dest="hard_edges", action="store_true", help="Force binary hard-edge triangle rendering.")
    parser.add_argument("--soft-edges", dest="hard_edges", action="store_false", help="Force soft differentiable rendering for preview/debugging.")
    parser.set_defaults(hard_edges=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with Path(args.params).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    width = args.width or int(payload["image_width"])
    height = args.height or int(payload["image_height"])
    temperature = args.temperature or float(payload.get("final_temperature", 0.06))
    background_rgb = tuple(float(value) for value in payload.get("background_rgb", [1.0, 1.0, 1.0]))
    hard_edges = args.hard_edges
    if hard_edges is None:
        hard_edges = payload.get("export_render_mode", "soft_edges") == "hard_edges"
    device = torch.device(args.device)
    background_grid = None
    if payload.get("background_grid_rgb") is not None:
        background_grid = torch.tensor(payload["background_grid_rgb"], dtype=torch.float32, device=device)
        if background_grid.ndim == 3:
            background_grid = background_grid.permute(2, 0, 1).unsqueeze(0)
    parameters = ellipse_parameters_from_json(payload["ellipses"], device=device, background_grid_rgb=background_grid)
    role_ids = parameters.role_ids
    base_count = int(payload.get("base_ellipse_count", int((role_ids == ROLE_BASE).sum().item())))
    texture_count = int(payload.get("texture_ellipse_count", int((role_ids == ROLE_TEXTURE).sum().item())))
    accent_count = int(payload.get("accent_ellipse_count", int((role_ids == ROLE_ACCENT).sum().item())))
    active_base_count = min(base_count, parameters.centers.shape[0])
    active_texture_count = min(texture_count, max(0, parameters.centers.shape[0] - active_base_count))
    active_accent_count = min(accent_count, max(0, parameters.centers.shape[0] - active_base_count - active_texture_count))

    with torch.no_grad():
        render = render_parameters(
            decoded=parameters,
            height=height,
            width=width,
            mask_temperature=temperature,
            background_rgb=background_rgb,
            hard_edges=hard_edges,
            active_base_count=active_base_count,
            active_texture_count=active_texture_count,
            active_accent_count=active_accent_count,
        )
    save_image(render.image, Path(args.output))
    print("Saved rerendered image to %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


