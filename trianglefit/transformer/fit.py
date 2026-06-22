from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import os
import random
import sys
import time
from typing import Dict, List

import numpy as np
import torch

from trianglefit.direct.config import LossWeights
from trianglefit.direct.fit_core import _background_grid_to_json, _decoded_to_cpu, _render_export_image
from trianglefit.direct.io import load_image, save_image
from trianglefit.direct.losses import CompositeEllipseLoss
from trianglefit.direct.renderer import render_parameters
from trianglefit.direct.utils import ensure_dir, flatten_metrics, serialize_json, seed_all

from .config import TransformerTriangleConfig
from .model import TriangleTransformerGenerator


@dataclass(frozen=True)
class TransformerFitArtifacts:
    final_image_path: str
    ellipses_path: str
    metrics_path: str
    progress_dir: str


@dataclass(frozen=True)
class CheckpointState:
    step: int
    best_loss: float
    history: List[Dict[str, object]]
    elapsed_seconds: float


def _rng_state_dict() -> Dict[str, object]:
    state: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state_dict(state: Dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    config: TransformerTriangleConfig,
    model: TriangleTransformerGenerator,
    optimizer: torch.optim.Optimizer,
    best_loss: float,
    best_decoded,
    history: List[Dict[str, object]],
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "config": config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_loss": best_loss,
            "best_decoded": best_decoded,
            "history": history,
            "elapsed_seconds": elapsed_seconds,
            "rng_state": _rng_state_dict(),
        },
        path,
    )


def _load_checkpoint(path: Path, map_location: torch.device) -> Dict[str, object]:
    return torch.load(path, map_location=map_location, weights_only=False)


def _config_signature(payload: Dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _config_summary(payload: Dict[str, object]) -> Dict[str, object]:
    keys = (
        "num_triangles",
        "work_size",
        "seed",
        "lr",
        "mask_temperature",
        "l1_weight",
        "lpips_weight",
        "d_model",
        "num_heads",
        "num_decoder_layers",
        "dim_feedforward",
        "pretrained_backbone",
        "freeze_backbone",
    )
    return {key: payload.get(key) for key in keys}


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a single image with a DETR-style triangle Transformer.")
    parser.add_argument("--input", required=True, help="Path to the target image.")
    parser.add_argument("--output", required=True, help="Directory to store final.png, ellipses.json and metrics.json.")
    parser.add_argument("--num-triangles", type=int, default=300, help="Number of generated triangles.")
    parser.add_argument("--steps", type=int, default=3000, help="Number of optimization steps.")
    parser.add_argument("--work-size", type=int, default=256, help="Longest side used during optimization.")
    parser.add_argument("--seed", type=int, default=None, help="Optional global random seed.")
    parser.add_argument("--device", default="auto", help="Torch device, for example auto, cpu or cuda.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for trainable Transformer parts.")
    parser.add_argument("--fast", action="store_true", help="Use L1 only and skip LPIPS.")
    parser.add_argument("--no-pretrained", action="store_true", help="Use randomly initialized ResNet18 features.")
    parser.add_argument("--finetune-backbone", action="store_true", help="Also train the ResNet18 backbone.")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Save checkpoint every N steps.")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a checkpoint path.")
    parser.add_argument("--resume-strict", action="store_true", help="Fail if the checkpoint config does not match the current CLI config.")
    return parser.parse_args(argv)


def resolve_seed(seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    return random.SystemRandom().randrange(0, 2**31)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _export_triangles_json(decoded, output_path: Path, image_width: int, image_height: int, config: TransformerTriangleConfig, perceptual_mode: str) -> None:
    cpu_decoded = _decoded_to_cpu(decoded)
    payload = {
        "image_width": image_width,
        "image_height": image_height,
        "num_triangles": config.num_triangles,
        "base_ellipse_count": config.base_count(),
        "texture_ellipse_count": config.texture_count(),
        "accent_ellipse_count": config.accent_count(),
        "background_grid_height": int(cpu_decoded.background_grid_rgb.shape[-2]) if cpu_decoded.background_grid_rgb is not None else 0,
        "background_grid_width": int(cpu_decoded.background_grid_rgb.shape[-1]) if cpu_decoded.background_grid_rgb is not None else 0,
        "background_grid_rgb": _background_grid_to_json(cpu_decoded),
        "final_temperature": config.mask_temperature,
        "export_render_mode": "hard_edges",
        "perceptual_mode": perceptual_mode,
        "ellipses": cpu_decoded.to_json_list(),
    }
    serialize_json(payload, output_path)


def fit_image_file(
    input_path: Path,
    output_dir: Path,
    config: TransformerTriangleConfig,
    device: torch.device,
    resume_checkpoint: Path | None = None,
    resume_strict: bool = False,
) -> TransformerFitArtifacts:
    seed_all(config.seed)
    ensure_dir(output_dir)
    progress_dir = ensure_dir(output_dir / "progress")

    loaded = load_image(input_path, config.work_size)
    target_work = loaded.working.to(device=device, dtype=torch.float32)
    _, _, work_height, work_width = target_work.shape
    full_height = loaded.original.shape[-2]
    full_width = loaded.original.shape[-1]
    save_image(loaded.working, output_dir / "target_work.png")

    generator = TriangleTransformerGenerator(
        num_triangles=config.num_triangles,
        base_count=config.base_count(),
        texture_count=config.texture_count(),
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        pretrained_backbone=config.pretrained_backbone,
        freeze_backbone=config.freeze_backbone,
        min_size=config.min_size,
        base_max_size=config.base_max_size,
        texture_max_size=config.texture_max_size,
    ).to(device=device)
    optimizer = torch.optim.AdamW((parameter for parameter in generator.parameters() if parameter.requires_grad), lr=config.lr)

    loss_weights = LossWeights(l1=config.l1_weight, lpips=config.lpips_weight)
    loss_module = CompositeEllipseLoss(
        area_regularization_weight=0.0,
        perceptual_enabled=config.lpips_weight > 0.0,
        edge_loss_enabled=False,
        band_loss_enabled=False,
    ).to(device=device)
    loss_module.set_target(target_work)

    base_count, texture_count, accent_count = generator.role_counts()
    history: List[Dict[str, object]] = []
    best_loss = float("inf")
    best_decoded = None
    start_step = 0
    elapsed_offset = 0.0
    start_time = time.perf_counter()
    if resume_checkpoint is not None:
        checkpoint = _load_checkpoint(resume_checkpoint, map_location=device)
        if resume_strict and _config_summary(checkpoint.get("config", {})) != _config_summary(config.to_dict()):
            raise ValueError("Checkpoint config does not match current config.")
        if int(checkpoint.get("step", 0)) > config.steps:
            raise ValueError("Checkpoint has already advanced beyond the requested step budget.")
        generator.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_loss = float(checkpoint.get("best_loss", best_loss))
        best_decoded = checkpoint.get("best_decoded", best_decoded)
        history = list(checkpoint.get("history", []))
        start_step = int(checkpoint.get("step", 0))
        elapsed_offset = float(checkpoint.get("elapsed_seconds", 0.0))
        if "rng_state" in checkpoint:
            _restore_rng_state_dict(checkpoint["rng_state"])
        print("Resumed from checkpoint %s at step %d" % (resume_checkpoint, start_step), flush=True)
    print(
        "Starting Transformer fit: triangles=%d steps=%d work_size=%dx%d device=%s perceptual=%s pretrained_backbone=%s freeze_backbone=%s"
        % (
            config.num_triangles,
            config.steps,
            work_width,
            work_height,
            device,
            loss_module.perceptual_mode,
            str(config.pretrained_backbone).lower(),
            str(config.freeze_backbone).lower(),
        ),
        flush=True,
    )

    for step in range(start_step, config.steps):
        one_based_step = step + 1
        decoded = generator(target_work)
        render = render_parameters(
            decoded=decoded,
            height=work_height,
            width=work_width,
            mask_temperature=config.mask_temperature,
            background_rgb=(1.0, 1.0, 1.0),
            hard_edges=False,
            active_base_count=base_count,
            active_texture_count=texture_count,
            active_accent_count=accent_count,
        )
        loss, metrics, _ = loss_module(
            render.image,
            target_work,
            render.decoded,
            loss_weights=loss_weights,
            weighted_l1_edge_gain=0.0,
            enable_lpips=config.lpips_weight > 0.0,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((parameter for parameter in generator.parameters() if parameter.requires_grad), max_norm=1.0)
        optimizer.step()

        current_loss = float(loss.detach().cpu().item())
        if current_loss < best_loss:
            best_loss = current_loss
            best_decoded = _decoded_to_cpu(render.decoded)

        if one_based_step == 1 or one_based_step % config.metrics_every == 0 or one_based_step == config.steps:
            entry: Dict[str, object] = {
                "step": one_based_step,
                "active_base_ellipses": base_count,
                "active_texture_ellipses": texture_count,
                "active_accent_ellipses": accent_count,
                "mask_temperature": config.mask_temperature,
            }
            entry.update(flatten_metrics(metrics))
            history.append(entry)
            elapsed = elapsed_offset + (time.perf_counter() - start_time)
            eta = (elapsed / float(one_based_step)) * max(0, config.steps - one_based_step)
            print(
                "[step %d/%d] loss=%.6f l1=%.6f lpips=%.6f rmse=%.6f elapsed=%.1fs eta=%.1fs"
                % (one_based_step, config.steps, entry["loss"], entry["l1"], entry["lpips"], entry["rgb_rmse"], elapsed, eta),
                flush=True,
            )

        if one_based_step % config.progress_every == 0 or one_based_step == config.steps:
            progress_render = _render_export_image(
                decoded=render.decoded,
                height=work_height,
                width=work_width,
                background_rgb=(1.0, 1.0, 1.0),
                active_base_count=base_count,
                active_texture_count=texture_count,
                active_accent_count=accent_count,
            )
            save_image(progress_render, progress_dir / ("step_%04d.png" % one_based_step))

        checkpoint_every = config.checkpoint_every
        if checkpoint_every > 0 and (one_based_step % checkpoint_every == 0 or one_based_step == config.steps):
            _save_checkpoint(
                output_dir / "checkpoints" / ("step_%06d.pt" % one_based_step),
                step=one_based_step,
                config=config,
                model=generator,
                optimizer=optimizer,
                best_loss=best_loss,
                best_decoded=best_decoded,
                history=history,
                elapsed_seconds=elapsed_offset + (time.perf_counter() - start_time),
            )

    final_decoded = generator(target_work)
    final_render = _render_export_image(
        decoded=final_decoded,
        height=work_height,
        width=work_width,
        background_rgb=(1.0, 1.0, 1.0),
        active_base_count=base_count,
        active_texture_count=texture_count,
        active_accent_count=accent_count,
    )
    save_image(final_render, output_dir / "final.png")
    save_image(final_render, output_dir / "final_work.png")
    final_fullres = _render_export_image(
        decoded=final_decoded,
        height=full_height,
        width=full_width,
        background_rgb=(1.0, 1.0, 1.0),
        active_base_count=base_count,
        active_texture_count=texture_count,
        active_accent_count=accent_count,
    )
    save_image(final_fullres, output_dir / "final_fullres.png")

    if best_decoded is not None:
        best_render = _render_export_image(
            decoded=best_decoded,
            height=work_height,
            width=work_width,
            background_rgb=(1.0, 1.0, 1.0),
            active_base_count=base_count,
            active_texture_count=texture_count,
            active_accent_count=accent_count,
        )
        save_image(best_render, output_dir / "best.png")

    _export_triangles_json(
        decoded=final_decoded,
        output_path=output_dir / "ellipses.json",
        image_width=full_width,
        image_height=full_height,
        config=config,
        perceptual_mode=loss_module.perceptual_mode,
    )
    serialize_json(
        {
            "config": config.to_dict(),
            "source_image": loaded.path,
            "image_size": {"width": full_width, "height": full_height},
            "working_size": {"width": work_width, "height": work_height},
            "perceptual_mode": loss_module.perceptual_mode,
            "history": history,
            "final": history[-1] if history else {},
            "best_loss": best_loss,
        },
        output_dir / "metrics.json",
    )
    total_elapsed = elapsed_offset + (time.perf_counter() - start_time)
    print("Completed Transformer fit in %.1fs" % total_elapsed, flush=True)

    return TransformerFitArtifacts(
        final_image_path=str(output_dir / "final.png"),
        ellipses_path=str(output_dir / "ellipses.json"),
        metrics_path=str(output_dir / "metrics.json"),
        progress_dir=str(progress_dir),
    )


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    seed = resolve_seed(args.seed)
    device = resolve_device(args.device)
    config = TransformerTriangleConfig(
        num_triangles=args.num_triangles,
        work_size=args.work_size,
        steps=args.steps,
        seed=seed,
        lr=args.lr,
        lpips_weight=0.0 if args.fast else 0.3,
        pretrained_backbone=not args.no_pretrained,
        freeze_backbone=not args.finetune_backbone,
        checkpoint_every=args.checkpoint_every if args.checkpoint_every is not None else 500,
    )
    print("Using seed=%d device=%s" % (seed, device))
    artifacts = fit_image_file(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        config=config,
        device=device,
        resume_checkpoint=Path(args.resume) if args.resume else None,
        resume_strict=args.resume_strict,
    )
    print("Saved final image to %s" % artifacts.final_image_path)
    print("Saved triangle parameters to %s" % artifacts.ellipses_path)
    print("Saved metrics to %s" % artifacts.metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


