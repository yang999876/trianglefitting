from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, List, Optional

import torch

from .config import FitConfig, StageConfig, StageLossConfig
from .initialize import TargetAnalysis, build_residual_components, initialize_ellipse_table, reinitialize_table_slice_from_residual
from .io import LoadedImage, load_image, save_image
from .losses import CompositeEllipseLoss
from .model import EllipseParameterTable, EllipseParameters
from .renderer import RenderResult, render_parameters, render_parameters_image_only, render_table
from .utils import ensure_dir, flatten_metrics, serialize_json, seed_all


def _sync_device_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class FitArtifacts:
    final_image_path: str
    ellipses_path: str
    metrics_path: str
    progress_dir: str


def _configure_optimizer(table: EllipseParameterTable, stage: StageConfig) -> torch.optim.Optimizer:
    table.apply_stage(stage)
    groups = table.optimizer_groups(stage)
    if not groups:
        raise RuntimeError("No trainable parameters configured for stage %s" % stage.name)
    return torch.optim.Adam(groups)


def _assert_finite(table: EllipseParameterTable) -> None:
    for name, parameter in table.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError("Encountered non-finite parameter values in %s" % name)


def _decoded_to_cpu(decoded: EllipseParameters) -> EllipseParameters:
    return EllipseParameters(
        centers=decoded.centers.detach().cpu(),
        sizes=decoded.sizes.detach().cpu(),
        theta=decoded.theta.detach().cpu(),
        rgb=decoded.rgb.detach().cpu(),
        alpha=decoded.alpha.detach().cpu(),
        role_ids=decoded.role_ids.detach().cpu(),
        background_grid_rgb=None if decoded.background_grid_rgb is None else decoded.background_grid_rgb.detach().cpu(),
    )


def _background_grid_to_json(decoded: EllipseParameters) -> Optional[List[List[List[float]]]]:
    if decoded.background_grid_rgb is None:
        return None
    grid = decoded.background_grid_rgb.detach().cpu()
    if grid.ndim == 4:
        grid = grid.squeeze(0).permute(1, 2, 0)
    return grid.tolist()


def _render_export_image(
    decoded: EllipseParameters,
    height: int,
    width: int,
    background_rgb,
    active_base_count: int,
    active_texture_count: int,
    active_accent_count: int,
) -> torch.Tensor:
    cpu_decoded = _decoded_to_cpu(decoded)
    with torch.no_grad():
        return render_parameters_image_only(
            decoded=cpu_decoded,
            height=height,
            width=width,
            background_rgb=background_rgb,
            active_base_count=active_base_count,
            active_texture_count=active_texture_count,
            active_accent_count=active_accent_count,
        )


def _current_decoded(
    table: EllipseParameterTable,
    height: int,
    width: int,
    active_base_count: int,
    active_texture_count: int,
    active_accent_count: int,
) -> EllipseParameters:
    min_size = 1.0 / float(max(height, width))
    return table.decode(
        min_size=min_size,
        active_base_count=active_base_count,
        active_texture_count=active_texture_count,
        active_accent_count=active_accent_count,
    )


def _coverage_stats(coverage_map: torch.Tensor) -> Dict[str, float]:
    flat = coverage_map.detach().reshape(-1)
    return {
        "coverage_mean": float(flat.mean().item()),
        "coverage_p95": float(torch.quantile(flat, 0.95).item()),
    }


def _background_and_base_rmse(render: RenderResult, target: torch.Tensor) -> Dict[str, float]:
    background_only = torch.sqrt(torch.mean((render.background_image - target) ** 2))
    background_plus_base = torch.sqrt(torch.mean((render.background_plus_base_image - target) ** 2))
    return {
        "background_only_rgb_rmse": float(background_only.item()),
        "background_plus_base_rgb_rmse": float(background_plus_base.item()),
    }


def _residual_keys_for(kind: str) -> Dict[str, str]:
    return {
        "distribution": "%s_distribution" % kind,
        "residual_map": "%s_residual_map" % kind,
    }


def _grow_new_role(
    table: EllipseParameterTable,
    target_work: torch.Tensor,
    target_analysis: TargetAnalysis,
    current_hard_render: RenderResult,
    kind: str,
    previous_count: int,
    active_count: int,
    config: FitConfig,
    seed: int,
) -> Optional[Dict[str, object]]:
    if active_count <= previous_count:
        return None
    residuals = build_residual_components(
        target=target_work,
        current_image=current_hard_render.image,
        coverage_map=current_hard_render.coverage_map,
        target_analysis=target_analysis,
        coverage_penalty_strength=config.coverage_penalty_strength,
    )
    start, _ = table.role_range(kind, previous_count)
    _, end = table.role_range(kind, active_count)
    residual_keys = _residual_keys_for(kind)
    reinitialize_table_slice_from_residual(
        table=table,
        start=start,
        end=end,
        target=target_work,
        distribution=residuals[residual_keys["distribution"]],
        residual_map=residuals[residual_keys["residual_map"]],
        target_analysis=target_analysis,
        seed=seed,
        kind=kind,
    )
    return {"type": "grow", "kind": kind, "start": start, "end": end}


def _lowest_contribution_indices(
    decoded: EllipseParameters,
    masks: torch.Tensor,
    residual_map: torch.Tensor,
    start: int,
    end: int,
    count: int,
) -> List[int]:
    if end <= start:
        return []
    residual_2d = residual_map.squeeze(0).squeeze(0)
    contributions = []
    for index in range(start, end):
        hard_mask = masks[index]
        contribution = torch.mean(hard_mask * residual_2d) * torch.mean(hard_mask)
        contributions.append(contribution)
    if not contributions:
        return []
    contribution_tensor = torch.stack(contributions, dim=0)
    count = max(1, min(count, int(contribution_tensor.shape[0])))
    return (torch.argsort(contribution_tensor, descending=False)[:count] + start).detach().cpu().tolist()


def _rebirth_role(
    table: EllipseParameterTable,
    target_work: torch.Tensor,
    target_analysis: TargetAnalysis,
    current_hard_render: RenderResult,
    kind: str,
    active_count: int,
    config: FitConfig,
    seed: int,
) -> List[int]:
    if active_count <= 0:
        return []
    residuals = build_residual_components(
        target=target_work,
        current_image=current_hard_render.image,
        coverage_map=current_hard_render.coverage_map,
        target_analysis=target_analysis,
        coverage_penalty_strength=config.coverage_penalty_strength,
    )
    start, end = table.role_range(kind, active_count)
    residual_keys = _residual_keys_for(kind)
    rebirth_count = max(
        config.rebirth_min_count,
        min(config.rebirth_max_count, int(round(active_count * config.rebirth_fraction))),
    )
    loser_indices = _lowest_contribution_indices(
        current_hard_render.decoded,
        current_hard_render.masks,
        residuals[residual_keys["residual_map"]],
        start,
        end,
        rebirth_count,
    )
    for offset, index in enumerate(loser_indices):
        reinitialize_table_slice_from_residual(
            table=table,
            start=index,
            end=index + 1,
            target=target_work,
            distribution=residuals[residual_keys["distribution"]],
            residual_map=residuals[residual_keys["residual_map"]],
            target_analysis=target_analysis,
            seed=seed + (offset * 19) + (index * 7),
            kind=kind,
        )
    return loser_indices


def export_ellipses_json(
    table: EllipseParameterTable,
    output_path: Path,
    image_size: Dict[str, int],
    config: FitConfig,
    final_temperature: float,
    perceptual_mode: str,
) -> None:
    decoded = table.decode(
        min_size=1e-8,
        active_base_count=table.base_count,
        active_texture_count=table.texture_count(),
        active_accent_count=table.accent_count(),
    )
    payload = {
        "image_width": image_size["width"],
        "image_height": image_size["height"],
        "background_rgb": list(config.background_rgb),
        "background_grid_height": int(decoded.background_grid_rgb.shape[-2]) if decoded.background_grid_rgb is not None else 0,
        "background_grid_width": int(decoded.background_grid_rgb.shape[-1]) if decoded.background_grid_rgb is not None else 0,
        "background_grid_rgb": _background_grid_to_json(decoded),
        "base_ellipse_count": table.base_count,
        "texture_ellipse_count": table.texture_count(),
        "accent_ellipse_count": table.accent_count(),
        "num_ellipses": config.num_ellipses,
        "seed": config.seed,
        "final_temperature": final_temperature,
        "export_render_mode": "hard_edges",
        "perceptual_mode": perceptual_mode,
        "ellipses": decoded.to_json_list(),
    }
    serialize_json(payload, output_path)


def fit_loaded_image(
    loaded: LoadedImage,
    output_dir: Path,
    config: FitConfig,
    device: torch.device,
) -> FitArtifacts:
    seed_all(config.seed)
    ensure_dir(output_dir)
    progress_dir = ensure_dir(output_dir / "progress")

    target_work = loaded.working.to(device=device, dtype=torch.float32)
    _, _, work_height, work_width = target_work.shape
    full_height = loaded.original.shape[-2]
    full_width = loaded.original.shape[-1]
    save_image(loaded.working, output_dir / "target_work.png")

    table, target_analysis = initialize_ellipse_table(target_work, config)
    table = table.to(device=device)
    target_analysis = TargetAnalysis(
        edges=target_analysis.edges.to(device=device),
        coherence=target_analysis.coherence.to(device=device),
        tangent_theta=target_analysis.tangent_theta.to(device=device),
    )
    perceptual_enabled = any(stage_loss.weights.lpips > 0.0 for stage_loss in config.stage_loss_schedule)
    edge_loss_enabled = any(stage_loss.weights.edge_l1 > 0.0 or stage_loss.weighted_l1_edge_gain > 0.0 for stage_loss in config.stage_loss_schedule)
    band_loss_enabled = any(stage_loss.weights.band_l1 > 0.0 for stage_loss in config.stage_loss_schedule)
    loss_module = CompositeEllipseLoss(
        area_regularization_weight=config.area_regularization_weight,
        area_regularizer_power=config.area_regularizer_power,
        large_area_threshold=config.large_area_threshold,
        large_area_penalty_scale=config.large_area_penalty_scale,
        base_area_penalty_scale=config.base_area_penalty_scale,
        texture_area_penalty_scale=config.texture_area_penalty_scale,
        accent_area_penalty_scale=config.accent_area_penalty_scale,
        perceptual_enabled=perceptual_enabled,
        edge_loss_enabled=edge_loss_enabled,
        band_loss_enabled=band_loss_enabled,
    ).to(device=device)
    loss_module.set_target(target_work)

    metrics_history: List[Dict[str, object]] = []
    event_history: List[Dict[str, object]] = []
    current_stage: Optional[StageConfig] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    previous_active_base_count = config.active_base_ellipse_count(0)
    previous_active_texture_count = config.active_texture_ellipse_count(0)
    previous_active_accent_count = config.active_accent_ellipse_count(0)
    best_decoded: Optional[EllipseParameters] = None
    best_base_count = 0
    best_texture_count = 0
    best_accent_count = 0
    best_temperature = 0.0
    best_step = 0
    best_hard_loss = float("inf")
    best_hard_rgb_rmse = float("inf")
    best_train_loss = float("inf")
    last_loss_improvement_step = 0
    last_grow_step = 0
    last_rebirth_step = 0
    start_time = time.perf_counter()
    profile_totals: Dict[str, float] = defaultdict(float)
    profile_counts: Dict[str, int] = defaultdict(int)

    print(
        "Starting fit: ellipses=%d steps=%d work_size=%dx%d device=%s mode=%s perceptual=%s"
        % (
            config.num_ellipses,
            config.steps,
            work_width,
            work_height,
            device,
            config.optimization_mode,
            loss_module.perceptual_mode,
        ),
        flush=True,
    )

    for step in range(config.steps):
        step_started = time.perf_counter()
        stage = config.get_stage(step)
        stage_loss = config.get_stage_loss(stage.name)
        active_base_count = config.active_base_ellipse_count(step)
        active_texture_count = config.active_texture_ellipse_count(step)
        active_accent_count = config.active_accent_ellipse_count(step)
        active_total = active_base_count + active_texture_count + active_accent_count
        one_based_step = step + 1
        progress_fraction = float(step + 1) / float(max(1, config.steps))
        allow_grow = False
        allow_rebirth = active_texture_count > 0 and one_based_step >= config.plateau_rebirth_start_step
        allow_late_texture_rebirth = False
        stabilize_mode = progress_fraction > 0.50
        grow_events: List[Dict[str, object]] = []
        rebirth_events: List[Dict[str, object]] = []
        growth_due = False
        plateau_rebirth_due = False

        if current_stage != stage:
            current_stage = stage
            if config.profile_enabled:
                _sync_device_if_needed(device)
            stage_switch_started = time.perf_counter()
            optimizer = _configure_optimizer(table, stage)
            if config.profile_enabled:
                _sync_device_if_needed(device)
                profile_totals["stage_switch_seconds"] += time.perf_counter() - stage_switch_started
                profile_counts["stage_switch_count"] += 1
            print(
                "[step %d/%d] stage=%s range=%.1f%%-%.1f%% active_base=%d/%d active_texture=%d/%d active_accent=%d/%d mask_temperature=%.4f geometry_lr=%.4g color_lr=%.4g"
                % (
                    step + 1,
                    config.steps,
                    current_stage.name,
                    current_stage.start_fraction * 100.0,
                    current_stage.end_fraction * 100.0,
                    active_base_count,
                    table.base_count,
                    active_texture_count,
                    table.texture_count(),
                    active_accent_count,
                    table.accent_count(),
                    current_stage.mask_temperature,
                    current_stage.geometry_lr,
                    current_stage.color_lr,
                ),
                flush=True,
            )
            print(
                "  schedule: allow_grow=%s allow_rebirth=%s allow_late_texture_rebirth=%s stabilize=%s"
                % (
                    str(allow_grow).lower(),
                    str(allow_rebirth).lower(),
                    str(allow_late_texture_rebirth).lower(),
                    str(stabilize_mode).lower(),
                ),
                flush=True,
            )

        assert optimizer is not None
        current_hard_before: Optional[RenderResult] = None
        if growth_due:
            if config.profile_enabled:
                _sync_device_if_needed(device)
            grow_prepare_started = time.perf_counter()
            current_decoded_before = _current_decoded(
                table=table,
                height=work_height,
                width=work_width,
                active_base_count=previous_active_base_count,
                active_texture_count=previous_active_texture_count,
                active_accent_count=previous_active_accent_count,
            )
            current_hard_before = render_parameters(
                decoded=current_decoded_before,
                height=work_height,
                width=work_width,
                mask_temperature=1e-3,
                background_rgb=config.background_rgb,
                hard_edges=True,
                active_base_count=previous_active_base_count,
                active_texture_count=previous_active_texture_count,
                active_accent_count=previous_active_accent_count,
            )
            if config.profile_enabled:
                _sync_device_if_needed(device)
                profile_totals["grow_prepare_seconds"] += time.perf_counter() - grow_prepare_started
                profile_counts["grow_prepare_count"] += 1

        if growth_due and current_hard_before is not None:
            if config.profile_enabled:
                _sync_device_if_needed(device)
            grow_started = time.perf_counter()
            for kind, previous_count, active_count, seed_offset in (
                ("base", previous_active_base_count, active_base_count, 1),
                ("texture", previous_active_texture_count, active_texture_count, 11),
                ("accent", previous_active_accent_count, active_accent_count, 23),
            ):
                if active_count > previous_count:
                    grow_event = _grow_new_role(
                        table=table,
                        target_work=target_work,
                        target_analysis=target_analysis,
                        current_hard_render=current_hard_before,
                        kind=kind,
                        previous_count=previous_count,
                        active_count=active_count,
                        config=config,
                        seed=config.seed + (step * 101) + seed_offset,
                    )
                    if grow_event is not None:
                        grow_events.append(grow_event)
                        event_history.append({"step": step + 1, **grow_event})
                        last_grow_step = step + 1
            if config.profile_enabled:
                _sync_device_if_needed(device)
                profile_totals["grow_reinit_seconds"] += time.perf_counter() - grow_started
                profile_counts["grow_reinit_count"] += 1
        previous_active_base_count = active_base_count
        previous_active_texture_count = active_texture_count
        previous_active_accent_count = active_accent_count

        if config.profile_enabled:
            _sync_device_if_needed(device)
        render_started = time.perf_counter()
        render = render_table(
            table=table,
            height=work_height,
            width=work_width,
            mask_temperature=current_stage.mask_temperature,
            background_rgb=config.background_rgb,
            hard_edges=False,
            active_base_count=active_base_count,
            active_texture_count=active_texture_count,
            active_accent_count=active_accent_count,
        )
        if config.profile_enabled:
            _sync_device_if_needed(device)
            profile_totals["train_render_seconds"] += time.perf_counter() - render_started
            profile_counts["train_render_count"] += 1

        if config.profile_enabled:
            _sync_device_if_needed(device)
        loss_started = time.perf_counter()
        enable_lpips = config.use_lpips_on_step(stage.name, step)
        loss, metrics, loss_timings = loss_module(
            render.image,
            target_work,
            render.decoded,
            loss_weights=stage_loss.weights,
            weighted_l1_edge_gain=stage_loss.weighted_l1_edge_gain,
            enable_lpips=enable_lpips,
            collect_timing=config.profile_enabled,
        )
        if config.profile_enabled:
            _sync_device_if_needed(device)
            profile_totals["train_loss_seconds"] += time.perf_counter() - loss_started
            profile_counts["train_loss_count"] += 1
            for key, value in loss_timings.items():
                profile_totals[key] += value
                profile_counts[key.replace("_seconds", "_count")] += 1
        current_loss_value = float(loss.detach().item())
        if current_loss_value + config.plateau_rebirth_min_delta < best_train_loss:
            best_train_loss = current_loss_value
            last_loss_improvement_step = one_based_step

        if config.profile_enabled:
            _sync_device_if_needed(device)
        zero_grad_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if config.profile_enabled:
            _sync_device_if_needed(device)
            profile_totals["train_zero_grad_seconds"] += time.perf_counter() - zero_grad_started
            profile_counts["train_zero_grad_count"] += 1

        if config.profile_enabled:
            _sync_device_if_needed(device)
        backward_started = time.perf_counter()
        loss.backward()
        if config.profile_enabled:
            _sync_device_if_needed(device)
            profile_totals["train_backward_seconds"] += time.perf_counter() - backward_started
            profile_counts["train_backward_count"] += 1

        if config.profile_enabled:
            _sync_device_if_needed(device)
        clip_started = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(list(table.trainable_parameters()), max_norm=1.0)
        if config.profile_enabled:
            _sync_device_if_needed(device)
            profile_totals["train_clip_grad_seconds"] += time.perf_counter() - clip_started
            profile_counts["train_clip_grad_count"] += 1

        if config.profile_enabled:
            _sync_device_if_needed(device)
        optimizer_started = time.perf_counter()
        optimizer.step()
        if config.profile_enabled:
            _sync_device_if_needed(device)
            profile_totals["train_optimizer_step_seconds"] += time.perf_counter() - optimizer_started
            profile_counts["train_optimizer_step_count"] += 1

        if config.profile_enabled:
            _sync_device_if_needed(device)
        finite_check_started = time.perf_counter()
        _assert_finite(table)
        if config.profile_enabled:
            _sync_device_if_needed(device)
            profile_totals["train_finite_check_seconds"] += time.perf_counter() - finite_check_started
            profile_counts["train_finite_check_count"] += 1
        plateau_rebirth_due = (
            allow_rebirth
            and (one_based_step - last_loss_improvement_step) >= config.plateau_rebirth_window
            and (one_based_step - last_rebirth_step) >= config.plateau_rebirth_cooldown
        )
        if plateau_rebirth_due:
            if config.profile_enabled:
                _sync_device_if_needed(device)
            rebirth_render_started = time.perf_counter()
            current_hard_after_grow = render_table(
                table=table,
                height=work_height,
                width=work_width,
                mask_temperature=current_stage.mask_temperature,
                background_rgb=config.background_rgb,
                hard_edges=True,
                active_base_count=active_base_count,
                active_texture_count=active_texture_count,
                active_accent_count=active_accent_count,
            )
            if config.profile_enabled:
                _sync_device_if_needed(device)
                profile_totals["rebirth_render_seconds"] += time.perf_counter() - rebirth_render_started
                profile_counts["rebirth_render_count"] += 1
            if config.profile_enabled:
                _sync_device_if_needed(device)
            rebirth_started = time.perf_counter()
            reborn = _rebirth_role(
                table=table,
                target_work=target_work,
                target_analysis=target_analysis,
                current_hard_render=current_hard_after_grow,
                kind="texture",
                active_count=active_texture_count,
                config=config,
                seed=config.seed + (step * 151) + 17,
            )
            if reborn:
                rebirth_events.append({"type": "plateau_rebirth", "kind": "texture", "indices": reborn})
                event_history.append({"step": one_based_step, "type": "plateau_rebirth", "kind": "texture", "indices": reborn})
                last_rebirth_step = one_based_step
                last_loss_improvement_step = one_based_step
            if config.profile_enabled:
                _sync_device_if_needed(device)
                profile_totals["rebirth_reinit_seconds"] += time.perf_counter() - rebirth_started
                profile_counts["rebirth_reinit_count"] += 1

        if config.profile_enabled:
            _sync_device_if_needed(device)
            profile_totals["train_step_wall_seconds"] += time.perf_counter() - step_started
            profile_counts["train_step_wall_count"] += 1

        if one_based_step % config.metrics_every == 0 or one_based_step == 1 or one_based_step == config.steps:
            if config.profile_enabled:
                _sync_device_if_needed(device)
            metrics_started = time.perf_counter()
            history_entry = {
                "step": one_based_step,
                "stage": current_stage.name,
                "active_base_ellipses": active_base_count,
                "active_texture_ellipses": active_texture_count,
                "active_accent_ellipses": active_accent_count,
                "active_ellipses": active_total,
                "mask_temperature": current_stage.mask_temperature,
                "lpips_enabled": enable_lpips,
            }
            history_entry.update(flatten_metrics(metrics))
            history_entry.update(_coverage_stats(render.coverage_map))
            history_entry.update(_background_and_base_rmse(render, target_work))
            history_entry["hard_loss"] = history_entry["loss"]
            history_entry["hard_l1"] = history_entry["l1"]
            history_entry["hard_band_l1"] = history_entry["band_l1"]
            history_entry["hard_ms_ssim"] = history_entry["ms_ssim"]
            history_entry["hard_lpips"] = history_entry["lpips"]
            history_entry["hard_edge_l1"] = history_entry["edge_l1"]
            history_entry["hard_rgb_rmse"] = history_entry["rgb_rmse"]
            history_entry["hard_bandpass_rmse"] = history_entry["bandpass_rmse"]
            if (
                history_entry["hard_loss"] < best_hard_loss
                or (
                    abs(history_entry["hard_loss"] - best_hard_loss) < 1e-8
                    and history_entry["hard_rgb_rmse"] < best_hard_rgb_rmse
                )
            ):
                best_hard_loss = float(history_entry["hard_loss"])
                best_hard_rgb_rmse = float(history_entry["hard_rgb_rmse"])
                best_step = one_based_step
                best_decoded = _decoded_to_cpu(render.decoded)
                best_base_count = active_base_count
                best_texture_count = active_texture_count
                best_accent_count = active_accent_count
                best_temperature = current_stage.mask_temperature
            history_entry["soft_hard_loss_gap"] = 0.0
            history_entry["soft_hard_rmse_gap"] = 0.0
            history_entry["allow_grow"] = allow_grow
            history_entry["allow_rebirth"] = allow_rebirth
            history_entry["allow_late_texture_rebirth"] = allow_late_texture_rebirth
            history_entry["plateau_rebirth_due"] = plateau_rebirth_due
            history_entry["best_train_loss"] = best_train_loss
            history_entry["last_loss_improvement_step"] = last_loss_improvement_step
            history_entry["stabilize_mode"] = stabilize_mode
            if grow_events:
                history_entry["grow_events"] = grow_events
            if rebirth_events:
                history_entry["rebirth_events"] = rebirth_events
            metrics_history.append(history_entry)
            elapsed = time.perf_counter() - start_time
            average_step_time = elapsed / float(one_based_step)
            eta_seconds = average_step_time * max(0, config.steps - one_based_step)
            if config.profile_enabled:
                _sync_device_if_needed(device)
                profile_totals["metrics_seconds"] += time.perf_counter() - metrics_started
                profile_counts["metrics_count"] += 1
            print(
                "[step %d/%d] active_base=%d/%d active_texture=%d/%d active_accent=%d/%d loss=%.6f l1=%.6f rmse=%.6f coverage_mean=%.3f elapsed=%.1fs eta=%.1fs"
                % (
                    one_based_step,
                    config.steps,
                    active_base_count,
                    table.base_count,
                    active_texture_count,
                    table.texture_count(),
                    active_accent_count,
                    table.accent_count(),
                    history_entry["loss"],
                    history_entry["l1"],
                    history_entry["rgb_rmse"],
                    history_entry["coverage_mean"],
                    elapsed,
                    eta_seconds,
                ),
                flush=True,
            )
            for grow_event in grow_events:
                print(
                    "  grow event: %s ellipses [%d, %d)"
                    % (grow_event["kind"], grow_event["start"], grow_event["end"]),
                    flush=True,
                )
            for rebirth_event in rebirth_events:
                print(
                    "  rebirth event: %s recycled %d ellipses %s"
                    % (rebirth_event["kind"], len(rebirth_event["indices"]), rebirth_event["indices"][:8]),
                    flush=True,
                )
            if config.profile_enabled:
                print(
                    "  profile avg/step: render=%.4fs loss=%.4fs backward=%.4fs optimizer=%.4fs zero_grad=%.4fs clip=%.4fs finite=%.4fs grow_prep=%.4fs grow=%.4fs rebirth_render=%.4fs rebirth=%.4fs metrics=%.4fs total=%.4fs"
                    % (
                        profile_totals["train_render_seconds"] / max(1, profile_counts["train_render_count"]),
                        profile_totals["train_loss_seconds"] / max(1, profile_counts["train_loss_count"]),
                        profile_totals["train_backward_seconds"] / max(1, profile_counts["train_backward_count"]),
                        profile_totals["train_optimizer_step_seconds"] / max(1, profile_counts["train_optimizer_step_count"]),
                        profile_totals["train_zero_grad_seconds"] / max(1, profile_counts["train_zero_grad_count"]),
                        profile_totals["train_clip_grad_seconds"] / max(1, profile_counts["train_clip_grad_count"]),
                        profile_totals["train_finite_check_seconds"] / max(1, profile_counts["train_finite_check_count"]),
                        profile_totals["grow_prepare_seconds"] / max(1, profile_counts["grow_prepare_count"]),
                        profile_totals["grow_reinit_seconds"] / max(1, profile_counts["grow_reinit_count"]),
                        profile_totals["rebirth_render_seconds"] / max(1, profile_counts["rebirth_render_count"]),
                        profile_totals["rebirth_reinit_seconds"] / max(1, profile_counts["rebirth_reinit_count"]),
                        profile_totals["metrics_seconds"] / max(1, profile_counts["metrics_count"]),
                        profile_totals["train_step_wall_seconds"] / max(1, profile_counts["train_step_wall_count"]),
                    ),
                    flush=True,
                )
                print(
                    "  profile loss avg: l1=%.4fs blur=%.4fs band=%.4fs ms_ssim=%.4fs lpips=%.4fs edge=%.4fs reg=%.4fs"
                    % (
                        profile_totals["loss_l1_seconds"] / max(1, profile_counts["loss_l1_count"]),
                        profile_totals["loss_blur_l1_seconds"] / max(1, profile_counts["loss_blur_l1_count"]),
                        profile_totals["loss_band_seconds"] / max(1, profile_counts["loss_band_count"]),
                        profile_totals["loss_ms_ssim_seconds"] / max(1, profile_counts["loss_ms_ssim_count"]),
                        profile_totals["loss_lpips_seconds"] / max(1, profile_counts["loss_lpips_count"]),
                        profile_totals["loss_edge_seconds"] / max(1, profile_counts["loss_edge_count"]),
                        profile_totals["loss_regularizer_seconds"] / max(1, profile_counts["loss_regularizer_count"]),
                    ),
                    flush=True,
                )

        if one_based_step % config.progress_every == 0 or one_based_step == config.steps:
            if config.profile_enabled:
                _sync_device_if_needed(device)
            export_started = time.perf_counter()
            progress_render = _render_export_image(
                decoded=render.decoded,
                height=work_height,
                width=work_width,
                background_rgb=config.background_rgb,
                active_base_count=active_base_count,
                active_texture_count=active_texture_count,
                active_accent_count=active_accent_count,
            )
            save_image(progress_render, progress_dir / ("step_%04d.png" % one_based_step))
            if config.profile_enabled:
                _sync_device_if_needed(device)
                profile_totals["export_progress_seconds"] += time.perf_counter() - export_started
                profile_counts["export_progress_count"] += 1
            print("[step %d/%d] saved progress image to %s" % (one_based_step, config.steps, progress_dir / ("step_%04d.png" % one_based_step)), flush=True)

    final_stage = config.get_stage(config.steps - 1)
    final_decoded = table.decode(
        min_size=1.0 / float(max(work_height, work_width)),
        active_base_count=table.base_count,
        active_texture_count=table.texture_count(),
        active_accent_count=table.accent_count(),
    )
    if config.profile_enabled:
        _sync_device_if_needed(device)
    final_export_started = time.perf_counter()
    final_render = _render_export_image(
        decoded=final_decoded,
        height=work_height,
        width=work_width,
        background_rgb=config.background_rgb,
        active_base_count=table.base_count,
        active_texture_count=table.texture_count(),
        active_accent_count=table.accent_count(),
    )
    save_image(final_render, output_dir / "final.png")
    save_image(final_render, output_dir / "final_work.png")
    final_fullres_render = _render_export_image(
        decoded=final_decoded,
        height=full_height,
        width=full_width,
        background_rgb=config.background_rgb,
        active_base_count=table.base_count,
        active_texture_count=table.texture_count(),
        active_accent_count=table.accent_count(),
    )
    save_image(final_fullres_render, output_dir / "final_fullres.png")
    if config.profile_enabled:
        _sync_device_if_needed(device)
        profile_totals["export_final_seconds"] += time.perf_counter() - final_export_started
        profile_counts["export_final_count"] += 1

    if best_decoded is not None:
        best_work_render = _render_export_image(
            decoded=best_decoded,
            height=work_height,
            width=work_width,
            background_rgb=config.background_rgb,
            active_base_count=best_base_count,
            active_texture_count=best_texture_count,
            active_accent_count=best_accent_count,
        )
        save_image(best_work_render, output_dir / "best.png")
        save_image(best_work_render, output_dir / "best_work.png")
        best_fullres_render = _render_export_image(
            decoded=best_decoded,
            height=full_height,
            width=full_width,
            background_rgb=config.background_rgb,
            active_base_count=best_base_count,
            active_texture_count=best_texture_count,
            active_accent_count=best_accent_count,
        )
        save_image(best_fullres_render, output_dir / "best_fullres.png")

    if config.profile_enabled:
        _sync_device_if_needed(device)
    params_export_started = time.perf_counter()
    export_ellipses_json(
        table=table.cpu(),
        output_path=output_dir / "ellipses.json",
        image_size={"width": full_width, "height": full_height},
        config=config,
        final_temperature=final_stage.mask_temperature,
        perceptual_mode=loss_module.perceptual_mode,
    )
    if config.profile_enabled:
        _sync_device_if_needed(device)
        profile_totals["export_params_json_seconds"] += time.perf_counter() - params_export_started
        profile_counts["export_params_json_count"] += 1

    if config.profile_enabled:
        _sync_device_if_needed(device)
    metrics_export_started = time.perf_counter()
    serialize_json(
        {
            "config": config.to_dict(),
            "source_image": loaded.path,
            "image_size": {"width": full_width, "height": full_height},
            "working_size": {"width": work_width, "height": work_height},
            "perceptual_mode": loss_module.perceptual_mode,
            "history": metrics_history,
            "events": event_history,
            "final": metrics_history[-1] if metrics_history else {},
            "best_step": best_step,
            "best_hard_loss": best_hard_loss if best_step else None,
            "best_hard_rgb_rmse": best_hard_rgb_rmse if best_step else None,
            "last_grow_step": last_grow_step,
            "last_rebirth_step": last_rebirth_step,
            "profile_totals_seconds": dict(profile_totals) if config.profile_enabled else {},
            "profile_counts": dict(profile_counts) if config.profile_enabled else {},
            "profile_average_seconds": {
                key.replace("_seconds", "_avg_seconds"): profile_totals[key] / max(1, profile_counts.get(key.replace("_seconds", "_count"), 1))
                for key in profile_totals
                if key.endswith("_seconds")
            } if config.profile_enabled else {},
        },
        output_dir / "metrics.json",
    )
    if config.profile_enabled:
        _sync_device_if_needed(device)
        profile_totals["export_metrics_json_seconds"] += time.perf_counter() - metrics_export_started
        profile_counts["export_metrics_json_count"] += 1

    total_elapsed = time.perf_counter() - start_time
    print("Completed fit in %.1fs" % total_elapsed, flush=True)
    if config.profile_enabled:
        print(
            "Profile summary: render=%.1fs loss=%.1fs backward=%.1fs optimizer=%.1fs progress_export=%.1fs final_export=%.1fs"
            % (
                profile_totals["train_render_seconds"],
                profile_totals["train_loss_seconds"],
                profile_totals["train_backward_seconds"],
                profile_totals["train_optimizer_step_seconds"],
                profile_totals["export_progress_seconds"],
                profile_totals["export_final_seconds"],
            ),
            flush=True,
        )

    return FitArtifacts(
        final_image_path=str(output_dir / "final.png"),
        ellipses_path=str(output_dir / "ellipses.json"),
        metrics_path=str(output_dir / "metrics.json"),
        progress_dir=str(progress_dir),
    )


def fit_image_file(
    input_path: Path,
    output_dir: Path,
    config: FitConfig,
    device: torch.device,
) -> FitArtifacts:
    loaded = load_image(input_path, config.work_size)
    return fit_loaded_image(loaded=loaded, output_dir=output_dir, config=config, device=device)


