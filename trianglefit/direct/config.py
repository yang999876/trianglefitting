from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class LossWeights:
    l1: float = 0.30
    blur_l1: float = 0.0
    band_l1: float = 0.0
    ms_ssim: float = 0.25
    lpips: float = 0.0
    edge_l1: float = 0.10

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class StageConfig:
    name: str
    start_step: int
    end_step: int
    start_fraction: float
    end_fraction: float
    mask_temperature: float
    optimize_centers: bool
    optimize_radii: bool
    optimize_theta: bool
    optimize_color: bool
    geometry_lr: float = 2e-2
    color_lr: float = 1e-2

    def contains(self, step_index: int) -> bool:
        return self.start_step <= step_index < self.end_step

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StageLossConfig:
    name: str
    weights: LossWeights
    weighted_l1_edge_gain: float
    lpips_cadence: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "weights": self.weights.to_dict(),
            "weighted_l1_edge_gain": self.weighted_l1_edge_gain,
            "lpips_cadence": self.lpips_cadence,
        }


def make_default_stage_schedule(total_steps: int) -> Tuple[StageConfig, ...]:
    return (
        StageConfig(
            name="single_stage",
            start_step=0,
            end_step=total_steps,
            start_fraction=0.0,
            end_fraction=1.0,
            mask_temperature=0.02,
            optimize_centers=True,
            optimize_radii=True,
            optimize_theta=True,
            optimize_color=True,
            geometry_lr=3e-3,
            color_lr=2e-3,
        ),
    )


def make_default_stage_loss_schedule(mode: str) -> Tuple[StageLossConfig, ...]:
    return (
        StageLossConfig("single_stage", LossWeights(l1=1.0, blur_l1=0.0, band_l1=0.0, ms_ssim=0.0, lpips=0.3, edge_l1=0.0), 0.0, 1),
    )


@dataclass(frozen=True)
class FitConfig:
    num_ellipses: int = 200
    work_size: int = 256
    steps: int = 8000
    seed: int = 42
    metrics_every: int = 400
    progress_every: int = 400
    area_regularization_weight: float = 0.0
    background_rgb: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    optimization_mode: str = "quality"
    profile_enabled: bool = False
    background_grid_size: Tuple[int, int] = (1, 1)
    base_ellipse_fraction: float = 0.20
    texture_ellipse_fraction: float = 0.80
    base_area_penalty_scale: float = 0.35
    texture_area_penalty_scale: float = 0.90
    accent_area_penalty_scale: float = 0.90
    area_regularizer_power: float = 1.5
    large_area_threshold: float = 0.03
    large_area_penalty_scale: float = 2.0
    coverage_penalty_strength: float = 1.5
    base_growth_schedule: Tuple[Tuple[float, float], ...] = (
        (1.00, 1.00),
    )
    texture_growth_schedule: Tuple[Tuple[float, float], ...] = (
        (1.00, 1.00),
    )
    accent_growth_schedule: Tuple[Tuple[float, float], ...] = (
        (1.00, 0.00),
    )
    rebirth_every: int = 0
    rebirth_fraction: float = 0.05
    rebirth_min_count: int = 2
    rebirth_max_count: int = 20
    plateau_rebirth_window: int = 120
    plateau_rebirth_cooldown: int = 120
    plateau_rebirth_start_step: int = 150
    plateau_rebirth_min_delta: float = 5e-4
    late_texture_rebirth_start_fraction: float = 0.50
    late_texture_rebirth_end_fraction: float = 0.80
    late_texture_rebirth_every: int = 0
    late_texture_rebirth_fraction: float = 0.025
    late_texture_rebirth_min_count: int = 2
    late_texture_rebirth_max_count: int = 8
    stage_schedule: Tuple[StageConfig, ...] = field(default_factory=lambda: make_default_stage_schedule(8000))
    stage_loss_schedule: Tuple[StageLossConfig, ...] = field(default_factory=lambda: make_default_stage_loss_schedule("quality"))

    def with_steps(self, steps: int) -> "FitConfig":
        return FitConfig(
            num_ellipses=self.num_ellipses,
            work_size=self.work_size,
            steps=steps,
            seed=self.seed,
            metrics_every=self.metrics_every,
            progress_every=self.progress_every,
            area_regularization_weight=self.area_regularization_weight,
            background_rgb=self.background_rgb,
            optimization_mode=self.optimization_mode,
            profile_enabled=self.profile_enabled,
            background_grid_size=self.background_grid_size,
            base_ellipse_fraction=self.base_ellipse_fraction,
            texture_ellipse_fraction=self.texture_ellipse_fraction,
            base_area_penalty_scale=self.base_area_penalty_scale,
            texture_area_penalty_scale=self.texture_area_penalty_scale,
            accent_area_penalty_scale=self.accent_area_penalty_scale,
            area_regularizer_power=self.area_regularizer_power,
            large_area_threshold=self.large_area_threshold,
            large_area_penalty_scale=self.large_area_penalty_scale,
            coverage_penalty_strength=self.coverage_penalty_strength,
            base_growth_schedule=self.base_growth_schedule,
            texture_growth_schedule=self.texture_growth_schedule,
            accent_growth_schedule=self.accent_growth_schedule,
            rebirth_every=self.rebirth_every,
            rebirth_fraction=self.rebirth_fraction,
            rebirth_min_count=self.rebirth_min_count,
            rebirth_max_count=self.rebirth_max_count,
            plateau_rebirth_window=self.plateau_rebirth_window,
            plateau_rebirth_cooldown=self.plateau_rebirth_cooldown,
            plateau_rebirth_start_step=self.plateau_rebirth_start_step,
            plateau_rebirth_min_delta=self.plateau_rebirth_min_delta,
            late_texture_rebirth_start_fraction=self.late_texture_rebirth_start_fraction,
            late_texture_rebirth_end_fraction=self.late_texture_rebirth_end_fraction,
            late_texture_rebirth_every=self.late_texture_rebirth_every,
            late_texture_rebirth_fraction=self.late_texture_rebirth_fraction,
            late_texture_rebirth_min_count=self.late_texture_rebirth_min_count,
            late_texture_rebirth_max_count=self.late_texture_rebirth_max_count,
            stage_schedule=make_default_stage_schedule(steps),
            stage_loss_schedule=make_default_stage_loss_schedule(self.optimization_mode),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_ellipses": self.num_ellipses,
            "work_size": self.work_size,
            "steps": self.steps,
            "seed": self.seed,
            "metrics_every": self.metrics_every,
            "progress_every": self.progress_every,
            "area_regularization_weight": self.area_regularization_weight,
            "background_rgb": list(self.background_rgb),
            "optimization_mode": self.optimization_mode,
            "profile_enabled": self.profile_enabled,
            "background_grid_size": list(self.background_grid_size),
            "base_ellipse_fraction": self.base_ellipse_fraction,
            "texture_ellipse_fraction": self.texture_ellipse_fraction,
            "base_area_penalty_scale": self.base_area_penalty_scale,
            "texture_area_penalty_scale": self.texture_area_penalty_scale,
            "accent_area_penalty_scale": self.accent_area_penalty_scale,
            "area_regularizer_power": self.area_regularizer_power,
            "large_area_threshold": self.large_area_threshold,
            "large_area_penalty_scale": self.large_area_penalty_scale,
            "coverage_penalty_strength": self.coverage_penalty_strength,
            "base_growth_schedule": [[fraction, active_fraction] for fraction, active_fraction in self.base_growth_schedule],
            "texture_growth_schedule": [[fraction, active_fraction] for fraction, active_fraction in self.texture_growth_schedule],
            "accent_growth_schedule": [[fraction, active_fraction] for fraction, active_fraction in self.accent_growth_schedule],
            "rebirth_every": self.rebirth_every,
            "rebirth_fraction": self.rebirth_fraction,
            "rebirth_min_count": self.rebirth_min_count,
            "rebirth_max_count": self.rebirth_max_count,
            "plateau_rebirth_window": self.plateau_rebirth_window,
            "plateau_rebirth_cooldown": self.plateau_rebirth_cooldown,
            "plateau_rebirth_start_step": self.plateau_rebirth_start_step,
            "plateau_rebirth_min_delta": self.plateau_rebirth_min_delta,
            "late_texture_rebirth_start_fraction": self.late_texture_rebirth_start_fraction,
            "late_texture_rebirth_end_fraction": self.late_texture_rebirth_end_fraction,
            "late_texture_rebirth_every": self.late_texture_rebirth_every,
            "late_texture_rebirth_fraction": self.late_texture_rebirth_fraction,
            "late_texture_rebirth_min_count": self.late_texture_rebirth_min_count,
            "late_texture_rebirth_max_count": self.late_texture_rebirth_max_count,
            "stage_schedule": [stage.to_dict() for stage in self.stage_schedule],
            "stage_loss_schedule": [stage.to_dict() for stage in self.stage_loss_schedule],
        }

    def get_stage(self, step_index: int) -> StageConfig:
        for stage in self.stage_schedule:
            if stage.contains(step_index):
                return stage
        return self.stage_schedule[-1]

    def get_stage_loss(self, stage_name: str) -> StageLossConfig:
        for stage_loss in self.stage_loss_schedule:
            if stage_loss.name == stage_name:
                return stage_loss
        return self.stage_loss_schedule[-1]

    def base_ellipse_count(self) -> int:
        return max(1, min(self.num_ellipses, int(round(self.num_ellipses * self.base_ellipse_fraction))))

    def texture_ellipse_count(self) -> int:
        remaining = max(0, self.num_ellipses - self.base_ellipse_count())
        if remaining == 0:
            return 0
        texture_count = int(round(self.num_ellipses * self.texture_ellipse_fraction))
        texture_count = max(1, texture_count)
        return min(remaining, texture_count)

    def accent_ellipse_count(self) -> int:
        return max(0, self.num_ellipses - self.base_ellipse_count() - self.texture_ellipse_count())

    def _active_count_for_schedule(self, step_index: int, total_count: int, schedule: Tuple[Tuple[float, float], ...]) -> int:
        if total_count <= 0:
            return 0
        progress = float(step_index + 1) / float(max(1, self.steps))
        active_fraction = schedule[-1][1]
        for threshold, threshold_active_fraction in schedule:
            if progress <= threshold:
                active_fraction = threshold_active_fraction
                break
        if active_fraction <= 0.0:
            return 0
        return max(1, min(total_count, int(round(total_count * active_fraction))))

    def active_base_ellipse_count(self, step_index: int) -> int:
        return self._active_count_for_schedule(step_index, self.base_ellipse_count(), self.base_growth_schedule)

    def active_texture_ellipse_count(self, step_index: int) -> int:
        return self._active_count_for_schedule(step_index, self.texture_ellipse_count(), self.texture_growth_schedule)

    def active_accent_ellipse_count(self, step_index: int) -> int:
        return self._active_count_for_schedule(step_index, self.accent_ellipse_count(), self.accent_growth_schedule)

    def active_ellipse_count(self, step_index: int) -> int:
        return (
            self.active_base_ellipse_count(step_index)
            + self.active_texture_ellipse_count(step_index)
            + self.active_accent_ellipse_count(step_index)
        )

    def use_lpips_on_step(self, stage_name: str, step_index: int) -> bool:
        stage_loss = self.get_stage_loss(stage_name)
        if stage_loss.weights.lpips <= 0.0:
            return False
        if stage_loss.lpips_cadence <= 1:
            return True
        return ((step_index + 1) % stage_loss.lpips_cadence) == 0

    @classmethod
    def from_args(
        cls,
        num_ellipses: int,
        work_size: int,
        steps: int,
        seed: int,
        optimization_mode: str = "quality",
        profile_enabled: bool = False,
    ) -> "FitConfig":
        if optimization_mode == "fast":
            area_regularization_weight = 0.0
            area_regularizer_power = 1.15
        else:
            area_regularization_weight = 0.0
            area_regularizer_power = 1.35
        return cls(
            num_ellipses=num_ellipses,
            work_size=work_size,
            steps=steps,
            seed=seed,
            optimization_mode=optimization_mode,
            profile_enabled=profile_enabled,
            area_regularization_weight=area_regularization_weight,
            area_regularizer_power=area_regularizer_power,
            stage_schedule=make_default_stage_schedule(steps),
            stage_loss_schedule=make_default_stage_loss_schedule(optimization_mode),
        )


def metrics_header() -> List[str]:
    return [
        "step",
        "loss",
        "l1",
        "blur_l1",
        "band_l1",
        "ms_ssim",
        "lpips",
        "edge_l1",
        "area_reg",
        "rgb_rmse",
        "bandpass_rmse",
        "hard_loss",
        "hard_l1",
        "hard_band_l1",
        "hard_ms_ssim",
        "hard_lpips",
        "hard_edge_l1",
        "hard_rgb_rmse",
        "hard_bandpass_rmse",
        "soft_hard_loss_gap",
        "soft_hard_rmse_gap",
        "background_only_rgb_rmse",
        "background_plus_base_rgb_rmse",
        "coverage_mean",
        "coverage_p95",
    ]


