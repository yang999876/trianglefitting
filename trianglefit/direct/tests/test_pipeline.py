from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from trianglefit.direct.config import FitConfig, make_default_stage_schedule
from trianglefit.direct.fit_core import fit_image_file
from trianglefit.direct.io import save_image
from trianglefit.direct.model import EllipseParameterTable


class PipelineTests(unittest.TestCase):
    def test_v42_quality_defaults(self) -> None:
        config = FitConfig.from_args(100, 64, 100, 1, optimization_mode="quality")
        stage_loss = config.get_stage_loss("single_stage")
        self.assertAlmostEqual(stage_loss.weights.blur_l1, 0.0)
        self.assertAlmostEqual(stage_loss.weights.band_l1, 0.0)
        self.assertAlmostEqual(stage_loss.weights.lpips, 0.3)
        self.assertAlmostEqual(stage_loss.weights.ms_ssim, 0.0)
        self.assertAlmostEqual(stage_loss.weights.edge_l1, 0.0)
        self.assertAlmostEqual(stage_loss.weights.l1, 1.0)
        self.assertEqual(stage_loss.lpips_cadence, 1)
        self.assertAlmostEqual(config.area_regularization_weight, 0.0)
        self.assertEqual(config.background_grid_size, (1, 1))
        self.assertAlmostEqual(config.base_ellipse_fraction, 0.20)
        self.assertAlmostEqual(config.texture_ellipse_fraction, 0.80)
        self.assertEqual(config.progress_every, 400)
        self.assertEqual(config.plateau_rebirth_window, 120)
        self.assertEqual(config.plateau_rebirth_cooldown, 120)
        self.assertEqual(config.plateau_rebirth_start_step, 150)

    def test_three_role_growth_are_distinct(self) -> None:
        config = FitConfig(num_ellipses=100, steps=100)
        self.assertEqual(config.base_ellipse_count(), 20)
        self.assertEqual(config.texture_ellipse_count(), 80)
        self.assertEqual(config.accent_ellipse_count(), 0)
        self.assertEqual(config.active_base_ellipse_count(0), 20)
        self.assertEqual(config.active_base_ellipse_count(24), 20)
        self.assertEqual(config.active_texture_ellipse_count(0), 80)
        self.assertEqual(config.active_texture_ellipse_count(49), 80)
        self.assertEqual(config.active_accent_ellipse_count(0), 0)
        self.assertEqual(config.rebirth_every, 0)
        self.assertEqual(config.late_texture_rebirth_every, 0)

    def test_role_alpha_is_locked_to_one(self) -> None:
        table = EllipseParameterTable.from_decoded(
            centers=torch.tensor([[0.2, 0.3], [0.7, 0.8], [0.4, 0.6]], dtype=torch.float32),
            sizes=torch.tensor([[0.12, 0.10], [0.09, 0.11], [0.06, 0.08]], dtype=torch.float32),
            theta=torch.tensor([[0.0], [1.0], [0.4]], dtype=torch.float32),
            rgb=torch.tensor([[0.2, 0.3, 0.4], [0.7, 0.6, 0.5], [0.4, 0.5, 0.6]], dtype=torch.float32),
            background_grid_rgb=torch.full((1, 3, 1, 1), 0.5, dtype=torch.float32),
            base_count=1,
            texture_count=1,
        )
        decoded = table.decode(
            active_base_count=1,
            active_texture_count=1,
            active_accent_count=1,
        )
        self.assertTrue(torch.allclose(decoded.alpha, torch.ones_like(decoded.alpha)))

    def test_default_schedule_uses_single_stage_defaults(self) -> None:
        schedule = make_default_stage_schedule(100)
        self.assertEqual(len(schedule), 1)
        self.assertEqual(schedule[0].name, "single_stage")
        self.assertAlmostEqual(schedule[0].start_fraction, 0.0)
        self.assertAlmostEqual(schedule[0].end_fraction, 1.0)
        self.assertAlmostEqual(schedule[0].mask_temperature, 0.02)
        self.assertAlmostEqual(schedule[0].geometry_lr, 0.003)
        self.assertAlmostEqual(schedule[0].color_lr, 0.002)

    def test_small_synthetic_fit_produces_background_and_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            target_image = torch.ones(1, 3, 32, 32, dtype=torch.float32)
            yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
            triangle_mask = (yy >= 8) & (yy <= 24) & (torch.abs(xx - 16) <= (yy - 8) * 0.5)
            target_image[:, 0, triangle_mask] = 0.2
            target_image[:, 1, triangle_mask] = 0.6
            target_image[:, 2, triangle_mask] = 0.9
            input_path = tmp_path / "target.png"
            save_image(target_image, input_path)

            config = FitConfig.from_args(12, 32, 12, 7, optimization_mode="quality").with_steps(12)
            config = FitConfig(
                **{
                    **config.__dict__,
                    "metrics_every": 4,
                    "progress_every": 6,
                }
            )

            artifacts = fit_image_file(input_path=input_path, output_dir=tmp_path / "output", config=config, device=torch.device("cpu"))
            self.assertTrue(Path(artifacts.final_image_path).exists())
            self.assertTrue(Path(artifacts.ellipses_path).exists())
            self.assertTrue(Path(artifacts.metrics_path).exists())
            self.assertTrue((tmp_path / "output" / "target_work.png").exists())
            self.assertTrue((tmp_path / "output" / "final_work.png").exists())
            self.assertTrue((tmp_path / "output" / "final_fullres.png").exists())
            with Path(artifacts.metrics_path).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertIn("history", payload)
            self.assertTrue(payload["history"])
            final_entry = payload["history"][-1]
            self.assertIn("background_only_rgb_rmse", final_entry)
            self.assertIn("background_plus_base_rgb_rmse", final_entry)
            self.assertIn("coverage_mean", final_entry)
            self.assertIn("soft_hard_loss_gap", final_entry)
            self.assertIn("bandpass_rmse", final_entry)
            self.assertIn("blur_l1", final_entry)
            self.assertIn("allow_grow", final_entry)
            self.assertIn("allow_rebirth", final_entry)
            self.assertIn("plateau_rebirth_due", final_entry)
            self.assertIn("best_train_loss", final_entry)
            self.assertIn("last_loss_improvement_step", final_entry)
            self.assertIn("stabilize_mode", final_entry)
            self.assertIn("events", payload)
            self.assertEqual(payload["events"], [])
            self.assertIn("best_step", payload)
            self.assertIn("last_grow_step", payload)
            self.assertIn("last_rebirth_step", payload)
            self.assertEqual(payload["last_grow_step"], 0)
            self.assertEqual(payload["last_rebirth_step"], 0)
            with Path(artifacts.ellipses_path).open("r", encoding="utf-8") as handle:
                ellipse_payload = json.load(handle)
            self.assertIn("background_grid_rgb", ellipse_payload)
            self.assertEqual(ellipse_payload["background_grid_height"], 1)
            self.assertIn("base_ellipse_count", ellipse_payload)
            self.assertIn("texture_ellipse_count", ellipse_payload)
            self.assertIn("accent_ellipse_count", ellipse_payload)
            first_triangle = ellipse_payload["ellipses"][0]
            self.assertIn("base", first_triangle)
            self.assertIn("height", first_triangle)
            self.assertNotIn("a", first_triangle)


if __name__ == "__main__":
    unittest.main()


