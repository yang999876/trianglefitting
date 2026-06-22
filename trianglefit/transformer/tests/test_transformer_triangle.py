from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from trianglefit.direct.io import save_image
from trianglefit.direct.renderer import render_parameters
from trianglefit.transformer.config import TransformerTriangleConfig
from trianglefit.transformer.fit import fit_image_file
from trianglefit.transformer.model import TriangleTransformerGenerator


class TransformerTriangleTests(unittest.TestCase):
    def _make_generator(self, config: TransformerTriangleConfig) -> TriangleTransformerGenerator:
        return TriangleTransformerGenerator(
            num_triangles=config.num_triangles,
            base_count=config.base_count(),
            texture_count=config.texture_count(),
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            pretrained_backbone=False,
            freeze_backbone=True,
            min_size=config.min_size,
            base_max_size=config.base_max_size,
            texture_max_size=config.texture_max_size,
        )

    def test_generator_outputs_valid_triangle_parameters(self) -> None:
        config = TransformerTriangleConfig(num_triangles=12, d_model=32, num_heads=4, num_decoder_layers=1, dim_feedforward=64)
        generator = self._make_generator(config)
        decoded = generator(torch.rand(1, 3, 64, 64))
        self.assertEqual(decoded.centers.shape, (12, 2))
        self.assertEqual(decoded.sizes.shape, (12, 2))
        self.assertEqual(decoded.theta.shape, (12, 1))
        self.assertEqual(decoded.rgb.shape, (12, 3))
        self.assertTrue(torch.all(decoded.centers >= 0.0))
        self.assertTrue(torch.all(decoded.centers <= 1.0))
        self.assertTrue(torch.all(decoded.sizes > 0.0))
        self.assertTrue(torch.all(decoded.rgb >= 0.0))
        self.assertTrue(torch.all(decoded.rgb <= 1.0))
        self.assertTrue(torch.allclose(decoded.alpha, torch.ones_like(decoded.alpha)))

    def test_initial_centers_are_spread_across_canvas(self) -> None:
        config = TransformerTriangleConfig(num_triangles=16, d_model=32, num_heads=4, num_decoder_layers=1, dim_feedforward=64)
        generator = self._make_generator(config)
        decoded = generator(torch.rand(1, 3, 64, 64))

        span = decoded.centers.max(dim=0).values - decoded.centers.min(dim=0).values
        self.assertGreater(float(span[0]), 0.5)
        self.assertGreater(float(span[1]), 0.5)

    def test_generator_output_renders(self) -> None:
        config = TransformerTriangleConfig(num_triangles=8, d_model=32, num_heads=4, num_decoder_layers=1, dim_feedforward=64)
        generator = self._make_generator(config)
        decoded = generator(torch.rand(1, 3, 64, 64))
        render = render_parameters(
            decoded=decoded,
            height=32,
            width=32,
            mask_temperature=config.mask_temperature,
            background_rgb=(1.0, 1.0, 1.0),
            active_base_count=config.base_count(),
            active_texture_count=config.texture_count(),
            active_accent_count=config.accent_count(),
        )
        self.assertEqual(render.image.shape, (1, 3, 32, 32))
        self.assertTrue(torch.all(render.image >= 0.0))
        self.assertTrue(torch.all(render.image <= 1.0))

    def test_small_synthetic_fit_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            target = torch.ones(1, 3, 64, 64, dtype=torch.float32)
            target[:, 0, 16:48, 16:48] = 0.2
            target[:, 1, 16:48, 16:48] = 0.6
            target[:, 2, 16:48, 16:48] = 0.9
            input_path = tmp_path / "target.png"
            save_image(target, input_path)

            config = TransformerTriangleConfig(
                num_triangles=8,
                work_size=64,
                steps=2,
                seed=5,
                d_model=32,
                num_heads=4,
                num_decoder_layers=1,
                dim_feedforward=64,
                metrics_every=1,
                progress_every=2,
                lpips_weight=0.0,
                pretrained_backbone=False,
                freeze_backbone=True,
            )
            artifacts = fit_image_file(input_path=input_path, output_dir=tmp_path / "out", config=config, device=torch.device("cpu"))
            self.assertTrue(Path(artifacts.final_image_path).exists())
            self.assertTrue(Path(artifacts.ellipses_path).exists())
            self.assertTrue(Path(artifacts.metrics_path).exists())
            with Path(artifacts.ellipses_path).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["num_triangles"], 8)
            self.assertIn("base", payload["ellipses"][0])
            self.assertIn("height", payload["ellipses"][0])

    def test_resume_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            target = torch.ones(1, 3, 64, 64, dtype=torch.float32)
            target[:, 0, 16:48, 16:48] = 0.2
            target[:, 1, 16:48, 16:48] = 0.6
            target[:, 2, 16:48, 16:48] = 0.9
            input_path = tmp_path / "target.png"
            save_image(target, input_path)

            base_config = TransformerTriangleConfig(
                num_triangles=8,
                work_size=64,
                steps=2,
                seed=11,
                d_model=32,
                num_heads=4,
                num_decoder_layers=1,
                dim_feedforward=64,
                metrics_every=1,
                progress_every=2,
                checkpoint_every=1,
                lpips_weight=0.0,
                pretrained_backbone=False,
                freeze_backbone=True,
            )
            first_run = fit_image_file(input_path=input_path, output_dir=tmp_path / "out1", config=base_config, device=torch.device("cpu"))
            checkpoint = tmp_path / "out1" / "checkpoints" / "step_000002.pt"
            self.assertTrue(checkpoint.exists())

            resumed_config = TransformerTriangleConfig(
                **{
                    **base_config.__dict__,
                    "steps": 4,
                }
            )
            second_run = fit_image_file(
                input_path=input_path,
                output_dir=tmp_path / "out2",
                config=resumed_config,
                device=torch.device("cpu"),
                resume_checkpoint=checkpoint,
                resume_strict=True,
            )
            self.assertTrue(Path(second_run.final_image_path).exists())
            self.assertTrue(Path(second_run.metrics_path).exists())
            self.assertTrue(Path(first_run.final_image_path).exists())


if __name__ == "__main__":
    unittest.main()


