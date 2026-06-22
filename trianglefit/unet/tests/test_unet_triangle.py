from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from trianglefit.direct.io import save_image
from trianglefit.direct.renderer import render_parameters
from trianglefit.unet.config import UNetTriangleConfig
from trianglefit.unet.fit import fit_image_file
from trianglefit.unet.model import TriangleUNetGenerator


class UNetTriangleTests(unittest.TestCase):
    def test_generator_outputs_valid_triangle_parameters(self) -> None:
        config = UNetTriangleConfig(num_triangles=12, hidden_channels=8)
        generator = TriangleUNetGenerator(
            num_triangles=config.num_triangles,
            base_count=config.base_count(),
            texture_count=config.texture_count(),
            hidden_channels=config.hidden_channels,
            min_size=config.min_size,
            base_max_size=config.base_max_size,
            texture_max_size=config.texture_max_size,
        )
        decoded = generator(torch.rand(1, 3, 32, 32))
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

    def test_generator_output_renders(self) -> None:
        config = UNetTriangleConfig(num_triangles=8, hidden_channels=8)
        generator = TriangleUNetGenerator(
            num_triangles=config.num_triangles,
            base_count=config.base_count(),
            texture_count=config.texture_count(),
            hidden_channels=config.hidden_channels,
        )
        decoded = generator(torch.rand(1, 3, 32, 32))
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
            target = torch.ones(1, 3, 32, 32, dtype=torch.float32)
            target[:, 0, 8:24, 8:24] = 0.2
            target[:, 1, 8:24, 8:24] = 0.6
            target[:, 2, 8:24, 8:24] = 0.9
            input_path = tmp_path / "target.png"
            save_image(target, input_path)

            config = UNetTriangleConfig(
                num_triangles=8,
                work_size=32,
                steps=2,
                seed=3,
                hidden_channels=8,
                metrics_every=1,
                progress_every=2,
                lpips_weight=0.0,
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


if __name__ == "__main__":
    unittest.main()



