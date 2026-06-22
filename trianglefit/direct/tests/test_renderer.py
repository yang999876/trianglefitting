from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from trianglefit.direct.io import save_image
from trianglefit.direct.model import EllipseParameters, ROLE_ACCENT, ROLE_BASE, ROLE_TEXTURE
from trianglefit.direct.renderer import render_parameters


class RendererTests(unittest.TestCase):
    def test_single_centered_triangle_stays_in_range(self) -> None:
        parameters = EllipseParameters(
            centers=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            sizes=torch.tensor([[0.24, 0.20]], dtype=torch.float32),
            theta=torch.tensor([[0.0]], dtype=torch.float32),
            rgb=torch.tensor([[0.9, 0.2, 0.1]], dtype=torch.float32),
            alpha=torch.tensor([[1.0]], dtype=torch.float32),
            role_ids=torch.tensor([ROLE_TEXTURE], dtype=torch.long),
            background_grid_rgb=None,
        )
        render = render_parameters(
            parameters,
            height=32,
            width=32,
            mask_temperature=0.1,
            background_rgb=(1.0, 1.0, 1.0),
            active_base_count=0,
            active_texture_count=1,
            active_accent_count=0,
        )
        self.assertEqual(render.image.shape, (1, 3, 32, 32))
        self.assertTrue(torch.all(render.image >= 0.0))
        self.assertTrue(torch.all(render.image <= 1.0))

    def test_hard_edge_masks_are_binary(self) -> None:
        parameters = EllipseParameters(
            centers=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            sizes=torch.tensor([[0.30, 0.24]], dtype=torch.float32),
            theta=torch.tensor([[0.3]], dtype=torch.float32),
            rgb=torch.tensor([[0.2, 0.7, 0.4]], dtype=torch.float32),
            alpha=torch.tensor([[1.0]], dtype=torch.float32),
            role_ids=torch.tensor([ROLE_BASE], dtype=torch.long),
            background_grid_rgb=torch.full((1, 3, 1, 1), 0.5, dtype=torch.float32),
        )
        render = render_parameters(
            parameters,
            height=48,
            width=48,
            mask_temperature=0.06,
            background_rgb=(1.0, 1.0, 1.0),
            hard_edges=True,
            active_base_count=1,
            active_texture_count=0,
            active_accent_count=0,
        )
        unique_values = torch.unique(render.masks)
        self.assertTrue(torch.equal(unique_values.cpu(), torch.tensor([0.0, 1.0])))

    def test_rerender_export_is_deterministic(self) -> None:
        parameters = EllipseParameters(
            centers=torch.tensor([[0.5, 0.5], [0.25, 0.7], [0.7, 0.3]], dtype=torch.float32),
            sizes=torch.tensor([[0.32, 0.24], [0.08, 0.11], [0.06, 0.09]], dtype=torch.float32),
            theta=torch.tensor([[0.1], [1.4], [0.7]], dtype=torch.float32),
            rgb=torch.tensor([[0.1, 0.2, 0.8], [0.8, 0.2, 0.1], [0.2, 0.8, 0.4]], dtype=torch.float32),
            alpha=torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32),
            role_ids=torch.tensor([ROLE_BASE, ROLE_TEXTURE, ROLE_ACCENT], dtype=torch.long),
            background_grid_rgb=torch.full((1, 3, 1, 1), 0.4, dtype=torch.float32),
        )
        render_a = render_parameters(
            parameters,
            height=40,
            width=48,
            mask_temperature=0.06,
            background_rgb=(1.0, 1.0, 1.0),
            hard_edges=True,
            active_base_count=1,
            active_texture_count=1,
            active_accent_count=1,
        )
        render_b = render_parameters(
            parameters,
            height=40,
            width=48,
            mask_temperature=0.06,
            background_rgb=(1.0, 1.0, 1.0),
            hard_edges=True,
            active_base_count=1,
            active_texture_count=1,
            active_accent_count=1,
        )
        self.assertTrue(torch.allclose(render_a.image, render_b.image, atol=1e-6, rtol=0.0))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "render.png"
            save_image(render_a.image, output)
            self.assertTrue(output.exists())

    def test_soft_and_hard_render_match_when_masks_are_binary(self) -> None:
        parameters = EllipseParameters(
            centers=torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32),
            sizes=torch.tensor([[0.5, 0.4], [0.2, 0.2]], dtype=torch.float32),
            theta=torch.tensor([[0.0], [0.0]], dtype=torch.float32),
            rgb=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32),
            alpha=torch.tensor([[1.0], [1.0]], dtype=torch.float32),
            role_ids=torch.tensor([ROLE_BASE, ROLE_TEXTURE], dtype=torch.long),
            background_grid_rgb=torch.full((1, 3, 1, 1), 1.0, dtype=torch.float32),
        )
        hard_render = render_parameters(
            parameters,
            height=48,
            width=48,
            mask_temperature=1e-6,
            background_rgb=(1.0, 1.0, 1.0),
            hard_edges=True,
            active_base_count=1,
            active_texture_count=1,
            active_accent_count=0,
        )
        soft_render = render_parameters(
            parameters,
            height=48,
            width=48,
            mask_temperature=1e-6,
            background_rgb=(1.0, 1.0, 1.0),
            hard_edges=False,
            active_base_count=1,
            active_texture_count=1,
            active_accent_count=0,
        )
        self.assertTrue(torch.allclose(hard_render.image, soft_render.image, atol=1e-4, rtol=0.0))


if __name__ == "__main__":
    unittest.main()


