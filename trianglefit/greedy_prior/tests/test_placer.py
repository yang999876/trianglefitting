from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from trianglefit.greedy_prior.place import parse_args
from trianglefit.greedy_prior import cuda_scoring
from trianglefit.greedy_prior.placer import (
    HillClimbTrianglePlacer,
    ShapeBounds,
    TrianglePlacementConfig,
    export_geometrize_json,
)


def _synthetic_target(size: int = 32) -> torch.Tensor:
    target = torch.ones(1, 3, size, size, dtype=torch.float32)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    triangle = (yy >= size // 4) & (yy <= (size * 3) // 4) & (torch.abs(xx - size // 2) <= (yy - size // 4) * 0.7)
    target[:, 0, triangle] = 0.10
    target[:, 1, triangle] = 0.55
    target[:, 2, triangle] = 0.90
    return target


class GreedyPlacerTests(unittest.TestCase):
    def _require_cuda(self) -> None:
        if not cuda_scoring.is_available():
            self.skipTest("CUDA greedy extension is not available.")

    def test_hill_climb_places_triangle_and_reduces_loss(self) -> None:
        self._require_cuda()
        target = _synthetic_target().cuda()
        config = TrianglePlacementConfig(
            num_triangles=2,
            candidate_count=96,
            max_shape_mutations=24,
            seed=11,
            max_half_base_fraction=0.45,
            max_height_fraction=0.70,
        )
        result = HillClimbTrianglePlacer().fit(target=target, config=config)
        self.assertGreater(result.triangle_count, 0)
        self.assertLessEqual(result.final_sse, result.initial_sse)
        for triangle in result.triangles:
            self.assertGreaterEqual(min(triangle.rgb), 0.0)
            self.assertLessEqual(max(triangle.rgb), 1.0)

    def test_shape_bounds_constrain_triangle_centers(self) -> None:
        self._require_cuda()
        target = _synthetic_target().cuda()
        bounds = ShapeBounds.from_values((0.4, 0.4, 0.6, 0.6))
        config = TrianglePlacementConfig(
            num_triangles=1,
            candidate_count=64,
            max_shape_mutations=8,
            seed=7,
            shape_bounds=bounds,
            max_half_base_fraction=0.30,
            max_height_fraction=0.45,
        )
        result = HillClimbTrianglePlacer().fit(target=target, config=config)
        self.assertGreater(result.triangle_count, 0)
        triangle = result.triangles[0]
        self.assertGreaterEqual(triangle.cx / result.width, bounds.x_min)
        self.assertLessEqual(triangle.cx / result.width, bounds.x_max)
        self.assertGreaterEqual(triangle.cy / result.height, bounds.y_min)
        self.assertLessEqual(triangle.cy / result.height, bounds.y_max)

    def test_export_uses_opaque_isosceles_triangle_schema(self) -> None:
        self._require_cuda()
        target = _synthetic_target().cuda()
        config = TrianglePlacementConfig(
            num_triangles=1,
            candidate_count=32,
            max_shape_mutations=4,
            seed=3,
            background_rgb=(255.0, 255.0, 255.0),
            max_half_base_fraction=0.40,
            max_height_fraction=0.60,
        )
        result = HillClimbTrianglePlacer().fit(target=target, config=config)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "greedy.json"
            export_geometrize_json(result.triangles, result.background_rgb, path, width=result.width, height=result.height)
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        self.assertEqual(payload["shapes"][0]["type"], 1)
        self.assertEqual(payload["shapes"][0]["color"], [255, 255, 255, 255])
        self.assertEqual(payload["shapes"][1]["type"], 512)
        self.assertEqual(payload["shapes"][1]["color"][3], 255)
        self.assertEqual(len(payload["shapes"][1]["data"]), 5)

    def test_cli_config_file_supplies_defaults_and_allows_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "greedy.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input": "assets/linaiya.png",
                        "output": "out/from_config",
                        "device": "cpu",
                        "num_triangles": 12,
                        "candidate_count": 64,
                        "background_rgb": [255, 255, 255],
                        "shape_bounds": [0.1, 0.2, 0.8, 0.9],
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(["--config", str(config_path), "--num-triangles", "5"])
        self.assertEqual(args.input, "assets/linaiya.png")
        self.assertEqual(args.output, "out/from_config")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.num_triangles, 5)
        self.assertEqual(args.candidate_count, 64)
        self.assertEqual(args.background_rgb, [255, 255, 255])
        self.assertEqual(args.shape_bounds, [0.1, 0.2, 0.8, 0.9])


if __name__ == "__main__":
    unittest.main()
