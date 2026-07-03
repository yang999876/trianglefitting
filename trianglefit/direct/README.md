# Differentiable Triangle Prototype

This experiment fits a single image using a fixed stack of differentiable opaque isosceles triangles.

## Usage

```bash
python -m trianglefit.direct.fit --input path/to/image.png --output out/direct --num-ellipses 200 --steps 8000 --work-size 256
```

For a faster, lower-memory run that uses a lighter loss mix:

```bash
python -m trianglefit.direct.fit --input path/to/image.png --output out/direct_fast --num-ellipses 200 --steps 8000 --work-size 256 --fast
```

To print coarse timing breakdowns for render, loss, backward, optimizer, and export:

```bash
python -m trianglefit.direct.fit --input path/to/image.png --output out/direct_profile --num-ellipses 200 --steps 800 --work-size 256 --profile
```

The command writes:

- `final.png`
- `final_work.png`
- `final_fullres.png`
- `target_work.png`
- `best.png`
- `ellipses.json`
- `metrics.json`
- `progress/step_XXXX.png`

By default the script auto-selects `cuda` when available and otherwise falls back to `cpu`. If you do not pass `--seed`, it generates a random seed and prints it so the run can be reproduced later.

To reproduce a render from exported parameters:

```bash
python -m trianglefit.direct.rerender --params out/direct/ellipses.json --output out/direct/rerender.png
```

If you want the soft differentiable preview for debugging, override the export mode:

```bash
python -m trianglefit.direct.rerender --params out/direct/ellipses.json --output out/direct/rerender_soft.png --soft-edges
```

## Geometrize warm start

To directly optimize a Geometrize triangle export with the current soft-mask renderer:

```bash
python -m trianglefit.direct.fit_geometrize_json --input assets/linaiya.png --init-json assets/linaiya.json --output out/direct_geometrize --steps 1000 --work-size 256 --device cuda
```

To run the same warm start through diffvg's renderer/backward:

```bash
python -m trianglefit.direct.fit_diffvg_backend --input assets/linaiya.png --init-json assets/linaiya.json --output out/diffvg_backend --steps 200 --work-size 256 --device cpu --samples 1
```

To refine a greedy isosceles-triangle warm start with diffvg while preserving the isosceles parameterization:

```bash
python -m trianglefit.direct.fit_diffvg_isosceles_backend --config configs/diffvg_isosceles_512_cuda.json
```

To create a Geometrize-style greedy warm start with the CUDA hill-climb prior:

```bash
python -m trianglefit.greedy_prior.place --config configs/greedy_place_256_cuda.json
```

This writes `greedy_geometrize.json`, which can be passed to `fit_diffvg_backend --init-json` for the later diffvg refinement stage. The greedy placer uses opaque isosceles triangles, fixed alpha `255`, and `num-triangles=300` by default. Unlike upstream Geometrize, all candidates are hill-climbed in parallel before the best one is added, and the search/apply loop runs inside the CUDA extension instead of the Torch op graph.

You can temporarily override config values from the command line:

```bash
python -m trianglefit.greedy_prior.place --config configs/greedy_place_256_cuda.json --seed 1234 --output out/greedy_seed_1234
```

Or use a config file:

```bash
python -m trianglefit.direct.fit_diffvg_backend --config configs/diffvg_json_1024_cuda.json
python -m trianglefit.direct.fit_diffvg_backend --config configs/diffvg_random_1024_cuda.json
```

The diffvg backend optimizes raw closed-path triangle vertices plus RGBA fill colors. It is meant as a backend comparison against the soft-mask triangle parameterization, not as a final exporter yet. It currently writes `target_work.png`, `initial.png`, `best.png`, `final.png`, `metrics.json`, and `progress/step_XXXX.png`.

## Notes

- Training uses a single average-color base coat (`1x1` background grid) plus triangles.
- All triangles use fixed `alpha = 1.0`; alpha is not learned.
- Each exported primitive stores `cx`, `cy`, `base`, `height`, `theta`, `r`, `g`, `b`, and `kind`.
- The main preview is training-resolution aligned:
  - `target_work.png` is the resized optimization target
  - `progress/step_XXXX.png`, `final.png`, and `final_work.png` are rendered at the same working resolution
  - `final_fullres.png` is an auxiliary rerender at the original image size
- Base and texture triangle budgets are fully active from the start of training.
- Texture rebirth is enabled only when training has plateaued for a while; it does not run on a fixed cadence.
- Training now uses a single stage for the full run:
  - `single_stage`: `0%` to `100%`
  - `mask_temperature = 0.02`
  - `geometry_lr = 0.003`
  - `color_lr = 0.002`
- The default loss is `L1 + 0.3 * LPIPS`.
- `metrics.json` also records:
  - `best_step`
  - `best_hard_loss`
  - `best_hard_rgb_rmse`
  - `last_grow_step`
  - `last_rebirth_step`
  - `soft_hard_loss_gap`
  - `soft_hard_rmse_gap`



