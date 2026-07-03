# Greedy Prior

This package generates a Geometrize-style greedy initialization for later optimization backends.

It is not a differentiable optimization backend. Each round samples opaque isosceles triangle candidates, hill-climbs all candidates in parallel, adds the best candidate to the current image, and exports a Geometrize-compatible JSON warm start.

```bash
python -m trianglefit.greedy_prior.place --config configs/greedy_place_256_cuda.json
```

The default config uses:

- `num_triangles = 300`
- `candidate_count = 2048`
- `max_shape_mutations = 2000`
- fixed alpha `255`
- one primitive type: opaque isosceles triangles

The main output is `greedy_geometrize.json`, which can be passed to the diffvg refinement backend:

```bash
python -m trianglefit.direct.fit_diffvg_backend --input assets/linaiya.png --init-json out/greedy_place_256_cuda/greedy_geometrize.json --output out/diffvg_refine --work-size 256 --device cuda
```
