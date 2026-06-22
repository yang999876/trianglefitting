# U-Net Triangle Prototype

This experiment fits one image by training a U-Net to generate a fixed set of opaque isosceles triangles.

```bash
python -m trianglefit.unet.fit --input assets/linaiya.png --output out/linaiya_unet_triangles --num-triangles 300 --steps 3000 --device cuda
```

Use `--fast` to train with `L1` only:

```bash
python -m trianglefit.unet.fit --input assets/linaiya.png --output out/linaiya_unet_triangles_fast --num-triangles 300 --steps 3000 --fast
```

## Notes

- The U-Net reads the target image and outputs triangle parameters directly.
- The generated parameters are rendered by the existing differentiable triangle renderer.
- This is a single-image overfit experiment, not a generalization experiment.
- New code intentionally includes Chinese comments for line-by-line learning.



