# DETR-style Triangle Transformer

This experiment fits one image by training Transformer triangle queries on top of a ResNet18 image backbone.

```bash
python -m trianglefit.transformer.fit --input assets/linaiya.png --output out/linaiya_transformer_triangles_300 --num-triangles 300 --steps 3000 --device cuda
```

Use `--fast` to train with `L1` only:

```bash
python -m trianglefit.transformer.fit --input assets/linaiya.png --output out/linaiya_transformer_triangles_fast --num-triangles 300 --steps 3000 --device cuda --fast
```

## Notes

- The default backbone is pretrained ResNet18 and is frozen.
- The trainable parts are the feature projection, triangle queries, Transformer decoder, and parameter head.
- `--no-pretrained` uses randomly initialized ResNet18 features.
- `--finetune-backbone` also trains ResNet18, which is less stable for the first experiments.
- New code includes Chinese comments around the key Transformer concepts.



