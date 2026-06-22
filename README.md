# Triangle Fitting Experiments

Use triangles to fit anime images.

Standalone extraction of the image fitting experiments from `geometrize`.

It includes four lines:

- `trianglefit.direct`: direct differentiable parameter optimization for triangles
- `trianglefit.direct.fit_diffvg_backend`: direct triangle point/color fitting through the diffvg renderer backend
- `trianglefit.unet`: U-Net that predicts triangle parameters
- `trianglefit.transformer`: DETR-style Transformer triangle generator

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick start

Direct parameter fit from a Geometrize JSON:

```bash
python -m trianglefit.direct.fit_geometrize_json --input assets/linaiya.png --init-json assets/linaiya.json --output out/direct --steps 1000 --work-size 256 --device cuda
```

diffvg backend fit from the same Geometrize JSON:

```bash
python -m trianglefit.direct.fit_diffvg_backend --input assets/linaiya.png --init-json assets/linaiya.json --output out/diffvg_backend --steps 200 --work-size 256 --device cpu --samples 1
```

The same command can be driven by a JSON config:

```bash
python -m trianglefit.direct.fit_diffvg_backend --config configs/diffvg_json_1024_cuda.json
```

The diffvg backend requires `pydiffvg`. With the local third-party checkout used in this workspace, a CPU build can be installed with:

```powershell
python -m pip install -r requirements.txt
Push-Location C:\Users\13442\Desktop\work\SVG\third-party\diffvg
cmd /c """C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"" -arch=x64 && set DIFFVG_CUDA=0 && python setup.py develop"
Pop-Location
```

U-Net fit:

```bash
python -m trianglefit.unet.fit --input assets/linaiya.png --output out/unet --num-triangles 300 --steps 3000 --device cuda
```

Transformer fit:

```bash
python -m trianglefit.transformer.fit --input assets/linaiya.png --output out/transformer --num-triangles 300 --steps 3000 --device cuda
```

## Layout

- `trianglefit/direct`: direct triangle parameter fitting and rerender tools
- `trianglefit/unet`: U-Net triangle generator
- `trianglefit/transformer`: Transformer triangle generator
- `assets`: small sample inputs and the Geometrize export used for warm-start testing
