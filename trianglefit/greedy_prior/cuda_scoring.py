from __future__ import annotations

from pathlib import Path
from typing import Tuple

import os
import shutil
import subprocess
import tempfile
import torch
from torch.utils.cpp_extension import load

_EXTENSION = None
_EXTENSION_ERROR: Exception | None = None


def _ensure_ninja_on_path() -> None:
    if os.name != "nt":
        return
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if any((Path(part) / "ninja.exe").exists() for part in path_parts if part):
        return
    candidates = [
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja"),
        Path("C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja"),
        Path("C:/Program Files/Microsoft Visual Studio/2022/Professional/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja"),
        Path("C:/Program Files/Microsoft Visual Studio/2022/Enterprise/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja"),
    ]
    for candidate in candidates:
        if (candidate / "ninja.exe").exists():
            os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")
            return


def _ensure_msvc_on_windows() -> None:
    if os.name != "nt":
        return
    if shutil.which("cl") is not None:
        return
    vcvars_candidates = [
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Auxiliary/Build/vcvars64.bat"),
        Path("C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Auxiliary/Build/vcvars64.bat"),
    ]
    vcvars = next((path for path in vcvars_candidates if path.exists()), None)
    if vcvars is None:
        return
    script_text = '@echo off\r\nset VSLANG=1033\r\ncall "%s" >nul\r\nset\r\n' % str(vcvars)
    script_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".cmd", delete=False, encoding="utf-8", newline="") as handle:
            handle.write(script_text)
            script_path = handle.name
        output = subprocess.check_output(["cmd", "/d", "/c", script_path], stderr=subprocess.STDOUT)
    except Exception:
        return
    finally:
        if script_path is not None:
            try:
                Path(script_path).unlink(missing_ok=True)
            except Exception:
                pass
    text = output.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            os.environ[key] = value
    os.environ.setdefault("TORCH_DONT_CHECK_COMPILER_ABI", "1")


def _ensure_cuda_arch() -> None:
    if "TORCH_CUDA_ARCH_LIST" in os.environ:
        return
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    if major >= 12:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0+PTX"


def _load_extension():
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None:
        raise _EXTENSION_ERROR
    _ensure_ninja_on_path()
    _ensure_msvc_on_windows()
    _ensure_cuda_arch()
    source_dir = Path(__file__).resolve().parent / "cuda"
    try:
        _EXTENSION = load(
            name="trianglefit_greedy_scoring_cuda",
            sources=[str(source_dir / "scoring.cpp"), str(source_dir / "scoring_kernel.cu")],
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        _EXTENSION_ERROR = exc
        raise
    return _EXTENSION


def is_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        _load_extension()
    except Exception:
        return False
    return True


def load_error() -> Exception | None:
    return _EXTENSION_ERROR


def score_triangles(
    target_chw: torch.Tensor,
    current_chw: torch.Tensor,
    centers: torch.Tensor,
    half_base: torch.Tensor,
    height: torch.Tensor,
    theta: torch.Tensor,
    current_sse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not target_chw.is_cuda:
        raise ValueError("CUDA scoring requires CUDA tensors.")
    extension = _load_extension()
    score_value = float(current_sse.detach().cpu().item()) if isinstance(current_sse, torch.Tensor) else float(current_sse)
    scores, colors, counts = extension.score_triangles(
        target_chw.contiguous(),
        current_chw.contiguous(),
        centers.contiguous(),
        half_base.contiguous(),
        height.contiguous(),
        theta.contiguous(),
        score_value,
    )
    return scores, colors, counts


def search_and_apply(
    target_chw: torch.Tensor,
    current_chw: torch.Tensor,
    current_sse: float,
    candidate_count: int,
    mutation_count: int,
    bounds_min_x: float,
    bounds_min_y: float,
    bounds_max_x: float,
    bounds_max_y: float,
    min_half_base: float,
    max_half_base: float,
    min_height: float,
    max_height: float,
    center_step_x: float,
    center_step_y: float,
    half_base_step: float,
    height_step: float,
    angle_step: float,
    seed: int,
    round_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not target_chw.is_cuda:
        raise ValueError("CUDA greedy search requires CUDA tensors.")
    extension = _load_extension()
    best_params, best_color, best_score = extension.search_and_apply(
        target_chw.contiguous(),
        current_chw.contiguous(),
        float(current_sse),
        int(candidate_count),
        int(mutation_count),
        float(bounds_min_x),
        float(bounds_min_y),
        float(bounds_max_x),
        float(bounds_max_y),
        float(min_half_base),
        float(max_half_base),
        float(min_height),
        float(max_height),
        float(center_step_x),
        float(center_step_y),
        float(half_base_step),
        float(height_step),
        float(angle_step),
        int(seed),
        int(round_index),
    )
    return best_params, best_color, best_score
