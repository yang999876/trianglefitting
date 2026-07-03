from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker
from mediapipe.tasks.python.vision import FaceLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode
import numpy as np
import torch

from ..direct.io import load_image, save_image
from ..direct.utils import ensure_dir, serialize_json, tensor_to_uint8_image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "third-party" / "mediapipe" / "face_landmarker.task"

LEFT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
RIGHT_EYE = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
MOUTH = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
NOSE = (1, 2, 4, 5, 6, 45, 64, 94, 97, 98, 168, 195, 197, 275, 294, 326, 327)


def _landmarks_to_points(
    landmarks,
    indices: Sequence[int],
    width: int,
    height: int,
) -> np.ndarray:
    points = []
    for index in indices:
        if index >= len(landmarks):
            continue
        landmark = landmarks[index]
        x = int(round(float(landmark.x) * float(width - 1)))
        y = int(round(float(landmark.y) * float(height - 1)))
        points.append((max(0, min(width - 1, x)), max(0, min(height - 1, y))))
    return np.asarray(points, dtype=np.int32)


def _expanded_hull(points: np.ndarray, width: int, height: int, expansion_fraction: float) -> np.ndarray:
    if points.shape[0] < 3:
        return points
    center = points.astype(np.float32).mean(axis=0, keepdims=True)
    radius = float(max(width, height)) * float(expansion_fraction)
    vectors = points.astype(np.float32) - center
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_vectors = vectors / np.maximum(lengths, 1.0)
    expanded = points.astype(np.float32) + safe_vectors * radius
    expanded[:, 0] = np.clip(expanded[:, 0], 0.0, float(width - 1))
    expanded[:, 1] = np.clip(expanded[:, 1], 0.0, float(height - 1))
    return cv2.convexHull(expanded.astype(np.int32))


def _paint_region(
    mask: np.ndarray,
    landmarks,
    indices: Sequence[int],
    weight: float,
    expansion_fraction: float,
) -> None:
    height, width = mask.shape
    points = _landmarks_to_points(landmarks=landmarks, indices=indices, width=width, height=height)
    if points.shape[0] < 3:
        return
    hull = _expanded_hull(points=points, width=width, height=height, expansion_fraction=expansion_fraction)
    if hull.shape[0] < 3:
        return
    region = np.zeros_like(mask)
    cv2.fillConvexPoly(region, hull, float(weight))
    np.maximum(mask, region, out=mask)


def _blur_mask(mask: np.ndarray, blur_fraction: float) -> np.ndarray:
    if blur_fraction <= 0.0:
        return mask
    height, width = mask.shape
    sigma = max(0.0, float(max(width, height)) * float(blur_fraction))
    if sigma <= 0.0:
        return mask
    blurred = cv2.GaussianBlur(mask, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.maximum(mask, blurred)


def _detect_faces(image_path: Path, model_path: Path, num_faces: int, min_confidence: float):
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_faces=int(num_faces),
        min_face_detection_confidence=float(min_confidence),
        min_face_presence_confidence=float(min_confidence),
        min_tracking_confidence=float(min_confidence),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    mp_image = mp.Image.create_from_file(str(image_path))
    with FaceLandmarker.create_from_options(options) as landmarker:
        return landmarker.detect(mp_image)


def build_attention_mask(
    image_path: Path,
    work_size: int,
    model_path: Path = DEFAULT_MODEL_PATH,
    base_weight: float = 1.0,
    eye_weight: float = 6.0,
    nose_weight: float = 3.0,
    mouth_weight: float = 5.0,
    expansion_fraction: float = 0.018,
    blur_fraction: float = 0.01,
    num_faces: int = 5,
    min_confidence: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if not model_path.exists():
        raise FileNotFoundError("MediaPipe face landmarker model not found: %s" % model_path)

    loaded = load_image(image_path, work_size)
    width, height = loaded.working_size
    mask = np.full((height, width), float(base_weight), dtype=np.float32)
    results = _detect_faces(
        image_path=image_path,
        model_path=model_path,
        num_faces=num_faces,
        min_confidence=min_confidence,
    )
    face_landmarks = list(results.face_landmarks or [])
    region_specs = (
        ("left_eye", LEFT_EYE, eye_weight),
        ("right_eye", RIGHT_EYE, eye_weight),
        ("nose", NOSE, nose_weight),
        ("mouth", MOUTH, mouth_weight),
    )
    for landmarks in face_landmarks:
        for _, indices, weight in region_specs:
            _paint_region(
                mask=mask,
                landmarks=landmarks,
                indices=indices,
                weight=float(weight),
                expansion_fraction=float(expansion_fraction),
            )
    mask = _blur_mask(mask=mask, blur_fraction=float(blur_fraction))
    mask = np.maximum(mask, float(base_weight)).astype(np.float32)
    tensor = torch.from_numpy(mask).view(1, 1, height, width)
    metadata: Dict[str, object] = {
        "input": str(image_path),
        "model": str(model_path),
        "work_size": int(work_size),
        "width": int(width),
        "height": int(height),
        "face_count": len(face_landmarks),
        "base_weight": float(base_weight),
        "eye_weight": float(eye_weight),
        "nose_weight": float(nose_weight),
        "mouth_weight": float(mouth_weight),
        "expansion_fraction": float(expansion_fraction),
        "blur_fraction": float(blur_fraction),
        "min": float(mask.min()),
        "max": float(mask.max()),
        "mean": float(mask.mean()),
    }
    return tensor, metadata


def _save_mask_visuals(mask: torch.Tensor, target: torch.Tensor, output_dir: Path) -> None:
    weight = mask.detach().cpu()
    normalized = (weight - weight.min()) / (weight.max() - weight.min()).clamp_min(1e-6)
    save_image(normalized.expand(1, 3, -1, -1), output_dir / "attention_mask.png")

    target_u8 = tensor_to_uint8_image(target)
    heat = (normalized[0, 0].numpy() * 255.0).round().astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = (target_u8.astype(np.float32) * 0.65 + heat_rgb.astype(np.float32) * 0.35).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "attention_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an eyes/nose/mouth attention mask from MediaPipe face landmarks.")
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--work-size", type=int, default=512, help="Longest side for the generated mask.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="MediaPipe face_landmarker.task path.")
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--eye-weight", type=float, default=6.0)
    parser.add_argument("--nose-weight", type=float, default=3.0)
    parser.add_argument("--mouth-weight", type=float, default=5.0)
    parser.add_argument("--expansion-fraction", type=float, default=0.018)
    parser.add_argument("--blur-fraction", type=float, default=0.01)
    parser.add_argument("--num-faces", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = ensure_dir(Path(args.output))
    mask, metadata = build_attention_mask(
        image_path=Path(args.input),
        work_size=int(args.work_size),
        model_path=Path(args.model_path),
        base_weight=float(args.base_weight),
        eye_weight=float(args.eye_weight),
        nose_weight=float(args.nose_weight),
        mouth_weight=float(args.mouth_weight),
        expansion_fraction=float(args.expansion_fraction),
        blur_fraction=float(args.blur_fraction),
        num_faces=int(args.num_faces),
        min_confidence=float(args.min_confidence),
    )
    loaded = load_image(Path(args.input), int(args.work_size))
    torch.save(mask, output_dir / "attention_mask.pt")
    np.save(output_dir / "attention_mask.npy", mask.detach().cpu().numpy())
    _save_mask_visuals(mask=mask, target=loaded.working, output_dir=output_dir)
    serialize_json(metadata, output_dir / "attention_metadata.json")
    print(
        "Saved attention mask: faces=%d size=%dx%d min=%.3f max=%.3f mean=%.3f"
        % (
            int(metadata["face_count"]),
            int(metadata["width"]),
            int(metadata["height"]),
            float(metadata["min"]),
            float(metadata["max"]),
            float(metadata["mean"]),
        ),
        flush=True,
    )
    print("Mask tensor: %s" % (output_dir / "attention_mask.pt"), flush=True)
    print("Mask preview: %s" % (output_dir / "attention_mask.png"), flush=True)
    print("Overlay: %s" % (output_dir / "attention_overlay.png"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
