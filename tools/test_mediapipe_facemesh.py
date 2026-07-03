from __future__ import annotations

import os
import sys

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker
from mediapipe.tasks.python.vision import FaceLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import face_landmarker


ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
OUT_DIR = ROOT / "out" / "mediapipe_facemesh"
MODEL_PATH = ROOT / "third-party" / "mediapipe" / "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
INPUT_IMAGES = ("linaiya.png", "zhongli_030.png")
REPORT_PATH = OUT_DIR / "report.txt"


def ensure_model() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"Downloading model to {MODEL_PATH}")
        urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def draw_face_mesh(image_path: Path, landmarker: FaceLandmarker) -> tuple[Path, int]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    mp_image = mp.Image.create_from_file(str(image_path))
    results = landmarker.detect(mp_image)

    annotated = image_bgr.copy()
    landmark_count = 0
    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            landmark_count += 1
            drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
                is_drawing_landmarks=False,
            )
            drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
                is_drawing_landmarks=False,
            )
            drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
                is_drawing_landmarks=False,
            )
            drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
                is_drawing_landmarks=False,
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{image_path.stem}_facemesh.png"
    cv2.imwrite(str(out_path), annotated)
    return out_path, landmark_count


def main() -> None:
    summaries: list[str] = []
    model_path = ensure_model()
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_faces=5,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with FaceLandmarker.create_from_options(options) as landmarker:
        for image_name in INPUT_IMAGES:
            image_path = ASSETS_DIR / image_name
            out_path, face_count = draw_face_mesh(image_path, landmarker)
            summaries.append(f"{image_name}: faces={face_count}, output={out_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(summaries) + "\n", encoding="utf-8")
    for summary in summaries:
        print(summary, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
