from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple


def _load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Expected pipeline config to contain a JSON object.")
    return config


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def serialize_json(data: Dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _write_stage_config(config: Dict[str, object], path: Path) -> Path:
    serialize_json(config, path)
    return path


def _attention_config_from_raw(raw_attention_config: object) -> Tuple[bool, Dict[str, object]]:
    if raw_attention_config is None:
        return False, {}
    if isinstance(raw_attention_config, bool):
        return raw_attention_config, {}
    if not isinstance(raw_attention_config, dict):
        raise ValueError("'attention' config value must be a JSON object or boolean.")
    enabled = bool(raw_attention_config.get("enabled", True))
    config = dict(raw_attention_config)
    config.pop("enabled", None)
    return enabled, config


def _run_attention_stage(input_path: Path, output_dir: Path, raw_config: Dict[str, object]) -> Tuple[Path | None, Dict[str, object]]:
    from ..attention import mediapipe_mask
    from ..direct.io import load_image

    stage_dir = ensure_dir(Path(str(raw_config.get("output", output_dir / "attention"))))
    work_size = int(raw_config.get("work_size", 512))
    model_path = Path(str(raw_config.get("model_path", mediapipe_mask.DEFAULT_MODEL_PATH)))
    require_face = bool(raw_config.get("require_face", True))
    mask, metadata = mediapipe_mask.build_attention_mask(
        image_path=input_path,
        work_size=work_size,
        model_path=model_path,
        base_weight=float(raw_config.get("base_weight", 1.0)),
        eye_weight=float(raw_config.get("eye_weight", 6.0)),
        nose_weight=float(raw_config.get("nose_weight", 3.0)),
        mouth_weight=float(raw_config.get("mouth_weight", 5.0)),
        expansion_fraction=float(raw_config.get("expansion_fraction", 0.018)),
        blur_fraction=float(raw_config.get("blur_fraction", 0.01)),
        num_faces=int(raw_config.get("num_faces", 5)),
        min_confidence=float(raw_config.get("min_confidence", 0.5)),
    )
    loaded = load_image(input_path, work_size)
    import numpy as np
    import torch

    torch.save(mask, stage_dir / "attention_mask.pt")
    np.save(stage_dir / "attention_mask.npy", mask.detach().cpu().numpy())
    mediapipe_mask._save_mask_visuals(mask=mask, target=loaded.working, output_dir=stage_dir)
    serialize_json(metadata, stage_dir / "attention_metadata.json")
    face_count = int(metadata.get("face_count", 0))
    print(
        "== Stage 1/3: attention mask ==\nSaved attention mask: faces=%d output=%s"
        % (face_count, stage_dir),
        flush=True,
    )
    if require_face and face_count <= 0:
        print("No face detected; stopping before greedy placement because attention.require_face is true.", flush=True)
        return None, metadata
    return stage_dir / "attention_mask.pt", metadata


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attention mask generation, CUDA greedy isosceles placement, then constrained diffvg refinement.")
    parser.add_argument("--config", required=True, help="Pipeline JSON config.")
    parser.add_argument("--input", default=None, help="Optional target image override.")
    parser.add_argument("--output", default=None, help="Optional output directory override.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    config = _load_config(config_path)

    input_value = args.input if args.input is not None else config.get("input")
    output_value = args.output if args.output is not None else config.get("output")
    if input_value is None or not str(input_value):
        raise ValueError("Pipeline config must provide 'input', or pass --input.")
    if output_value is None or not str(output_value):
        raise ValueError("Pipeline config must provide 'output', or pass --output.")
    input_path = Path(str(input_value))
    output_dir = ensure_dir(Path(str(output_value)))

    attention_enabled, attention_config = _attention_config_from_raw(config.get("attention"))
    raw_greedy_config = config.get("greedy", {})
    raw_diffvg_config = config.get("diffvg", {})
    if not isinstance(raw_greedy_config, dict) or not isinstance(raw_diffvg_config, dict):
        raise ValueError("'greedy' and 'diffvg' config values must be JSON objects.")
    greedy_config = dict(raw_greedy_config)
    diffvg_config = dict(raw_diffvg_config)

    attention_mask_path = None
    attention_metadata = None
    if attention_enabled:
        attention_config.setdefault("output", str(output_dir / "attention"))
        attention_mask_path, attention_metadata = _run_attention_stage(input_path=input_path, output_dir=output_dir, raw_config=attention_config)
        if attention_mask_path is None:
            serialize_json(
                {
                    "input": str(input_path),
                    "output": str(output_dir),
                    "attention_output": str(attention_config.get("output")),
                    "attention_metadata": attention_metadata,
                    "stopped": "no_face_detected",
                },
                output_dir / "pipeline_manifest.json",
            )
            return 2

    greedy_dir = ensure_dir(output_dir / "greedy")
    diffvg_dir = ensure_dir(output_dir / "diffvg")
    greedy_config.update({"input": str(input_path), "output": str(greedy_dir)})
    diffvg_config.update(
        {
            "input": str(input_path),
            "init_json": str(greedy_dir / "greedy_geometrize.json"),
            "output": str(diffvg_dir),
        }
    )
    if attention_mask_path is not None:
        greedy_config["attention_mask"] = str(attention_mask_path)
        diffvg_config["rebirth_attention_mask"] = str(attention_mask_path)

    resolved_dir = ensure_dir(output_dir / "resolved_configs")
    greedy_config_path = _write_stage_config(greedy_config, resolved_dir / "greedy.json")
    diffvg_config_path = _write_stage_config(diffvg_config, resolved_dir / "diffvg_isosceles.json")
    serialize_json(
        {
            "input": str(input_path),
            "output": str(output_dir),
            "attention_config": attention_config if attention_enabled else None,
            "attention_output": str(attention_config.get("output")) if attention_enabled else None,
            "attention_mask": str(attention_mask_path) if attention_mask_path is not None else None,
            "attention_metadata": attention_metadata,
            "greedy_config": str(greedy_config_path),
            "diffvg_config": str(diffvg_config_path),
            "greedy_output": str(greedy_dir),
            "diffvg_output": str(diffvg_dir),
        },
        output_dir / "pipeline_manifest.json",
    )

    from ..greedy_prior import place as greedy_place

    print("== Stage %d/%d: greedy prior ==" % ((2 if attention_enabled else 1), (3 if attention_enabled else 2)), flush=True)
    greedy_status = greedy_place.main(["--config", str(greedy_config_path)])
    if greedy_status != 0:
        return int(greedy_status)

    from ..direct import fit_diffvg_isosceles_backend

    print("== Stage %d/%d: diffvg isosceles refinement ==" % ((3 if attention_enabled else 2), (3 if attention_enabled else 2)), flush=True)
    diffvg_status = fit_diffvg_isosceles_backend.main(["--config", str(diffvg_config_path)])
    if diffvg_status != 0:
        return int(diffvg_status)

    print("Saved greedy output to %s" % greedy_dir, flush=True)
    print("Saved refined output to %s" % diffvg_dir, flush=True)
    print("Final image: %s" % (diffvg_dir / "final_fullres.png"), flush=True)
    print("Final JSON: %s" % (diffvg_dir / "optimized_geometrize_fullres.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
