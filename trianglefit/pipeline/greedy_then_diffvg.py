from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

from ..direct import fit_diffvg_isosceles_backend
from ..direct.utils import ensure_dir, serialize_json
from ..greedy_prior import place as greedy_place


def _load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Expected pipeline config to contain a JSON object.")
    return config


def _write_stage_config(config: Dict[str, object], path: Path) -> Path:
    serialize_json(config, path)
    return path


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CUDA greedy isosceles placement, then constrained diffvg refinement.")
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

    raw_greedy_config = config.get("greedy", {})
    raw_diffvg_config = config.get("diffvg", {})
    if not isinstance(raw_greedy_config, dict) or not isinstance(raw_diffvg_config, dict):
        raise ValueError("'greedy' and 'diffvg' config values must be JSON objects.")
    greedy_config = dict(raw_greedy_config)
    diffvg_config = dict(raw_diffvg_config)

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

    resolved_dir = ensure_dir(output_dir / "resolved_configs")
    greedy_config_path = _write_stage_config(greedy_config, resolved_dir / "greedy.json")
    diffvg_config_path = _write_stage_config(diffvg_config, resolved_dir / "diffvg_isosceles.json")
    serialize_json(
        {
            "input": str(input_path),
            "output": str(output_dir),
            "greedy_config": str(greedy_config_path),
            "diffvg_config": str(diffvg_config_path),
            "greedy_output": str(greedy_dir),
            "diffvg_output": str(diffvg_dir),
        },
        output_dir / "pipeline_manifest.json",
    )

    print("== Stage 1/2: greedy prior ==", flush=True)
    greedy_status = greedy_place.main(["--config", str(greedy_config_path)])
    if greedy_status != 0:
        return int(greedy_status)

    print("== Stage 2/2: diffvg isosceles refinement ==", flush=True)
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
