from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import List


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run several diffvg fits in parallel with different seeds.")
    parser.add_argument("--config", required=True, help="Base JSON config for fit_diffvg_backend.")
    parser.add_argument("--seeds", required=True, help="Comma-separated seeds, for example 1,2,3,4.")
    parser.add_argument("--jobs", type=int, default=2, help="Maximum concurrent processes.")
    parser.add_argument("--output-prefix", default=None, help="Output directory prefix. Defaults to config output.")
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Expected config file to contain a JSON object.")
    return config


def _write_run_config(base_config: dict, config_path: Path, seed: int, output_prefix: str) -> Path:
    run_config = dict(base_config)
    run_config["seed"] = int(seed)
    run_config["output"] = "%s_seed_%s" % (output_prefix, seed)
    run_config_path = config_path.parent / ("%s.seed_%s.json" % (config_path.stem, seed))
    with run_config_path.open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
        handle.write("\n")
    return run_config_path


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    base_config = _load_config(config_path)
    seeds = [int(seed.strip()) for seed in str(args.seeds).split(",") if seed.strip()]
    if not seeds:
        raise ValueError("No seeds were provided.")
    jobs = max(1, int(args.jobs))
    output_prefix = args.output_prefix or str(base_config.get("output", "out/diffvg_multistart"))

    pending = [_write_run_config(base_config, config_path, seed, output_prefix) for seed in seeds]
    running: list[tuple[Path, subprocess.Popen]] = []
    failed = 0

    while pending or running:
        while pending and len(running) < jobs:
            run_config = pending.pop(0)
            command = [sys.executable, "-m", "trianglefit.direct.fit_diffvg_backend", "--config", str(run_config)]
            print("Starting %s" % " ".join(command), flush=True)
            running.append((run_config, subprocess.Popen(command)))

        next_running: list[tuple[Path, subprocess.Popen]] = []
        for run_config, process in running:
            code = process.poll()
            if code is None:
                next_running.append((run_config, process))
            elif code != 0:
                failed += 1
                print("Failed %s with exit code %d" % (run_config, code), flush=True)
            else:
                print("Finished %s" % run_config, flush=True)
        running = next_running
        if pending or running:
            time.sleep(2.0)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
