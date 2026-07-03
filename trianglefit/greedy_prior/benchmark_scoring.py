from __future__ import annotations

import argparse
from typing import List

import torch

from .placer import HillClimbTrianglePlacer, ShapeBounds, TriangleBatch, TrianglePlacementConfig, _image_sse, _pixel_grid


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark greedy-prior triangle scoring implementations.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark needs a CUDA device.")

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    target = torch.rand((3, int(args.height), int(args.width)), device=device, dtype=torch.float32, generator=generator)
    current = torch.rand_like(target)
    target_sq = target.square()
    current_sse = _image_sse(target.unsqueeze(0), current.unsqueeze(0))
    grid_x, grid_y = _pixel_grid(height=int(args.height), width=int(args.width), device=device, dtype=torch.float32)

    config = TrianglePlacementConfig(
        candidate_count=int(args.candidates),
        candidate_chunk_size=int(args.chunk_size),
        shape_bounds=ShapeBounds(),
    )
    placer = HillClimbTrianglePlacer()
    candidates = placer._random_candidates(
        count=int(args.candidates),
        width=int(args.width),
        height=int(args.height),
        config=config,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    candidates = TriangleBatch(
        centers=candidates.centers.contiguous(),
        half_base=candidates.half_base.contiguous(),
        height=candidates.height.contiguous(),
        theta=candidates.theta.contiguous(),
    )

    for _ in range(int(args.warmup)):
        placer._score_candidates_torch(candidates, target, target_sq, current, current_sse, grid_x, grid_y, int(args.chunk_size))
        placer._score_candidates_cuda(candidates, target, current, current_sse, int(args.chunk_size))
    _sync(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    torch_result = None
    for _ in range(int(args.iters)):
        torch_result = placer._score_candidates_torch(candidates, target, target_sq, current, current_sse, grid_x, grid_y, int(args.chunk_size))
    end.record()
    _sync(device)
    torch_ms = start.elapsed_time(end) / float(args.iters)

    start.record()
    cuda_result = None
    for _ in range(int(args.iters)):
        cuda_result = placer._score_candidates_cuda(candidates, target, current, current_sse, int(args.chunk_size))
    end.record()
    _sync(device)
    cuda_ms = start.elapsed_time(end) / float(args.iters)

    assert torch_result is not None
    assert cuda_result is not None
    score_diff = torch.max(torch.abs(torch_result.scores - cuda_result.scores)).detach().cpu().item()
    color_diff = torch.max(torch.abs(torch_result.colors - cuda_result.colors)).detach().cpu().item()
    count_diff = torch.max(torch.abs(torch_result.counts - cuda_result.counts)).detach().cpu().item()
    speedup = torch_ms / cuda_ms if cuda_ms > 0.0 else float("inf")

    print("height=%d width=%d candidates=%d chunk_size=%d" % (args.height, args.width, args.candidates, args.chunk_size))
    print("torch_score_ms=%.4f" % torch_ms)
    print("cuda_score_ms=%.4f" % cuda_ms)
    print("speedup=%.2fx" % speedup)
    print("max_score_abs_diff=%.6g" % score_diff)
    print("max_color_abs_diff=%.6g" % color_diff)
    print("max_count_abs_diff=%.6g" % count_diff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
