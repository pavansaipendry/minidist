"""Collective microbenchmark: latency and bandwidth per op per message size.

One command, machine-readable output:

    python scripts/bench_comm.py --device cuda --world-size 2
    -> results/bench_comm_{device}_ws{N}.json

Bandwidth conventions follow nccl-tests: algbw = bytes / time; busbw scales
algbw by the traffic factor of the algorithm's optimal ring — all_reduce
2(n-1)/n, reduce_scatter and all_gather (n-1)/n, broadcast 1 — so busbw is
directly comparable to the link's physical bandwidth regardless of op.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from minidist.config import DistConfig  # noqa: E402
from minidist.launcher import launch  # noqa: E402


def _timed_avg_s(fn: Callable[[], None], iters: int, device: torch.device) -> float:
    """Average seconds per call, iters back-to-back. Barrier + sync align ranks
    before t0; syncing only at the end amortizes launch overhead the way real
    training does."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dist.barrier()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / iters


def bench_worker(
    rank: int,
    cfg: DistConfig,
    *,
    sizes_mb: list[float],
    iters: int,
    warmup: int,
    out_dir: Path,
) -> None:
    device = cfg.device(rank)
    world_size = dist.get_world_size()
    rows: list[dict] = []

    for size_mb in sizes_mb:
        # Round numel up to a world_size multiple so reduce_scatter/all_gather
        # shard shapes are exact.
        numel = max(world_size, int(size_mb * 1e6 / 4) // world_size * world_size)
        full = torch.randn(numel, device=device)
        shard = torch.randn(numel // world_size, device=device)
        gathered = torch.empty(numel, device=device)

        ops: dict[str, Callable[[], None]] = {
            "all_reduce": lambda: dist.all_reduce(full, op=dist.ReduceOp.SUM),
            "reduce_scatter": lambda: dist.reduce_scatter_single(
                shard, full, op=dist.ReduceOp.SUM
            ),
            "all_gather": lambda: dist.all_gather_single(gathered, shard),
            "broadcast": lambda: dist.broadcast(full, src=0),
        }
        busbw_factor = {
            "all_reduce": 2 * (world_size - 1) / world_size,
            "reduce_scatter": (world_size - 1) / world_size,
            "all_gather": (world_size - 1) / world_size,
            "broadcast": 1.0,
        }

        for name, fn in ops.items():
            _timed_avg_s(fn, warmup, device)
            avg_s = _timed_avg_s(fn, iters, device)
            # The slowest rank defines the collective's real duration.
            avg_t = torch.tensor([avg_s], device=device)
            dist.all_reduce(avg_t, op=dist.ReduceOp.MAX)
            avg_s = avg_t.item()
            nbytes = numel * 4
            rows.append(
                {
                    "op": name,
                    "size_mb": round(nbytes / 1e6, 3),
                    "avg_ms": round(avg_s * 1e3, 4),
                    "algbw_gbps": round(nbytes / avg_s / 1e9, 3),
                    "busbw_gbps": round(nbytes * busbw_factor[name] / avg_s / 1e9, 3),
                }
            )

    if rank == 0:
        payload = {
            "meta": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "device": cfg.device_type,
                "backend": cfg.backend,
                "world_size": world_size,
                "torch": torch.__version__,
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                if cfg.device_type == "cuda"
                else [],
                "iters": iters,
                "warmup": warmup,
            },
            "results": rows,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"bench_comm_{cfg.device_type}_ws{world_size}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out_path}")
        for r in rows:
            print(
                f"  {r['op']:<15} {r['size_mb']:>8.2f}MB  {r['avg_ms']:>9.3f}ms  "
                f"busbw {r['busbw_gbps']:>7.2f} GB/s"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sizes-mb", type=str, default="0.25,1,4,16,64")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()

    cfg = DistConfig(
        world_size=args.world_size,
        backend="nccl" if args.device == "cuda" else "gloo",
        device_type=args.device,
        log_dir=args.out_dir / "logs" / f"bench_comm_ws{args.world_size}",
        init_timeout_s=120.0,
    )
    worker = partial(
        bench_worker,
        sizes_mb=[float(s) for s in args.sizes_mb.split(",")],
        iters=args.iters,
        warmup=args.warmup,
        out_dir=args.out_dir,
    )
    launch(worker, cfg)


if __name__ == "__main__":
    main()
