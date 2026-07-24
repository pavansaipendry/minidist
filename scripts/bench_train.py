"""Training step-time benchmark for every parallelism mode.

One command, machine-readable output:

    python scripts/bench_train.py --device cuda --world-size 2 --modes all
    -> results/bench_train_{mode}_{device}_ws{N}.json  (one file per mode)

Measured per rank (after warmup): step-time mean/p50/p90, peak CUDA memory,
plus ZeRO's optimizer-state accounting. Batches are pregenerated and moved to
the device BEFORE timing, so numbers are compute+comm, not CPU data sampling.
A barrier fences every step: per-step times are then comparable across ranks
(the price is that inter-step pipeline overlap is excluded — acceptable for
comparing modes against each other).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from minidist.baseline import build_model, build_optimizer, compute_loss  # noqa: E402
from minidist.config import DistConfig, ModelConfig, TrainConfig  # noqa: E402
from minidist.data import MarkovDataset  # noqa: E402
from minidist.launcher import launch  # noqa: E402
from minidist.model import TinyTransformer  # noqa: E402
from minidist.parallel.dp import DPOptions, GradBuckets, allreduce_gradients, broadcast_parameters  # noqa: E402
from minidist.parallel.tp import TPTransformer  # noqa: E402
from minidist.parallel.zero import ShardedAdamW  # noqa: E402

# ddp/fsdp are torch's own wrappers — allowed here ONLY as comparison baselines
# for our dp/zero implementations, never in the core.
MODES = ("dp_naive", "dp_bucketed", "zero1", "zero2", "tp", "ddp", "fsdp")


def _percentile(sorted_vals: list[float], q: float) -> float:
    idx = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def gpu_topology() -> str | None:
    """The interconnect matrix (NVLink/PCIe/SHM) the numbers were measured on —
    recorded so the fabric is documented, not inferred from bandwidth."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=10
        )
        return out.stdout or None
    except Exception:
        return None


def bench_worker(
    rank: int,
    cfg: DistConfig,
    *,
    mode: str,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    steps: int,
    warmup: int,
    repeats: int,
    out_dir: Path,
) -> None:
    device = cfg.device(rank)
    world_size = dist.get_world_size()
    total_iters = warmup + steps
    dataset = MarkovDataset(model_cfg.vocab_size, train_cfg.seq_len, train_cfg.data_seed)

    if mode == "tp":
        # TP replicates data: every rank runs the SAME full batch; the model is
        # what's sharded. Replicated params (LN, embeddings, biases) receive
        # identical grads everywhere, so a plain local AdamW keeps them in sync.
        torch.manual_seed(train_cfg.seed)
        ref = TinyTransformer(model_cfg)
        model: torch.nn.Module = TPTransformer(model_cfg)
        model.load_from_unsharded(ref)
        model = model.to(device)
        optimizer = build_optimizer(model, train_cfg)
        batches = [
            tuple(t.to(device) for t in dataset.global_batch(s, train_cfg.global_batch_size))
            for s in range(total_iters)
        ]

        def step_fn(i: int) -> None:
            inputs, targets = batches[i]
            loss = compute_loss(model, inputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    else:
        assert train_cfg.global_batch_size % world_size == 0
        local_bs = train_cfg.global_batch_size // world_size
        rows = slice(rank * local_bs, (rank + 1) * local_bs)
        model = build_model(model_cfg, train_cfg.seed).to(device)
        broadcast_parameters(model)
        batches = [
            tuple(
                t[rows].to(device)
                for t in dataset.global_batch(s, train_cfg.global_batch_size)
            )
            for s in range(total_iters)
        ]

        if mode == "ddp":
            # Baseline: torch DDP defaults (its own bucketing + backward-hook
            # overlap). Same averaging semantics as our all_reduce(SUM)/ws.
            ddp_model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[rank] if device.type == "cuda" else None
            )
            optimizer = build_optimizer(ddp_model, train_cfg)

            def step_fn(i: int) -> None:
                inputs, targets = batches[i]
                loss = compute_loss(ddp_model, inputs, targets)
                optimizer.zero_grad()
                loss.backward()  # DDP all_reduces inside backward hooks
                optimizer.step()

        elif mode == "fsdp":
            # Baseline for ZeRO: FSDP2 (fully_shard) shards params+grads+optim
            # state per block — closest torch-native analogue to our stage 2.
            from torch.distributed.fsdp import fully_shard

            for block in model.blocks:
                fully_shard(block)
            fully_shard(model)
            optimizer = build_optimizer(model, train_cfg)

            def step_fn(i: int) -> None:
                inputs, targets = batches[i]
                loss = compute_loss(model, inputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        elif mode in ("zero1", "zero2"):
            sharded = ShardedAdamW(
                model.parameters(),
                lr=train_cfg.lr,
                betas=train_cfg.betas,
                weight_decay=train_cfg.weight_decay,
                stage=1 if mode == "zero1" else 2,
                bucket_bytes=DPOptions().bucket_bytes,
            )

            def step_fn(i: int) -> None:
                inputs, targets = batches[i]
                loss = compute_loss(model, inputs, targets)
                loss.backward()
                sharded.step()
                sharded.zero_grad()

        else:
            optimizer = build_optimizer(model, train_cfg)
            buckets = (
                GradBuckets(list(model.parameters()), DPOptions().bucket_bytes)
                if mode == "dp_bucketed"
                else None
            )

            def step_fn(i: int) -> None:
                inputs, targets = batches[i]
                loss = compute_loss(model, inputs, targets)
                optimizer.zero_grad()
                loss.backward()
                if buckets is not None:
                    buckets.allreduce_and_average(world_size)
                else:
                    allreduce_gradients(model, world_size)
                optimizer.step()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def timed_step(i: int) -> float:
        # Barrier fences the step so all ranks time the same work window.
        dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        step_fn(i)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return (time.perf_counter() - t0) * 1e3

    for i in range(warmup):
        timed_step(i)
    # Repeats re-time the same `steps` batches: identical work each pass, so the
    # spread across repeat means is machine/fabric noise, not workload variance.
    repeat_means: list[float] = []
    times_ms: list[float] = []
    for _ in range(repeats):
        pass_times = [timed_step(warmup + i) for i in range(steps)]
        times_ms.extend(pass_times)
        repeat_means.append(sum(pass_times) / len(pass_times))

    ordered = sorted(times_ms)
    mean = sum(times_ms) / len(times_ms)
    stats: dict = {
        "rank": rank,
        "mean_ms": round(mean, 4),
        "p50_ms": round(_percentile(ordered, 0.50), 4),
        "p90_ms": round(_percentile(ordered, 0.90), 4),
        "max_ms": round(ordered[-1], 4),
        "repeat_means_ms": [round(m, 4) for m in repeat_means],
        "repeat_spread_ms": round(max(repeat_means) - min(repeat_means), 4),
    }
    if device.type == "cuda":
        stats["peak_mem_mb"] = round(torch.cuda.max_memory_allocated(device) / 1e6, 2)
    if mode in ("zero1", "zero2"):
        stats["zero_memory"] = sharded.memory_report()

    all_stats: list[dict | None] = [None] * world_size
    dist.all_gather_object(all_stats, stats)

    if rank == 0:
        tokens_per_step = train_cfg.global_batch_size * train_cfg.seq_len
        slowest_mean_s = max(s["mean_ms"] for s in all_stats) / 1e3
        payload = {
            "meta": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "mode": mode,
                "device": cfg.device_type,
                "backend": cfg.backend,
                "world_size": world_size,
                "torch": torch.__version__,
                "nccl": ".".join(str(v) for v in torch.cuda.nccl.version())
                if cfg.device_type == "cuda"
                else None,
                "gpu_topology": gpu_topology() if cfg.device_type == "cuda" else None,
                "steps": steps,
                "warmup": warmup,
                "repeats": repeats,
                "global_batch_size": train_cfg.global_batch_size,
                "seq_len": train_cfg.seq_len,
                "tokens_per_step": tokens_per_step,
            },
            "per_rank": all_stats,
            "tokens_per_s": round(tokens_per_step / slowest_mean_s, 1),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"bench_train_{mode}_{cfg.device_type}_ws{world_size}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(
            f"{mode:<12} ws={world_size} mean={payload['per_rank'][0]['mean_ms']:.2f}ms "
            f"tokens/s={payload['tokens_per_s']:,.0f}  -> {out_path.name}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--modes", type=str, default="all", help=f"comma list of {MODES} or 'all'")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1,
                        help="re-time the same steps N times; spread across repeats = noise")
    parser.add_argument("--global-batch", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()

    modes = MODES if args.modes == "all" else tuple(args.modes.split(","))
    for m in modes:
        assert m in MODES, f"unknown mode {m}"
    if args.device == "cpu" and "fsdp" in modes:
        # FSDP2's DeviceMesh init probes accelerator backends and crashes on
        # macOS CPU (torch.mps.is_initialized missing in torch 2.13) — GPU-only.
        print("skipping fsdp on cpu (FSDP2 DeviceMesh requires an accelerator backend)")
        modes = tuple(m for m in modes if m != "fsdp")

    from dataclasses import replace

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    if args.global_batch:
        train_cfg = replace(train_cfg, global_batch_size=args.global_batch)
    if args.seq_len:
        train_cfg = replace(train_cfg, seq_len=args.seq_len)
        assert train_cfg.seq_len <= model_cfg.max_seq_len

    for mode in modes:
        cfg = DistConfig(
            world_size=args.world_size,
            backend="nccl" if args.device == "cuda" else "gloo",
            device_type=args.device,
            log_dir=args.out_dir / "logs" / f"bench_train_{mode}_ws{args.world_size}",
            init_timeout_s=120.0,
        )
        worker = partial(
            bench_worker,
            mode=mode,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            steps=args.steps,
            warmup=args.warmup,
            repeats=args.repeats,
            out_dir=args.out_dir,
        )
        launch(worker, cfg)


if __name__ == "__main__":
    main()
