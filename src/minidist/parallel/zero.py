"""ZeRO stage 1 and 2: shard optimizer state (and gradient reduction) across ranks.

The core idea: in DP, every rank redundantly holds the full AdamW state
(exp_avg + exp_avg_sq = 8 bytes/param in fp32 — 2x the model itself). ZeRO
partitions the FLAT parameter space into world_size equal slices; rank r keeps
optimizer state only for slice r, steps only slice r, and the ranks reassemble
the full updated parameters with one all_gather.

Stage 1: gradient averaging identical to DP (full all_reduce; every rank still
materializes the full averaged gradient); only the optimizer update is sharded.

Stage 2: replaces all_reduce with reduce_scatter — each rank receives ONLY its
slice of the summed gradient. Per-step comm volume is unchanged (ring
all_reduce moves 2N; reduce_scatter N + param all_gather N = 2N): ZeRO's win is
MEMORY, not bandwidth. On GPU, reduce_scatter is also what permits freeing
per-param grads bucket-by-bucket during backward; this CPU phase implements the
dataflow, not the transient-memory optimization.

Run manually with:  python -m minidist.parallel.zero
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn

from minidist.baseline import build_model, compute_loss
from minidist.config import DistConfig, ModelConfig, TrainConfig
from minidist.data import MarkovDataset
from minidist.launcher import launch
from minidist.log_utils import get_logger
from minidist.parallel.dp import GradBuckets, broadcast_parameters


@dataclass(frozen=True)
class ZeroOptions:
    stage: int = 2  # 1: shard optimizer state; 2: also reduce_scatter gradients
    bucket_bytes: int = 256 * 1024  # stage-1 grad all_reduce bucketing (as in DP)


class ShardedAdamW:
    """AdamW over one flat, padded, rank-owned slice of the parameter space.

    The partition is BY FLAT OFFSET, not by parameter: a slice boundary may cut a
    tensor mid-run. That is legal only because AdamW is purely ELEMENTWISE — the
    update for element i reads nothing but (param[i], grad[i], m[i], v[i]). An
    optimizer with per-tensor coupling (e.g. LAMB's trust ratios) could not be
    flat-sharded without extra communication. Equal slice sizes are what let
    reduce_scatter/all_gather run as single fixed-shape collectives.

    The inner optimizer is torch.optim.AdamW on the master shard — this project
    owns the collectives, not the AdamW arithmetic, and reusing torch's kernel
    keeps the update bitwise-identical to the DP baseline's.

    Like the bucket layout in DP: offsets/slice sizes are a pure function of
    (parameter order, sizes, world_size), so every rank computes the identical
    partition. Disagreement would mean reduce_scatter hands rank r a slice rank s
    thinks it owns — silent corruption, not a crash.
    """

    def __init__(
        self,
        params: Sequence[nn.Parameter],
        *,
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
        stage: int,
        bucket_bytes: int,
    ) -> None:
        assert stage in (1, 2)
        self._stage = stage
        self._params = list(params)
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()

        offsets: list[int] = []
        total = 0
        for p in self._params:
            offsets.append(total)
            total += p.numel()
        self._offsets = offsets
        self._numel_total = total
        # Pad the flat space so it splits into world_size EQUAL slices; the pad
        # tail belongs to the last rank and stays zero forever (zero grad + zero
        # value means AdamW never moves it).
        self._shard_numel = math.ceil(total / self._world_size)
        self._padded_numel = self._shard_numel * self._world_size
        lo = self._rank * self._shard_numel
        self._shard_range = (lo, lo + self._shard_numel)

        # The master shard is this rank's authoritative copy of its slice. All
        # buffers live on the params' device — NCCL requires CUDA tensors.
        device = self._params[0].device
        self._master = torch.zeros(self._shard_numel, device=device)
        for p, p_slice, s_slice in self._shard_overlaps():
            self._master[s_slice].copy_(p.data.view(-1)[p_slice])
        self._master.grad = torch.zeros(self._shard_numel, device=device)
        self._inner = torch.optim.AdamW(
            [self._master], lr=lr, betas=betas, weight_decay=weight_decay
        )

        # Comm buffers allocated once, reused every step.
        self._flat_params = torch.zeros(self._padded_numel, device=device)
        self._buckets = GradBuckets(self._params, bucket_bytes) if stage == 1 else None
        self._flat_grads = torch.zeros(self._padded_numel, device=device) if stage == 2 else None

    def _shard_overlaps(self) -> Iterator[tuple[nn.Parameter, slice, slice]]:
        """Yield (param, slice-into-param, slice-into-shard) for every piece of a
        parameter that lands inside this rank's flat slice [lo, hi)."""
        lo, hi = self._shard_range
        for p, off in zip(self._params, self._offsets):
            a, b = max(off, lo), min(off + p.numel(), hi)
            if a < b:
                yield p, slice(a - off, b - off), slice(a - lo, b - lo)

    def zero_grad(self) -> None:
        # set_to_none frees the gradient storage; autograd re-allocates next
        # backward. After step(), no rank holds gradients at all.
        for p in self._params:
            p.grad = None

    def step(self) -> None:
        if self._stage == 1:
            # Stage 1: full-gradient averaging IDENTICAL to plain DP (same bucket
            # layout, same collectives) — every rank ends with the full averaged
            # grad and copies out just the slice its optimizer owns.
            assert self._buckets is not None
            self._buckets.allreduce_and_average(self._world_size)
            for p, p_slice, s_slice in self._shard_overlaps():
                self._master.grad[s_slice].copy_(p.grad.view(-1)[p_slice])
        else:
            # Stage 2: one reduce_scatter replaces the all_reduce. Input is this
            # rank's full local gradient laid out flat; output is ONLY the summed
            # slice this rank owns. Every rank must zero the pad tail — the
            # collective sums whatever is there, and stale values would corrupt
            # the last rank's slice.
            assert self._flat_grads is not None
            flat = self._flat_grads
            for p, off in zip(self._params, self._offsets):
                flat[off : off + p.numel()].copy_(p.grad.view(-1))
            flat[self._numel_total :].zero_()
            dist.reduce_scatter_single(self._master.grad, flat, op=dist.ReduceOp.SUM)
            # Same SUM+divide averaging as DP, applied to the one slice we own.
            self._master.grad.div_(self._world_size)

        # Sharded update: this rank's AdamW holds state for shard_numel elements
        # only — that is the entire memory claim of ZeRO, verified by
        # memory_report() below.
        self._inner.step()

        # Reassemble the full updated parameter vector everywhere: rank r
        # contributes its master shard, every rank receives all of them. This is
        # the collective that replaces DP's implicit "every rank stepped every
        # param" — skip it and replicas diverge on the very next forward.
        dist.all_gather_single(self._flat_params, self._master)
        for p, off in zip(self._params, self._offsets):
            p.data.view(-1).copy_(self._flat_params[off : off + p.numel()])

    def memory_report(self) -> dict[str, int]:
        """Per-rank byte accounting — the gate that proves sharding is real."""
        opt_state_bytes = 0
        for state in self._inner.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    opt_state_bytes += value.numel() * value.element_size()
        return {
            "optimizer_state_bytes": opt_state_bytes,
            "master_shard_bytes": self._master.numel() * self._master.element_size(),
            "shard_numel": self._shard_numel,
            "total_numel": self._numel_total,
            # What plain DP would hold on EVERY rank: exp_avg + exp_avg_sq, fp32.
            "unsharded_optimizer_state_bytes": 2 * 4 * self._numel_total,
        }


def zero_worker(
    rank: int,
    cfg: DistConfig,
    *,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    opts: ZeroOptions,
    out_dir: Path,
) -> None:
    logger = get_logger()
    world_size = dist.get_world_size()
    assert train_cfg.global_batch_size % world_size == 0
    local_bs = train_cfg.global_batch_size // world_size
    device = cfg.device(rank)

    model = build_model(model_cfg, train_cfg.seed).to(device)
    broadcast_parameters(model)
    optimizer = ShardedAdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
        stage=opts.stage,
        bucket_bytes=opts.bucket_bytes,
    )
    dataset = MarkovDataset(model_cfg.vocab_size, train_cfg.seq_len, train_cfg.data_seed)
    logger.info("ZeRO-%d: shard %d/%d params on this rank",
                opts.stage, optimizer.memory_report()["shard_numel"], optimizer.memory_report()["total_numel"])

    losses: list[float] = []
    for step in range(train_cfg.steps):
        inputs, targets = dataset.global_batch(step, train_cfg.global_batch_size)
        rows = slice(rank * local_bs, (rank + 1) * local_bs)
        loss = compute_loss(model, inputs[rows].to(device), targets[rows].to(device))
        loss.backward()
        optimizer.step()  # all collectives live inside
        optimizer.zero_grad()

        global_loss = loss.detach().clone()
        dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
        global_loss.div_(world_size)
        losses.append(global_loss.item())
        if step % 10 == 0:
            logger.info("step %3d global_loss %.4f", step, losses[-1])

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"losses": losses, "memory": optimizer.memory_report()}
    (out_dir / f"zero_rank{rank}.json").write_text(json.dumps(payload))


def run_zero(
    dist_cfg: DistConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    opts: ZeroOptions,
    out_dir: Path,
) -> tuple[list[float], list[dict]]:
    """Launch ZeRO training; return (loss curve, per-rank memory reports)."""
    worker = partial(zero_worker, model_cfg=model_cfg, train_cfg=train_cfg, opts=opts, out_dir=out_dir)
    launch(worker, dist_cfg)
    payloads = [
        json.loads((out_dir / f"zero_rank{r}.json").read_text())
        for r in range(dist_cfg.world_size)
    ]
    curves = [p["losses"] for p in payloads]
    for r in range(1, len(curves)):
        if curves[r] != curves[0]:
            raise AssertionError(f"rank {r} loss curve differs from rank 0: replicas desynced")
    return curves[0], [p["memory"] for p in payloads]


def main() -> None:
    """Informal check vs DP (formal gates in tests/): expect bitwise-level agreement."""
    from minidist.parallel.dp import DPOptions, run_dp

    model_cfg, train_cfg = ModelConfig(), TrainConfig()
    results_root = Path("results")

    dp_curves: dict[int, list[float]] = {}
    for ws in (2, 4):
        out = results_root / f"dp_ws{ws}_ref"
        dp_curves[ws] = run_dp(
            DistConfig(world_size=ws, log_dir=out / "logs"), model_cfg, train_cfg,
            DPOptions(use_buckets=True), out,
        )

    for stage, ws in [(1, 2), (1, 4), (2, 2), (2, 4)]:
        out = results_root / f"zero{stage}_ws{ws}"
        curve, memory = run_zero(
            DistConfig(world_size=ws, log_dir=out / "logs"), model_cfg, train_cfg,
            ZeroOptions(stage=stage), out,
        )
        max_diff = max(abs(a - b) for a, b in zip(curve, dp_curves[ws]))
        mem = memory[0]
        print(
            f"zero{stage} ws={ws}: max_diff_vs_dp={max_diff:.2e} "
            f"opt_state/rank={mem['optimizer_state_bytes']:,}B "
            f"unsharded={mem['unsharded_optimizer_state_bytes']:,}B"
        )


if __name__ == "__main__":
    main()
