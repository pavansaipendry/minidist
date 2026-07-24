"""Manual data parallelism from raw collectives. No DDP.

Scheme: every rank holds a full model replica; the global batch is split
row-wise across ranks; after backward, gradients are all_reduce-averaged.
Because every replica then holds the identical averaged gradient (all_reduce
leaves the same result everywhere) and started from identical weights, every
replica applies the identical AdamW update — the replicas stay in lockstep
forever without ever communicating parameters again.

Run manually with:  python -m minidist.parallel.dp
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn as nn

from minidist.baseline import build_model, build_optimizer, compute_loss
from minidist.config import DistConfig, ModelConfig, TrainConfig
from minidist.data import MarkovDataset
from minidist.launcher import launch
from minidist.log_utils import get_logger


@dataclass(frozen=True)
class DPOptions:
    use_buckets: bool = True
    bucket_bytes: int = 256 * 1024
    # False only in the negative-control gate: it proves the curve-matching test
    # would actually catch a missing all_reduce (replicas silently diverging).
    sync_grads: bool = True


def broadcast_parameters(model: nn.Module, src: int = 0) -> None:
    """Make rank `src`'s weights everyone's weights.

    Same-seed construction already gives identical replicas here, but the
    broadcast is the load-bearing mechanism in general (checkpoint resume on one
    rank, init nondeterminism on GPU), and it makes DP correctness independent of
    seeding discipline. Buffers need no sync: the only one (causal mask) is
    deterministic from config.
    """
    for param in model.parameters():
        dist.broadcast(param.data, src=src)


def allreduce_gradients(model: nn.Module, world_size: int) -> None:
    """Naive DP reduction: one all_reduce per gradient tensor.

    The iteration order of model.parameters() is identical on every rank because
    every rank built the identical module tree — that shared order is the ONLY
    thing matching collective k on rank 0 with collective k on rank 3. A model
    whose structure differed across ranks would mix unrelated tensors or hang.

    SUM then divide, not ReduceOp.AVG: gloo has no AVG, and SUM+div is portable
    to NCCL unchanged. Averaging (not summing) keeps the update equal to the
    gradient of the mean loss over the GLOBAL batch, so hyperparameters (lr)
    mean the same thing at every world size.
    """
    for param in model.parameters():
        if param.grad is None:  # every param trains here; the guard documents intent
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


class GradBuckets:
    """Fixed partition of gradients into flat buffers; one all_reduce per bucket.

    Why buckets: per-collective latency dominates for small tensors, so fewer,
    larger messages are cheaper than one call per parameter — and the bucket is
    the unit that a GPU version overlaps with compute (launch bucket i's
    all_reduce on a comm stream while backward still computes earlier layers).

    Buckets are filled in REVERSE parameter order: autograd produces gradients
    output-to-input, so reverse order means bucket 0's gradients are ready first
    during backward. On CPU/gloo we still reduce only after backward completes;
    adopting the overlap-friendly layout now makes overlap a launch-site change
    later, not a relayout.

    The partition is a pure function of (parameter order, sizes) — identical on
    every rank. If ranks disagreed on layout, the flat buffers would misalign and
    all_reduce would silently average UNRELATED gradients: corruption, not a
    crash. This is the classic silent DP bug the design must make impossible.
    """

    def __init__(self, params: Sequence[nn.Parameter], bucket_bytes: int) -> None:
        params = list(params)
        dtypes = {p.dtype for p in params}
        # One flat buffer per bucket implies one dtype; mixed precision would need
        # per-dtype bucketing.
        assert len(dtypes) == 1, f"uniform dtype required, got {dtypes}"

        self._buckets: list[list[nn.Parameter]] = []
        current: list[nn.Parameter] = []
        current_bytes = 0
        for p in reversed(params):
            nbytes = p.numel() * p.element_size()
            if current and current_bytes + nbytes > bucket_bytes:
                self._buckets.append(current)
                current, current_bytes = [], 0
            current.append(p)  # a single param larger than the cap gets its own bucket
            current_bytes += nbytes
        if current:
            self._buckets.append(current)

        # Allocated once, reused every step, on the params' own device: on GPU
        # this becomes the persistent comm buffer a side stream reads from.
        self._flats = [
            torch.empty(
                sum(p.numel() for p in bucket), dtype=params[0].dtype, device=params[0].device
            )
            for bucket in self._buckets
        ]

    @property
    def num_buckets(self) -> int:
        return len(self._buckets)

    def allreduce_and_average(self, world_size: int) -> None:
        for bucket, flat in zip(self._buckets, self._flats):
            offset = 0
            for p in bucket:
                n = p.numel()
                # .view(-1) (not reshape) so a non-contiguous grad fails loudly
                # instead of silently copying through a temporary.
                flat[offset : offset + n].copy_(p.grad.view(-1))
                offset += n
            # ONE collective for the whole bucket — this call replaces len(bucket)
            # per-tensor all_reduces.
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(world_size)
            offset = 0
            for p in bucket:
                n = p.numel()
                p.grad.view(-1).copy_(flat[offset : offset + n])
                offset += n


def dp_worker(
    rank: int,
    cfg: DistConfig,
    *,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    opts: DPOptions,
    out_dir: Path,
) -> None:
    logger = get_logger()
    world_size = dist.get_world_size()
    # Equal shards are a correctness requirement, not a convenience:
    # mean-of-rank-means == global-batch mean only when shards are equal.
    assert train_cfg.global_batch_size % world_size == 0
    local_bs = train_cfg.global_batch_size // world_size
    device = cfg.device(rank)

    model = build_model(model_cfg, train_cfg.seed).to(device)
    broadcast_parameters(model)
    optimizer = build_optimizer(model, train_cfg)
    dataset = MarkovDataset(model_cfg.vocab_size, train_cfg.seq_len, train_cfg.data_seed)
    buckets = GradBuckets(list(model.parameters()), opts.bucket_bytes) if opts.use_buckets else None
    if buckets is not None:
        logger.info("bucketed grad reduction: %d buckets", buckets.num_buckets)

    losses: list[float] = []
    for step in range(train_cfg.steps):
        # Every rank materializes the IDENTICAL global batch (pure function of
        # (data_seed, step)) and slices its own contiguous rows — sharding without
        # any data communication.
        inputs, targets = dataset.global_batch(step, train_cfg.global_batch_size)
        rows = slice(rank * local_bs, (rank + 1) * local_bs)
        loss = compute_loss(model, inputs[rows].to(device), targets[rows].to(device))

        optimizer.zero_grad()
        loss.backward()
        if not opts.sync_grads:
            pass  # negative control: replicas free-run on their own shards
        elif buckets is not None:
            buckets.allreduce_and_average(world_size)
        else:
            allreduce_gradients(model, world_size)
        # After the average, every rank steps with the identical gradient —
        # replicas stay in lockstep with no parameter sync ever needed again.
        optimizer.step()

        # Report the same quantity the baseline logs: mean loss over the GLOBAL
        # batch. The local loss only covers this rank's shard; averaging the
        # equal-shard means reconstructs the global mean exactly. This collective
        # sits at the same program point on every rank — after the grad
        # reduction — preserving the global collective order.
        global_loss = loss.detach().clone()
        dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
        global_loss.div_(world_size)
        losses.append(global_loss.item())
        if step % 10 == 0:
            logger.info("step %3d global_loss %.4f", step, losses[-1])

    out_dir.mkdir(parents=True, exist_ok=True)
    # Workers can't return values through mp.spawn; files are the channel (and
    # every rank writes so the parent can verify cross-rank agreement).
    (out_dir / f"dp_losses_rank{rank}.json").write_text(json.dumps(losses))


def run_dp(
    dist_cfg: DistConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    opts: DPOptions,
    out_dir: Path,
) -> list[float]:
    """Launch DP training; return the global loss curve.

    Asserts all ranks reported the bitwise-identical curve — all_reduce results
    are identical everywhere, so any disagreement means desynced replicas.
    """
    worker = partial(dp_worker, model_cfg=model_cfg, train_cfg=train_cfg, opts=opts, out_dir=out_dir)
    launch(worker, dist_cfg)
    curves = [
        json.loads((out_dir / f"dp_losses_rank{r}.json").read_text())
        for r in range(dist_cfg.world_size)
    ]
    for r in range(1, len(curves)):
        if curves[r] != curves[0]:
            raise AssertionError(f"rank {r} loss curve differs from rank 0: replicas desynced")
    return curves[0]


def main() -> None:
    """Informal check: DP curves vs the single-process reference (formal gates in tests/)."""
    from minidist.baseline import train_baseline

    model_cfg, train_cfg = ModelConfig(), TrainConfig()
    reference = train_baseline(model_cfg, train_cfg)

    for world_size, use_buckets in [(2, False), (2, True), (4, True)]:
        out_dir = Path("results") / f"dp_ws{world_size}_{'bucketed' if use_buckets else 'naive'}"
        dist_cfg = DistConfig(world_size=world_size, log_dir=out_dir / "logs")
        curve = run_dp(dist_cfg, model_cfg, train_cfg, DPOptions(use_buckets=use_buckets), out_dir)
        max_diff = max(abs(a - b) for a, b in zip(reference, curve))
        print(
            f"ws={world_size} buckets={use_buckets}: "
            f"last_loss={curve[-1]:.6f} ref={reference[-1]:.6f} max_step_diff={max_diff:.2e}"
        )


if __name__ == "__main__":
    main()
