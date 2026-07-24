"""Step-4 exit criteria: DP is provably equivalent to single-process training.

Tolerance calibration (measured, not guessed): the only legitimate difference
between DP and the reference is floating-point summation order (mean over the
full batch vs. mean-per-shard then average). Measured over 30 steps that drift
is ~1.4e-6; a real bug (missing/extra all_reduce, forgotten averaging) diverges
by >1e-2 within a few steps. The 5e-5 gate sits orders of magnitude from both.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import _env
from minidist.baseline import build_model, compute_loss, train_baseline
from minidist.config import DistConfig, ModelConfig, TrainConfig
from minidist.data import MarkovDataset
from minidist.launcher import launch
from minidist.parallel.dp import (
    DPOptions,
    GradBuckets,
    allreduce_gradients,
    broadcast_parameters,
    run_dp,
)

MODEL_CFG = ModelConfig()
GATE_CFG = replace(TrainConfig(), steps=20)

CURVE_ATOL = 5e-5


@pytest.fixture(scope="module")
def reference_curve() -> list[float]:
    return train_baseline(MODEL_CFG, GATE_CFG, _env.reference_device())


def _dp_curve(world_size: int, opts: DPOptions, out_dir: Path) -> list[float]:
    dist_cfg = _env.make_dist_cfg(world_size, out_dir / "logs")
    return run_dp(dist_cfg, MODEL_CFG, GATE_CFG, opts, out_dir)


@pytest.mark.parametrize("world_size", _env.WORLD_SIZES)
def test_dp_curve_matches_reference(
    world_size: int, reference_curve: list[float], tmp_path: Path
) -> None:
    curve = _dp_curve(world_size, DPOptions(use_buckets=True), tmp_path)
    assert len(curve) == len(reference_curve)
    # Per-step, not just final loss: two curves can cross while both being wrong.
    for step, (dp, ref) in enumerate(zip(curve, reference_curve)):
        assert abs(dp - ref) < CURVE_ATOL, (
            f"step {step}: dp={dp:.8f} ref={ref:.8f} diff={abs(dp - ref):.2e}"
        )


def test_bucketed_equals_naive(tmp_path: Path) -> None:
    naive = _dp_curve(2, DPOptions(use_buckets=False), tmp_path / "naive")
    bucketed = _dp_curve(2, DPOptions(use_buckets=True), tmp_path / "bucketed")
    # Bucketing changes message boundaries, not any element's cross-rank summation
    # order, so these are bitwise-identical in practice. The 1e-7 slack only
    # guards against gloo choosing a different reduction algorithm by message size.
    for step, (a, b) in enumerate(zip(naive, bucketed)):
        assert abs(a - b) < 1e-7, f"step {step}: naive={a!r} bucketed={b!r}"


def test_gate_catches_missing_allreduce(
    reference_curve: list[float], tmp_path: Path
) -> None:
    """Negative control: without grad sync the gate MUST fail loudly.

    If this test ever breaks, the curve-matching gate has gone vacuous (e.g. the
    task stopped being learnable and every curve 'matches' every other).
    """
    curve = _dp_curve(2, DPOptions(use_buckets=False, sync_grads=False), tmp_path)
    max_diff = max(abs(a - b) for a, b in zip(curve, reference_curve))
    assert max_diff > 1e-2, f"unsynced DP stayed within {max_diff:.2e} of reference"


def _grad_gate_worker(
    rank: int, cfg: DistConfig, *, model_cfg: ModelConfig, train_cfg: TrainConfig
) -> None:
    """Runs on every rank: proves the averaged DP gradient equals the gradient of
    an equivalent single-process large batch, computed independently in-process —
    no cross-process value passing, every rank verifies the identity itself."""
    world_size = dist.get_world_size()
    local_bs = train_cfg.global_batch_size // world_size
    device = cfg.device(rank)

    model = build_model(model_cfg, train_cfg.seed).to(device)
    broadcast_parameters(model)
    dataset = MarkovDataset(model_cfg.vocab_size, train_cfg.seq_len, train_cfg.data_seed)
    inputs, targets = dataset.global_batch(0, train_cfg.global_batch_size)
    inputs, targets = inputs.to(device), targets.to(device)
    rows = slice(rank * local_bs, (rank + 1) * local_bs)

    compute_loss(model, inputs[rows], targets[rows]).backward()
    local_grads = [p.grad.detach().clone() for p in model.parameters()]

    allreduce_gradients(model, world_size)
    dp_grads = [p.grad.detach().clone() for p in model.parameters()]

    # Bucketed path over the SAME local grads must reproduce the naive result.
    for p, g in zip(model.parameters(), local_grads):
        p.grad.copy_(g)
    GradBuckets(list(model.parameters()), bucket_bytes=256 * 1024).allreduce_and_average(world_size)
    for p, dp_grad in zip(model.parameters(), dp_grads):
        # Observed bitwise-equal; tolerance only for gloo algorithm selection.
        torch.testing.assert_close(p.grad, dp_grad, rtol=0.0, atol=1e-7)

    # Reference: identical fresh model (same seed), FULL global batch, zero
    # collectives. The DP average must equal this large-batch gradient because
    # mean-of-equal-shard-means == global mean, applied to the loss's gradient.
    ref_model = build_model(model_cfg, train_cfg.seed).to(device)
    compute_loss(ref_model, inputs, targets).backward()
    for dp_grad, ref_param in zip(dp_grads, ref_model.parameters()):
        torch.testing.assert_close(dp_grad, ref_param.grad)


@pytest.mark.parametrize("world_size", _env.WORLD_SIZES)
def test_allreduced_grads_equal_large_batch_grads(world_size: int, tmp_path: Path) -> None:
    dist_cfg = _env.make_dist_cfg(world_size, tmp_path / "logs")
    launch(partial(_grad_gate_worker, model_cfg=MODEL_CFG, train_cfg=GATE_CFG), dist_cfg)
