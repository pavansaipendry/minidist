"""Step-5 exit criteria: ZeRO-1/2 reproduce plain DP exactly, and the optimizer
state is genuinely sharded (byte accounting, not vibes).

Tolerance calibration (measured):
  - ZeRO-1 is BITWISE identical to bucketed DP — it runs the exact same grad
    collectives, and AdamW is elementwise, so slicing the flat space changes
    nothing per element.
  - ZeRO-2 at ws=2 is also bitwise; at ws=4 reduce_scatter's per-element
    summation order differs from all_reduce's, measured drift ~9.5e-7 over a
    full run. Gate is 5e-6. A real bug (wrong slice ownership, stale padding,
    missing all_gather) diverges by >1e-2 within a few steps.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

import _env
from minidist.baseline import build_model
from minidist.config import ModelConfig, TrainConfig
from minidist.parallel.dp import DPOptions, run_dp
from minidist.parallel.zero import ZeroOptions, run_zero

MODEL_CFG = ModelConfig()
GATE_CFG = replace(TrainConfig(), steps=20)

WORLD_SIZES = _env.WORLD_SIZES
ZERO2_VS_DP_ATOL = 5e-6


@pytest.fixture(scope="module")
def dp_curves(tmp_path_factory: pytest.TempPathFactory) -> dict[int, list[float]]:
    curves = {}
    for ws in WORLD_SIZES:
        out = tmp_path_factory.mktemp(f"dp_ws{ws}")
        cfg = _env.make_dist_cfg(ws, out / "logs")
        curves[ws] = run_dp(cfg, MODEL_CFG, GATE_CFG, DPOptions(use_buckets=True), out)
    return curves


@pytest.fixture(scope="module")
def zero_runs(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[tuple[int, int], tuple[list[float], list[dict]]]:
    runs = {}
    for stage in (1, 2):
        for ws in WORLD_SIZES:
            out = tmp_path_factory.mktemp(f"zero{stage}_ws{ws}")
            cfg = _env.make_dist_cfg(ws, out / "logs")
            runs[(stage, ws)] = run_zero(cfg, MODEL_CFG, GATE_CFG, ZeroOptions(stage=stage), out)
    return runs


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_zero1_bitwise_matches_dp(
    world_size: int, dp_curves: dict, zero_runs: dict
) -> None:
    # Bitwise (==) on purpose: stage 1 shares DP's exact grad collectives and the
    # sharded AdamW is elementwise-identical. If a torch/gloo upgrade ever breaks
    # this, that's worth knowing — relax to 1e-7 only with a note explaining what
    # changed.
    curve, _ = zero_runs[(1, world_size)]
    assert curve == dp_curves[world_size]


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_zero2_matches_dp(world_size: int, dp_curves: dict, zero_runs: dict) -> None:
    curve, _ = zero_runs[(2, world_size)]
    for step, (z, d) in enumerate(zip(curve, dp_curves[world_size])):
        assert abs(z - d) < ZERO2_VS_DP_ATOL, (
            f"step {step}: zero2={z:.8f} dp={d:.8f} diff={abs(z - d):.2e}"
        )


@pytest.mark.parametrize("stage", (1, 2))
@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_optimizer_state_sharding_is_real(
    stage: int, world_size: int, zero_runs: dict
) -> None:
    _, reports = zero_runs[(stage, world_size)]
    assert len(reports) == world_size

    total_numel = build_model(MODEL_CFG, GATE_CFG.seed).param_count()
    shard_numel = math.ceil(total_numel / world_size)
    unsharded_bytes = 2 * 4 * total_numel  # exp_avg + exp_avg_sq, fp32
    # Slack covers AdamW's scalar `step` tensor and any few-byte bookkeeping.
    slack = 64

    for rank, report in enumerate(reports):
        assert report["total_numel"] == total_numel
        got = report["optimizer_state_bytes"]
        expected = 2 * 4 * shard_numel
        assert abs(got - expected) <= slack, (
            f"rank {rank}: opt state {got}B, expected ~{expected}B (1/{world_size} "
            f"of {unsharded_bytes}B) — sharding not real"
        )

    # The shards must also add up to exactly one copy of the full state — no
    # rank secretly holds a replica.
    total_sharded = sum(r["optimizer_state_bytes"] for r in reports)
    assert abs(total_sharded - 2 * 4 * shard_numel * world_size) <= slack * world_size
    assert total_sharded < unsharded_bytes + slack * world_size
