"""Env-driven gate configuration — the SAME correctness gates run on CPU or GPU.

Default (no env): cpu/gloo at world sizes 2 and 4 — the hermetic Phase-1 suite.
GPU verification (scripts/verify_gpu.py sets these):

    MINIDIST_DEVICE=cuda MINIDIST_WORLD_SIZES=2 python -m pytest

World sizes must not exceed the GPU count: NCCL deadlocks (rather than erroring)
when two ranks share a device.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from minidist.config import DistConfig

DEVICE_TYPE = os.environ.get("MINIDIST_DEVICE", "cpu")
BACKEND = "nccl" if DEVICE_TYPE == "cuda" else "gloo"
WORLD_SIZES = tuple(
    int(s) for s in os.environ.get("MINIDIST_WORLD_SIZES", "2,4").split(",")
)


def make_dist_cfg(world_size: int, log_dir: Path) -> DistConfig:
    return DistConfig(
        world_size=world_size,
        backend=BACKEND,
        device_type=DEVICE_TYPE,
        log_dir=log_dir,
        # NCCL init (CUDA context + communicator setup) is slower than gloo's.
        init_timeout_s=120.0,
    )


def reference_device() -> torch.device:
    """Device for in-process (non-spawned) reference computations."""
    return torch.device("cuda", 0) if DEVICE_TYPE == "cuda" else torch.device("cpu")
