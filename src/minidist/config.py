"""Run configuration. Plain dataclasses — no framework config magic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class DistConfig:
    """Everything the launcher needs to form a process group.

    The CPU -> GPU migration is exactly two field changes (backend="nccl",
    device_type="cuda"); nothing else in the codebase may branch on the backend.
    The pairs that make sense: ("gloo", "cpu") and ("nccl", "cuda").
    """

    world_size: int = 4
    backend: str = "gloo"
    device_type: str = "cpu"  # "cuda" => one process per GPU, rank r pins cuda:r
    master_addr: str = "127.0.0.1"
    # 0 means: parent picks a free port at launch time. A fixed port would make
    # back-to-back runs (and the pytest suite) flake with "address already in use"
    # while the previous group's socket lingers in TIME_WAIT.
    master_port: int = 0
    # Mismatched collectives (one rank calls, another doesn't) present as infinite
    # silent hangs, because collectives have no tags — they match by call order.
    # A short timeout converts that hang into a loud exception.
    init_timeout_s: float = 60.0
    log_dir: Path = Path("logs")
    seed: int = 1234

    def device(self, rank: int) -> torch.device:
        """The device this rank computes (and communicates) on."""
        return torch.device("cuda", rank) if self.device_type == "cuda" else torch.device("cpu")


@dataclass(frozen=True)
class ModelConfig:
    """Deliberately tiny (~2M params): Phase 1 debugs parallelism logic, not models."""

    vocab_size: int = 512
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 1024
    max_seq_len: int = 64


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 30
    # Must stay divisible by every world_size we test: DP ranks take equal shards,
    # and mean-of-rank-means == global mean ONLY for equal shard sizes.
    global_batch_size: int = 8
    seq_len: int = 32
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    # Two independent seeds: model init and data stream must not be entangled,
    # or changing one silently changes the other and "same data" comparisons lie.
    seed: int = 1234
    data_seed: int = 4321
