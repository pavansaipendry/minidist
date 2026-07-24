"""all_reduce smoke test: proves rendezvous, participation, and replication.

Run manually with:  python -m minidist.smoke
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from minidist.config import DistConfig
from minidist.launcher import launch
from minidist.log_utils import get_logger


def smoke_worker(rank: int, cfg: DistConfig) -> None:
    logger = get_logger()

    contribution = torch.tensor([float(rank + 1)], device=cfg.device(rank))
    logger.info("before all_reduce: %s", contribution.tolist())

    # all_reduce(SUM): every rank contributes its tensor and every rank receives
    # the identical elementwise sum, in place. This "same result everywhere"
    # replication property is exactly what DP gradient averaging will rely on.
    # If any rank skipped this call, the others would block here until timeout —
    # which is also the failure mode this smoke test exists to surface early.
    dist.all_reduce(contribution, op=dist.ReduceOp.SUM)

    # Ranks contribute 1..N, so every rank must now hold N(N+1)/2.
    expected = torch.tensor([cfg.world_size * (cfg.world_size + 1) / 2.0], device=contribution.device)
    logger.info("after  all_reduce: %s (expected %s)", contribution.tolist(), expected.tolist())
    # Assert on EVERY rank: a wrong value on any single rank fails that process,
    # and mp.spawn propagates the failure to the parent.
    torch.testing.assert_close(contribution, expected)


if __name__ == "__main__":
    launch(smoke_worker, DistConfig())
