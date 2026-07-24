"""Harness exit criteria: the process group forms, collectives work at multiple
world sizes, and failures in any rank surface in the parent instead of hanging."""

from __future__ import annotations

from pathlib import Path

import pytest

import _env
from minidist.launcher import launch
from minidist.smoke import smoke_worker


@pytest.mark.parametrize("world_size", _env.WORLD_SIZES)
def test_all_reduce_smoke(world_size: int, tmp_path: Path) -> None:
    cfg = _env.make_dist_cfg(world_size, tmp_path)
    launch(smoke_worker, cfg)  # workers assert internally; failure propagates here


def _failing_worker(rank: int, cfg: DistConfig) -> None:
    # Rank 1 dies before any collective; rank 0 proceeds to the lifecycle barrier
    # and blocks there. mp.spawn must notice rank 1's death and terminate rank 0 —
    # this test pins down that the harness fails loudly instead of deadlocking.
    if rank == 1:
        raise RuntimeError("intentional failure in rank 1")


def test_worker_failure_propagates(tmp_path: Path) -> None:
    cfg = _env.make_dist_cfg(2, tmp_path)
    with pytest.raises(Exception, match="intentional failure"):
        launch(_failing_worker, cfg)


def test_per_rank_log_files_written(tmp_path: Path) -> None:
    cfg = _env.make_dist_cfg(2, tmp_path)
    launch(smoke_worker, cfg)
    for rank in range(cfg.world_size):
        log_file = cfg.log_dir / f"rank{rank}.log"
        assert log_file.exists()
        assert f"[rank {rank}]" in log_file.read_text()
