"""Process harness: spawn N workers, form a process group, tear down cleanly.

The lifecycle contract every worker runs under:

    setup logging -> cap threads -> init_process_group -> worker() -> barrier -> destroy

Workers passed to launch() only ever contain training/parallelism logic; the
rendezvous and teardown mechanics live here and nowhere else.
"""

from __future__ import annotations

import os
import socket
from dataclasses import replace
from datetime import timedelta
from typing import Callable, Protocol

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from minidist.config import DistConfig
from minidist.log_utils import setup_logging


class WorkerFn(Protocol):
    """A worker must be a picklable top-level function: mp.spawn re-imports the
    module in each child (spawn start method) and looks the function up by name."""

    def __call__(self, rank: int, cfg: DistConfig) -> None: ...


def _find_free_port(addr: str) -> int:
    """Ask the OS for an unused port by binding to port 0, then release it.

    There is a tiny bind race between close() and the child re-binding, but on a
    single machine running tests sequentially it is not observable in practice.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((addr, 0))
        return s.getsockname()[1]


def _worker_entry(rank: int, worker: Callable[[int, DistConfig], None], cfg: DistConfig) -> None:
    logger = setup_logging(rank, cfg.log_dir)

    # N ranks on one machine would otherwise each spawn a full OpenMP thread pool
    # and thrash the CPU, making any step-time comparison meaningless.
    torch.set_num_threads(max(1, (os.cpu_count() or cfg.world_size) // cfg.world_size))

    if cfg.device_type == "cuda":
        # One process == one GPU. Pin BEFORE init_process_group so the NCCL
        # communicator binds to the right device; two ranks landing on the same
        # GPU deadlocks NCCL rather than erroring.
        torch.cuda.set_device(rank)

    # Rendezvous: rank 0 hosts a TCPStore at master_addr:master_port; every rank
    # (including rank 0) connects to it, publishes its own endpoint, and reads the
    # others'. The backend then builds its transport between all ranks (gloo: TCP
    # mesh; NCCL: P2P/SHM rings). The timeout applies both here and to every
    # subsequent collective: a rank that reaches a collective the others never
    # call fails after init_timeout_s instead of hanging forever.
    dist.init_process_group(
        backend=cfg.backend,
        init_method=f"tcp://{cfg.master_addr}:{cfg.master_port}",
        rank=rank,
        world_size=cfg.world_size,
        timeout=timedelta(seconds=cfg.init_timeout_s),
        # Binds the group to this rank's GPU eagerly and gives barrier() an
        # unambiguous device (otherwise NCCL guesses and warns).
        device_id=torch.device("cuda", rank) if cfg.device_type == "cuda" else None,
    )
    logger.info(
        "process group up: backend=%s rank=%d/%d master=%s:%d",
        cfg.backend, rank, cfg.world_size, cfg.master_addr, cfg.master_port,
    )
    try:
        worker(rank, cfg)
        # Barrier BEFORE teardown. Collectives have no tags; they match across ranks
        # purely by call order. If a fast rank tore down while a slow rank was still
        # inside a collective, the slow rank would die with a misleading
        # connection-reset error pointing at networking instead of at the real bug.
        # The barrier is the "every rank finished its work" checkpoint.
        dist.barrier()
        logger.info("worker done, exiting cleanly")
    finally:
        # Always free the store and sockets — even on failure — so the next
        # launch() in the same test session can form a fresh group.
        dist.destroy_process_group()


def launch(worker: Callable[[int, DistConfig], None], cfg: DistConfig) -> None:
    """Run `worker(rank, cfg)` on cfg.world_size processes; block until all exit.

    mp.spawn (join=True) propagates the first failing rank's traceback to the
    parent and terminates the surviving ranks — without that, one crashed rank
    would leave its peers blocked inside a collective waiting for it.
    """
    if cfg.master_port == 0:
        cfg = replace(cfg, master_port=_find_free_port(cfg.master_addr))
    mp.spawn(_worker_entry, args=(worker, cfg), nprocs=cfg.world_size, join=True)
