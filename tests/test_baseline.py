"""Step-2 exit criteria: the reference trainer is deterministic, actually learns,
and the data stream is a pure function of (data_seed, step)."""

from __future__ import annotations

from dataclasses import replace

import torch

import _env
from minidist.baseline import build_model, train_baseline
from minidist.config import ModelConfig, TrainConfig
from minidist.data import MarkovDataset

MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()


def test_two_runs_bitwise_identical() -> None:
    # Bitwise equality (==, not allclose) is the whole point: both runs execute in
    # this same process with identical thread settings, so any difference means a
    # nondeterministic op crept into the model or data path.
    cfg = replace(TRAIN_CFG, steps=10)
    device = _env.reference_device()
    assert train_baseline(MODEL_CFG, cfg, device) == train_baseline(MODEL_CFG, cfg, device)


def test_loss_decreases() -> None:
    losses = train_baseline(MODEL_CFG, TRAIN_CFG, _env.reference_device())
    # Observed drop with defaults is ~1.7 nats; 1.0 leaves slack without letting a
    # flat (i.e. broken-task) curve pass.
    assert losses[-1] < losses[0] - 1.0, f"first={losses[0]:.4f} last={losses[-1]:.4f}"


def test_param_count_in_target_range() -> None:
    model = build_model(MODEL_CFG, TRAIN_CFG.seed)
    assert 1_500_000 <= model.param_count() <= 5_000_000


def test_data_is_pure_function_of_seed_and_step() -> None:
    ds = MarkovDataset(MODEL_CFG.vocab_size, TRAIN_CFG.seq_len, TRAIN_CFG.data_seed)
    x1, y1 = ds.global_batch(step=3, batch_size=8)
    x2, y2 = ds.global_batch(step=3, batch_size=8)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)

    # Fixed dataset: the stream cycles with period num_batches...
    x_cycle, _ = ds.global_batch(step=3 + ds.num_batches, batch_size=8)
    assert torch.equal(x1, x_cycle)
    # ...but consecutive steps within an epoch are distinct batches.
    x_next, _ = ds.global_batch(step=4, batch_size=8)
    assert not torch.equal(x1, x_next)

    assert x1.shape == y1.shape == (8, TRAIN_CFG.seq_len)
    assert x1.dtype == torch.long
    assert int(x1.max()) < MODEL_CFG.vocab_size and int(x1.min()) >= 0
    # Targets are inputs shifted by one — the LM objective wiring.
    assert torch.equal(x1[:, 1:], y1[:, :-1])
