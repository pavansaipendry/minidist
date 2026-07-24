"""Step-6 exit criteria: each TP module reproduces its unsharded reference in
isolation (forward AND backward through the f/g conjugate pair), then the full
TP transformer matches end-to-end.

Tolerances: torch.testing.assert_close fp32 defaults (rtol 1.3e-6, atol 1e-5).
The only legitimate difference is summation order — the row-parallel all_reduce
adds d_ff/ws-sized partial products in ring order instead of one full-width
matmul accumulation. A misplaced f/g or a wrong shard slice produces O(1)
errors, not O(1e-6)."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import _env
from minidist.baseline import build_optimizer, compute_loss, train_baseline
from minidist.config import DistConfig, ModelConfig, TrainConfig
from minidist.data import MarkovDataset
from minidist.launcher import launch
from minidist.model import MLP, CausalSelfAttention, TinyTransformer
from minidist.parallel.tp import TPCausalSelfAttention, TPMLP, TPTransformer

MODEL_CFG = ModelConfig()
WORLD_SIZES = _env.WORLD_SIZES
# Measured max per-step drift of TP training vs baseline: ~2.4e-6 over 20 steps
# (per-block g-reduce summation order). Same margin discipline as the DP gate.
TP_TRAIN_ATOL = 5e-5


def _replicated_input(shape: tuple[int, ...], seed: int, device: torch.device) -> torch.Tensor:
    # Same generator seed on every rank -> every rank feeds the identical input,
    # mimicking the replicated activations TP assumes at region boundaries.
    # Generated on CPU (bitwise-identical everywhere), then moved.
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen).to(device)


def _mlp_gate_worker(rank: int, cfg: DistConfig, *, model_cfg: ModelConfig) -> None:
    world_size = dist.get_world_size()
    device = cfg.device(rank)
    torch.manual_seed(7)  # identical reference module on every rank
    ref = MLP(model_cfg).to(device)
    tp = TPMLP(model_cfg).to(device)
    tp.load_from_unsharded(ref)

    x = _replicated_input((2, 8, model_cfg.d_model), seed=11, device=device)
    x_ref = x.clone().requires_grad_(True)
    x_tp = x.clone().requires_grad_(True)

    out_ref = ref(x_ref)
    out_tp = tp(x_tp)
    torch.testing.assert_close(out_tp, out_ref)

    # Backward: a loss that depends nonlinearly on every output element, so any
    # f/g misplacement shows up in some gradient somewhere.
    out_ref.square().sum().backward()
    out_tp.square().sum().backward()
    # Input grad exercises f's backward all_reduce: each rank's branch holds only
    # a PARTIAL d(loss)/dx until the sum.
    torch.testing.assert_close(x_tp.grad, x_ref.grad)

    shard = model_cfg.d_ff // world_size
    rows = slice(rank * shard, (rank + 1) * shard)
    torch.testing.assert_close(tp.fc1.linear.weight.grad, ref.fc1.weight.grad[rows])
    torch.testing.assert_close(tp.fc1.linear.bias.grad, ref.fc1.bias.grad[rows])
    torch.testing.assert_close(tp.fc2.linear.weight.grad, ref.fc2.weight.grad[:, rows])
    # Replicated bias: identical full gradient on every rank (the DP-like sync).
    torch.testing.assert_close(tp.fc2.bias.grad, ref.fc2.bias.grad)


def _attn_gate_worker(rank: int, cfg: DistConfig, *, model_cfg: ModelConfig) -> None:
    world_size = dist.get_world_size()
    device = cfg.device(rank)
    torch.manual_seed(13)
    ref = CausalSelfAttention(model_cfg).to(device)
    tp = TPCausalSelfAttention(model_cfg).to(device)
    tp.load_from_unsharded(ref)

    x = _replicated_input((2, 8, model_cfg.d_model), seed=17, device=device)
    x_ref = x.clone().requires_grad_(True)
    x_tp = x.clone().requires_grad_(True)

    out_ref = ref(x_ref)
    out_tp = tp(x_tp)
    torch.testing.assert_close(out_tp, out_ref)

    out_ref.square().sum().backward()
    out_tp.square().sum().backward()
    torch.testing.assert_close(x_tp.grad, x_ref.grad)

    d = model_cfg.d_model
    ld = tp.local_dim
    block = lambda s: slice(s * d + rank * ld, s * d + (rank + 1) * ld)  # noqa: E731
    expected_qkv_w = torch.cat(
        [ref.qkv.weight.grad[block(0)], ref.qkv.weight.grad[block(1)], ref.qkv.weight.grad[block(2)]]
    )
    torch.testing.assert_close(tp.qkv.weight.grad, expected_qkv_w)
    torch.testing.assert_close(
        tp.proj.linear.weight.grad, ref.proj.weight.grad[:, rank * ld : (rank + 1) * ld]
    )
    torch.testing.assert_close(tp.proj.bias.grad, ref.proj.bias.grad)


def _e2e_gate_worker(rank: int, cfg: DistConfig, *, model_cfg: ModelConfig) -> None:
    device = cfg.device(rank)
    torch.manual_seed(19)
    ref = TinyTransformer(model_cfg).to(device)
    tp = TPTransformer(model_cfg).to(device)
    tp.load_from_unsharded(ref)

    gen = torch.Generator().manual_seed(23)
    tokens = torch.randint(0, model_cfg.vocab_size, (2, 16), generator=gen).to(device)
    with torch.no_grad():
        logits_ref = ref(tokens)
        logits_tp = tp(tokens)
    # Every rank must hold the full, identical logits: each block's two g-reduces
    # returned its output to the replicated region.
    torch.testing.assert_close(logits_tp, logits_ref)


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_tp_mlp_matches_unsharded(world_size: int, tmp_path: Path) -> None:
    cfg = _env.make_dist_cfg(world_size, tmp_path / "logs")
    launch(partial(_mlp_gate_worker, model_cfg=MODEL_CFG), cfg)


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_tp_attention_matches_unsharded(world_size: int, tmp_path: Path) -> None:
    cfg = _env.make_dist_cfg(world_size, tmp_path / "logs")
    launch(partial(_attn_gate_worker, model_cfg=MODEL_CFG), cfg)


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_tp_transformer_matches_unsharded_end_to_end(world_size: int, tmp_path: Path) -> None:
    cfg = _env.make_dist_cfg(world_size, tmp_path / "logs")
    launch(partial(_e2e_gate_worker, model_cfg=MODEL_CFG), cfg)


def _tp_training_gate_worker(
    rank: int, cfg: DistConfig, *, model_cfg: ModelConfig, train_cfg: TrainConfig
) -> None:
    """TP TRAINING must reproduce the single-process loss trajectory.

    This closes the loop the per-layer gates argue but don't prove: sharded
    params update locally, replicated params (LN, embeddings, biases) receive
    identical gradients on every rank and so stay in sync under a plain local
    AdamW — no gradient collective exists in TP, and this gate is what verifies
    none is needed."""
    device = cfg.device(rank)
    # Every rank computes the identical reference in-process (deterministic).
    ref_losses = train_baseline(model_cfg, train_cfg, device)

    torch.manual_seed(train_cfg.seed)
    ref = TinyTransformer(model_cfg)
    tp = TPTransformer(model_cfg)
    tp.load_from_unsharded(ref)
    tp = tp.to(device)
    optimizer = build_optimizer(tp, train_cfg)
    dataset = MarkovDataset(model_cfg.vocab_size, train_cfg.seq_len, train_cfg.data_seed)

    for step in range(train_cfg.steps):
        # TP replicates data: the FULL batch on every rank, same as the baseline.
        inputs, targets = dataset.global_batch(step, train_cfg.global_batch_size)
        loss = compute_loss(tp, inputs.to(device), targets.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        diff = abs(loss.item() - ref_losses[step])
        assert diff < TP_TRAIN_ATOL, (
            f"step {step}: tp={loss.item():.8f} ref={ref_losses[step]:.8f} diff={diff:.2e}"
        )


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_tp_training_matches_baseline(world_size: int, tmp_path: Path) -> None:
    cfg = _env.make_dist_cfg(world_size, tmp_path / "logs")
    gate_cfg = TrainConfig(steps=20)
    launch(partial(_tp_training_gate_worker, model_cfg=MODEL_CFG, train_cfg=gate_cfg), cfg)
