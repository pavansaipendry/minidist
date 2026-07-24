"""Single-process reference trainer.

The loss curve produced here is the ground truth that every parallel
implementation (DP, ZeRO-1/2) must reproduce step-for-step within tolerance.

Run manually with:  python -m minidist.baseline   (writes results/baseline_loss.json)

Determinism contract (what makes two runs bitwise-identical):
  - model init: torch.manual_seed(train_cfg.seed) immediately before construction;
  - data: pure function of (data_seed, step) — see data.py;
  - no dropout / no RNG in forward;
  - CPU kernels: bitwise-reproducible run-to-run for a fixed thread count. The
    intra-op thread count changes reduction split points, so bitwise comparisons
    are only valid between runs with the same thread settings.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from minidist.config import ModelConfig, TrainConfig
from minidist.data import MarkovDataset
from minidist.model import TinyTransformer


def build_model(model_cfg: ModelConfig, seed: int) -> TinyTransformer:
    # Single source of init randomness. DP ranks (step 3) construct with this same
    # seed AND receive a rank-0 broadcast — identical starting weights two ways.
    torch.manual_seed(seed)
    return TinyTransformer(model_cfg)


def build_optimizer(model: torch.nn.Module, train_cfg: TrainConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
    )


def compute_loss(model: TinyTransformer, inputs: Tensor, targets: Tensor) -> Tensor:
    logits = model(inputs)
    # Mean over all tokens. With equal-sized per-rank shards (enforced in DP),
    # mean-of-rank-means equals this global mean exactly — the property that lets
    # DP report a comparable loss without weighting tricks.
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def train_baseline(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device = torch.device("cpu"),
) -> list[float]:
    """Train on the synthetic stream; return the per-step loss curve.

    Init always happens on CPU (build_model) and is then moved: seeded CPU init
    is bitwise-identical regardless of target device, so CPU and GPU runs start
    from the same weights. The curves themselves are NOT comparable across
    devices (different kernels); gates always compare within one device.
    """
    model = build_model(model_cfg, train_cfg.seed).to(device)
    optimizer = build_optimizer(model, train_cfg)
    dataset = MarkovDataset(model_cfg.vocab_size, train_cfg.seq_len, train_cfg.data_seed)

    losses: list[float] = []
    for step in range(train_cfg.steps):
        inputs, targets = dataset.global_batch(step, train_cfg.global_batch_size)
        loss = compute_loss(model, inputs.to(device), targets.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def main() -> None:
    model_cfg, train_cfg = ModelConfig(), TrainConfig()
    losses = train_baseline(model_cfg, train_cfg)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "baseline_loss.json"
    out_path.write_text(
        json.dumps(
            {
                "model_cfg": asdict(model_cfg),
                "train_cfg": asdict(train_cfg),
                "torch_version": torch.__version__,
                "losses": losses,
            },
            indent=2,
        )
    )
    print(f"steps={train_cfg.steps} first_loss={losses[0]:.4f} last_loss={losses[-1]:.4f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
