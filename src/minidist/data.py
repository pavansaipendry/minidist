"""Seeded synthetic dataset: sequences from a fixed random first-order Markov chain.

Why a Markov chain and not uniform noise: uniform targets pin the loss at ln(V)
forever, and a *flat* reference curve would "match" even a buggy DP
implementation. A learnable task gives a decreasing curve, so divergence between
implementations actually shows up.

Why batches are a pure function of (data_seed, step): every process — the
single-process baseline and every DP rank — can independently materialize the
IDENTICAL global batch for a given step and slice out its own shard. "Same
effective global batch as the reference" is then true by construction, with no
data broadcast needed.
"""

from __future__ import annotations

import torch
from torch import Tensor


class MarkovDataset:
    def __init__(
        self, vocab_size: int, seq_len: int, data_seed: int, num_batches: int = 8
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_batches = num_batches
        self._data_seed = data_seed
        gen = torch.Generator().manual_seed(data_seed)
        # Sharp logits (std 3) concentrate each row on a few successors, giving a
        # low-entropy chain — a big gap below ln(V) for the model to learn into.
        logits = 3.0 * torch.randn(vocab_size, vocab_size, generator=gen)
        self._transition_probs = logits.softmax(dim=-1)

    def global_batch(self, step: int, batch_size: int) -> tuple[Tensor, Tensor]:
        """Return (inputs, targets), each (batch_size, seq_len), for this step.

        Deterministic in (data_seed, step, batch_size) only — a fresh Generator
        per call, so global process RNG state never leaks into the data stream.
        Always generated ON CPU (callers .to(device) afterwards): CPU sampling is
        bitwise-identical everywhere, so CPU and GPU runs train on the same data.

        The dataset is FIXED: num_batches distinct batches cycled epoch-style.
        Revisiting data is what makes the tiny model's loss visibly decrease in
        a few dozen steps; an infinite fresh stream learns too slowly for the
        reference curve to have a distinctive shape worth matching.
        """
        gen = torch.Generator().manual_seed(self._data_seed + 1 + (step % self.num_batches))
        tokens = torch.empty(batch_size, self.seq_len + 1, dtype=torch.long)
        tokens[:, 0] = torch.randint(0, self.vocab_size, (batch_size,), generator=gen)
        for t in range(self.seq_len):
            row_probs = self._transition_probs[tokens[:, t]]
            tokens[:, t + 1] = torch.multinomial(row_probs, 1, generator=gen).squeeze(1)
        return tokens[:, :-1], tokens[:, 1:]
