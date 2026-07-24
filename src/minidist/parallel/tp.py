"""Tensor parallelism: Megatron-style column/row-parallel linears, head-sharded
attention, and a TP transformer. TP group = the whole world in Phase 1;
composing TP with DP later means threading an explicit ProcessGroup through the
collectives here — an argument, not a rewrite.

The entire scheme rests on two conjugate autograd functions:

  f = copy_to_tp_region:    forward identity,   backward all_reduce(SUM)
  g = reduce_from_tp_region: forward all_reduce(SUM), backward identity

They sit at the sharded region's boundary. f marks where one replicated tensor
fans out into per-rank branches (each branch produces an independent partial
d(loss)/d(x), so backward must SUM them). g marks where per-rank partial
results re-join into one replicated tensor (each rank computed a partial sum of
the output, so forward must SUM; the incoming gradient is already replicated,
so backward passes through). Every TP bug is a misplacement of f or g.

Comm cost per transformer block: 2 all_reduces forward (one per g: attention
proj, MLP fc2) + 2 backward (one per f) — vs DP's zero forward collectives.
This is why TP wants the fastest interconnect available.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from minidist.config import ModelConfig
from minidist.model import MLP, Block, CausalSelfAttention, TinyTransformer


class _CopyToTPRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, x: Tensor) -> Tensor:
        return x

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> Tensor:
        # Clone before the in-place all_reduce: autograd may hand us a gradient
        # buffer that other consumers of x still read.
        grad = grad_output.clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad


class _ReduceFromTPRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, x: Tensor) -> Tensor:
        # Clone: all_reduce mutates in place, and the caller's partial-sum tensor
        # must not be silently overwritten under autograd's feet.
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> Tensor:
        return grad_output


def copy_to_tp_region(x: Tensor) -> Tensor:
    return _CopyToTPRegion.apply(x)


def reduce_from_tp_region(x: Tensor) -> Tensor:
    return _ReduceFromTPRegion.apply(x)


class ColumnParallelLinear(nn.Module):
    """Shards the OUTPUT dimension: rank r holds rows [r*out/ws, (r+1)*out/ws) of
    W (and of the bias). Takes the full replicated input, returns this rank's
    slice of the output — deliberately left sharded, because the consumer
    (activation + row-parallel linear) wants exactly that shard."""

    def __init__(self, in_features: int, out_features: int, world_size: int) -> None:
        super().__init__()
        assert out_features % world_size == 0
        self.linear = nn.Linear(in_features, out_features // world_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(copy_to_tp_region(x))


class RowParallelLinear(nn.Module):
    """Shards the INPUT dimension: rank r holds columns [r*in/ws, (r+1)*in/ws) of
    W. The input arrives ALREADY SHARDED; each rank computes a partial product
    over its columns; g sums the partials into the replicated output.

    The bias lives outside the reduce and is added AFTER it: if every rank added
    its (replicated) bias to its partial, the all_reduce would add the bias
    world_size times — the classic row-parallel bug. The bias is replicated and
    receives identical gradients on every rank, so it stays in sync the same way
    DP replicas do."""

    def __init__(self, in_features: int, out_features: int, world_size: int) -> None:
        super().__init__()
        assert in_features % world_size == 0
        self.linear = nn.Linear(in_features // world_size, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x_shard: Tensor) -> Tensor:
        partial = self.linear(x_shard)
        return reduce_from_tp_region(partial) + self.bias


class TPMLP(nn.Module):
    """fc1 column-parallel, GELU on the shard, fc2 row-parallel.

    The GELU needs no communication ONLY because it is elementwise: neuron j of
    the hidden layer never reads neuron k. The single forward collective for the
    whole block is fc2's g — d_ff/ws hidden neurons per rank never leave it."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.fc1 = ColumnParallelLinear(cfg.d_model, cfg.d_ff, self.world_size)
        self.fc2 = RowParallelLinear(cfg.d_ff, cfg.d_model, self.world_size)
        self._d_ff = cfg.d_ff

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))

    @torch.no_grad()
    def load_from_unsharded(self, ref: MLP) -> None:
        shard = self._d_ff // self.world_size
        rows = slice(self.rank * shard, (self.rank + 1) * shard)
        self.fc1.linear.weight.copy_(ref.fc1.weight[rows])
        self.fc1.linear.bias.copy_(ref.fc1.bias[rows])
        self.fc2.linear.weight.copy_(ref.fc2.weight[:, rows])
        self.fc2.bias.copy_(ref.fc2.bias)


class TPCausalSelfAttention(nn.Module):
    """Head-sharded attention: rank r owns n_heads/ws complete heads.

    Attention is embarrassingly parallel ACROSS heads — scores, mask, softmax and
    the value matmul never mix heads — so all of that runs locally on each
    rank's head group. The only inter-head coupling in the whole block is the
    output projection summing head contributions, and that is exactly a
    row-parallel linear: one all_reduce per forward.

    The fused QKV shard is column-parallel in spirit, but its rows are NOT a
    contiguous slice of the reference weight: the (3d, d) fused matrix is laid
    out [q; k; v], and rank r needs the r-th head-group block from EACH of the
    three sections. load_from_unsharded gathers those three row-blocks; the
    local module is then a plain Linear(d, 3d/ws) behind an explicit f."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        assert cfg.n_heads % self.world_size == 0, (
            f"n_heads={cfg.n_heads} must divide by TP world_size={self.world_size}"
        )
        self.local_heads = cfg.n_heads // self.world_size
        self.head_dim = cfg.d_model // cfg.n_heads
        self.local_dim = self.local_heads * self.head_dim  # d_model / ws
        self._d_model = cfg.d_model

        self.qkv = nn.Linear(cfg.d_model, 3 * self.local_dim)
        self.proj = RowParallelLinear(cfg.d_model, cfg.d_model, self.world_size)
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv(copy_to_tp_region(x)).chunk(3, dim=-1)
        q = q.view(B, T, self.local_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.local_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.local_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        attn = scores.softmax(dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(B, T, self.local_dim)
        return self.proj(y)

    @torch.no_grad()
    def load_from_unsharded(self, ref: CausalSelfAttention) -> None:
        d, ld, r = self._d_model, self.local_dim, self.rank
        # Row-block of section s (0=q, 1=k, 2=v) belonging to this rank's heads.
        block = lambda s: slice(s * d + r * ld, s * d + (r + 1) * ld)  # noqa: E731
        self.qkv.weight.copy_(
            torch.cat([ref.qkv.weight[block(0)], ref.qkv.weight[block(1)], ref.qkv.weight[block(2)]])
        )
        self.qkv.bias.copy_(
            torch.cat([ref.qkv.bias[block(0)], ref.qkv.bias[block(1)], ref.qkv.bias[block(2)]])
        )
        self.proj.linear.weight.copy_(ref.proj.weight[:, r * ld : (r + 1) * ld])
        self.proj.bias.copy_(ref.proj.bias)


class TPBlock(nn.Module):
    """LayerNorms are replicated (tiny, and they need the full d_model vector);
    residual adds operate on replicated tensors — both branches re-enter the
    replicated region through their g before the add."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = TPCausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = TPMLP(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    @torch.no_grad()
    def load_from_unsharded(self, ref: Block) -> None:
        self.ln1.load_state_dict(ref.ln1.state_dict())
        self.ln2.load_state_dict(ref.ln2.state_dict())
        self.attn.load_from_unsharded(ref.attn)
        self.mlp.load_from_unsharded(ref.mlp)


class TPTransformer(nn.Module):
    """Embeddings, final norm and LM head stay replicated: sharding the vocab
    dimension (Megatron's vocab-parallel embedding + cross-entropy) needs a
    gather or a parallel softmax and is out of Phase-1 scope. The redundant
    replicated compute is the price; the TP savings live in the blocks."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(TPBlock(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: Tensor) -> Tensor:
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def load_from_unsharded(self, ref: TinyTransformer) -> None:
        self.tok_emb.load_state_dict(ref.tok_emb.state_dict())
        self.pos_emb.load_state_dict(ref.pos_emb.state_dict())
        self.ln_f.load_state_dict(ref.ln_f.state_dict())
        self.head.load_state_dict(ref.head.state_dict())
        for tp_block, ref_block in zip(self.blocks, ref.blocks):
            tp_block.load_from_unsharded(ref_block)
