"""Render the README charts from a campaign summary.json.

    python scripts/plot_results.py --summary results/l4x4/summary.json
    -> results/plots/{scaling,overlap_gap,comm_busbw}.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette (slots 1-4) + chart chrome, light surface.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
)


def _style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(False)


def plot_scaling(summary: dict, out: Path) -> None:
    series = [
        ("dp_bucketed", "our DP (bucketed)", BLUE),
        ("ddp", "torch DDP", ORANGE),
        ("zero2", "our ZeRO-2", AQUA),
        ("tp", "our TP", YELLOW),
    ]
    ws = [1, 2, 4]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=220)
    ax.plot(ws, ws, "--", color=MUTED, linewidth=1.5, zorder=1)
    ax.annotate("ideal (linear)", xy=(1.55, 1.66), color=MUTED, fontsize=9, rotation=38)
    # Hand-set end-label offsets: DDP (1.19x) and ZeRO-2 (1.14x) ends sit 0.05
    # apart, too close for 9pt text without a nudge.
    label_dy = {"ddp": 0.025, "zero2": -0.025}
    for mode, label, color in series:
        rows = summary["train"]["large"][mode]
        speed = [1.0] + [rows[f"ws{w}"]["speedup_vs_ws1"] for w in (2, 4)]
        ax.plot(ws, speed, "-o", color=color, linewidth=2, markersize=7, zorder=3)
        ax.annotate(
            f"{label}  {speed[-1]:.2f}x", xy=(4, speed[-1]),
            xytext=(4.08, speed[-1] + label_dy.get(mode, 0.0)),
            color=INK2, fontsize=9, va="center",
        )
    ax.axhline(1.0, color=BASELINE, linewidth=1)
    ax.set_xlim(0.85, 5.6)
    ax.set_ylim(0.55, 2.0)
    ax.set_xticks(ws, [str(w) for w in ws])
    ax.set_xlabel("GPUs")
    ax.set_ylabel("speedup vs 1 GPU (same global batch)")
    ax.set_title(
        "Strong scaling at fixed global batch — 4x L4 over PCIe\n"
        "1.86M-param model: communication dominates, TP goes backwards",
        loc="left", fontsize=12, color=INK, fontweight="bold",
    )
    ax.legend(
        handles=[plt.Line2D([], [], color=c, linewidth=2, label=l) for _, l, c in series],
        loc="upper right", fontsize=9, labelcolor=INK2,
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "scaling.png", bbox_inches="tight")
    plt.close(fig)


def plot_overlap_gap(summary: dict, out: Path) -> None:
    workloads = [("tiny", "tiny workload\n(256 tokens/step)"), ("large", "large workload\n(4,096 tokens/step)")]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=220)
    for i, (wl, _) in enumerate(workloads):
        ours = summary["train"][wl]["dp_bucketed"]["ws2"]["tokens_per_s"]
        ddp = summary["train"][wl]["ddp"]["ws2"]["tokens_per_s"]
        rel = ddp / ours
        ax.bar(i - 0.21, 1.0, width=0.38, color=BLUE, zorder=3)
        ax.bar(i + 0.21, rel, width=0.38, color=ORANGE, zorder=3)
        ax.annotate("1.00x", xy=(i - 0.21, 1.0), xytext=(i - 0.21, 1.02),
                    ha="center", color=INK, fontsize=10)
        ax.annotate(f"{rel:.2f}x", xy=(i + 0.21, rel), xytext=(i + 0.21, rel + 0.02),
                    ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(workloads)), [c for _, c in workloads], color=INK2)
    ax.set_ylim(0, 1.62)
    ax.set_ylabel("throughput relative to our bucketed DP (2 GPUs)")
    ax.set_title(
        "What comm/compute overlap buys: torch DDP vs our DP\n"
        "the gap is the all_reduce time DDP hides under backward",
        loc="left", fontsize=12, color=INK, fontweight="bold",
    )
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=BLUE, label="our DP (bucketed, no overlap)"),
            plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="torch DDP (hook-driven overlap)"),
        ],
        loc="upper right", fontsize=9, labelcolor=INK2,
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "overlap_gap.png", bbox_inches="tight")
    plt.close(fig)


def plot_comm(summary: dict, out: Path) -> None:
    ops = [
        ("all_reduce", BLUE),
        ("reduce_scatter", ORANGE),
        ("all_gather", AQUA),
        ("broadcast", YELLOW),
    ]
    rows = summary["comm"]["ws4"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=220)
    for op, color in ops:
        pts = sorted((r["size_mb"], r["busbw_gbps"]) for r in rows if r["op"] == op)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=7, zorder=3)
        ax.annotate(op, xy=(xs[-1], ys[-1]), xytext=(xs[-1] * 1.12, ys[-1]),
                    color=INK2, fontsize=9, va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.25, 1, 4, 16, 64], ["0.25", "1", "4", "16", "64"])
    ax.set_xlim(0.2, 220)
    ax.set_xlabel("message size (MB)")
    ax.set_ylabel("bus bandwidth (GB/s)")
    ax.set_title(
        "Collective bus bandwidth — 4x L4 over PCIe (NCCL)\n"
        "small messages are latency-bound; the plateau is the fabric",
        loc="left", fontsize=12, color=INK, fontweight="bold",
    )
    ax.legend(
        handles=[plt.Line2D([], [], color=c, linewidth=2, label=o) for o, c in ops],
        loc="upper left", fontsize=9, labelcolor=INK2,
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "comm_busbw.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/l4x4/summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_scaling(summary, args.out_dir)
    plot_overlap_gap(summary, args.out_dir)
    plot_comm(summary, args.out_dir)
    for name in ("scaling", "overlap_gap", "comm_busbw"):
        print(f"wrote {args.out_dir / name}.png")


if __name__ == "__main__":
    main()
