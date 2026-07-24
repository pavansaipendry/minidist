"""Summarize a measurement campaign directory into one table + one JSON.

    python scripts/summarize.py --dir results/l4x4
    -> prints scaling tables, writes results/l4x4/summary.json

Speedup is strong-scaling: the global batch is FIXED as world size grows, so
ideal speedup at ws=N is N× the ws=1 tokens/s of the same mode.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()

    train: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted(args.dir.rglob("bench_train_*.json")):
        data = json.loads(path.read_text())
        meta = data["meta"]
        workload = path.parent.name if path.parent != args.dir else "default"
        train[workload][meta["mode"]][meta["world_size"]] = {
            "mean_ms": data["per_rank"][0]["mean_ms"],
            "tokens_per_s": data["tokens_per_s"],
            "peak_mem_mb": data["per_rank"][0].get("peak_mem_mb"),
        }

    summary: dict = {"train": {}, "comm": {}}
    for workload, modes in sorted(train.items()):
        print(f"\n=== workload: {workload} ===")
        print(f"{'mode':<12} " + "".join(f"{f'ws{w}':>22}" for w in (1, 2, 4)))
        summary["train"][workload] = {}
        for mode, by_ws in sorted(modes.items()):
            cells, row = [], {}
            base = by_ws.get(1, {}).get("tokens_per_s")
            for ws in (1, 2, 4):
                if ws not in by_ws:
                    cells.append(f"{'—':>22}")
                    continue
                tps = by_ws[ws]["tokens_per_s"]
                speedup = f" ({tps / base:.2f}x)" if base and ws > 1 else ""
                cells.append(f"{tps:>13,.0f} tok/s{speedup:>7}")
                row[f"ws{ws}"] = by_ws[ws] | (
                    {"speedup_vs_ws1": round(tps / base, 3)} if base and ws > 1 else {}
                )
            print(f"{mode:<12} " + "".join(cells))
            summary["train"][workload][mode] = row

    for path in sorted(args.dir.glob("bench_comm_*.json")):
        data = json.loads(path.read_text())
        ws = data["meta"]["world_size"]
        summary["comm"][f"ws{ws}"] = data["results"]
        biggest = [r for r in data["results"] if r["size_mb"] == max(x["size_mb"] for x in data["results"])]
        print(f"\n=== comm ws{ws} (largest message) ===")
        for r in biggest:
            print(f"  {r['op']:<15} {r['size_mb']:>7.1f}MB  {r['avg_ms']:>8.3f}ms  busbw {r['busbw_gbps']:>6.2f} GB/s")

    out = args.dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
