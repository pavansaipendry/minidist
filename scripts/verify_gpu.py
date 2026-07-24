"""One-command GPU correctness verification.

Runs the SAME pytest gates as Phase 1, on CUDA/NCCL, at every world size the
node supports, then writes a machine-readable summary. Exit code == pytest's.

Usage:  python scripts/verify_gpu.py
Output: results/gpu_verify.json + results/gpu_gates.junit.xml
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        print("verify_gpu: no CUDA device visible — this script only runs on GPU nodes")
        return 2
    device_count = torch.cuda.device_count()
    if device_count < 2:
        print(f"verify_gpu: need >=2 GPUs for distributed gates, found {device_count}")
        return 2

    # Never oversubscribe: two NCCL ranks on one GPU deadlock rather than error.
    world_sizes = [ws for ws in (2, 4) if ws <= device_count]

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    junit_path = out_dir / "gpu_gates.junit.xml"

    env = dict(
        os.environ,
        MINIDIST_DEVICE="cuda",
        MINIDIST_WORLD_SIZES=",".join(str(w) for w in world_sizes),
    )
    started = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit_path}"],
        cwd=REPO_ROOT,
        env=env,
    )
    duration = time.time() - started

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": ".".join(str(v) for v in torch.cuda.nccl.version()),
        "gpus": [torch.cuda.get_device_name(i) for i in range(device_count)],
        "world_sizes": world_sizes,
        "pytest_exit_code": proc.returncode,
        "passed": proc.returncode == 0,
        "duration_s": round(duration, 1),
    }
    (out_dir / "gpu_verify.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
