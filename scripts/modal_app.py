"""Run any minidist command on Modal GPUs; results/ files come back locally.

Usage (from the repo root):

    modal run scripts/modal_app.py --cmd "python scripts/verify_gpu.py"
    modal run scripts/modal_app.py --cmd "python scripts/bench_comm.py --device cuda --world-size 2"
    MINIDIST_GPU=L4:4 modal run scripts/modal_app.py --cmd "..."

The GPU spec comes from MINIDIST_GPU (default T4:2) and is baked at import
time — set it in the shell, not inside the container. The whole repo (minus
venv/git/results) is mounted into the container; anything the command writes
under results/ is copied back into the local results/ directory, so the
workflow is identical to running locally: one command, files out.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

GPU = os.environ.get("MINIDIST_GPU", "T4:2")
REPO_ROOT = Path(__file__).resolve().parents[1]

app = modal.App("minidist")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0", "numpy", "pytest")
    .add_local_dir(
        REPO_ROOT,
        "/root/minidist",
        ignore=[".venv/**", ".git/**", "logs/**", "results/**", "**/__pycache__/**"],
    )
)


@app.function(gpu=GPU, image=image, timeout=3600)
def run_remote(cmd: str) -> tuple[int, str, dict[str, str]]:
    env = dict(os.environ, PYTHONPATH="/root/minidist/src")
    proc = subprocess.run(
        cmd, shell=True, cwd="/root/minidist", env=env, capture_output=True, text=True
    )
    files: dict[str, str] = {}
    results = Path("/root/minidist/results")
    if results.exists():
        for p in results.rglob("*"):
            if p.is_file() and p.suffix in (".json", ".csv", ".xml"):
                files[str(p.relative_to(results))] = p.read_text()
    output = proc.stdout[-8000:] + ("\n--- stderr ---\n" + proc.stderr[-8000:] if proc.stderr else "")
    return proc.returncode, output, files


@app.local_entrypoint()
def main(cmd: str = "python scripts/verify_gpu.py") -> None:
    print(f"[modal_app] gpu={GPU} cmd={cmd!r}")
    code, output, files = run_remote.remote(cmd)
    print(output)
    for rel, content in files.items():
        dest = REPO_ROOT / "results" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        print(f"[modal_app] saved results/{rel}")
    if code != 0:
        raise SystemExit(code)
