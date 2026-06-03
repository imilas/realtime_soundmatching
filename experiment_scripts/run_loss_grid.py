#!/usr/bin/env python3
"""Run synth x loss x method experiment grids.

This is a Python replacement for the larger shell cross-product. It launches
one `paper_experiments/run_paper.py` subprocess per cell, so each cell remains
resumable through the existing per-cell pickle files.

Examples
--------
    python experiment_scripts/run_loss_grid.py --trials 10 --budget 50 --jobs 2
    python experiment_scripts/run_loss_grid.py --methods CMA-ES BO --losses all
    python experiment_scripts/run_loss_grid.py --losses "L2 Spectral" SIMSE_Spec
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "mpl"))

from paper_experiments.config import METHODS, SYNTHS  # noqa: E402
from paper_experiments.run_paper import GD_SUPPORTED_LOSSES  # noqa: E402
from utils.loss_functions import ALL_LOSSES  # noqa: E402


@dataclass(frozen=True)
class Cell:
    synth: str
    loss: str
    method: str


def _parse_list(values: list[str] | None, default: list[str], choices: set[str], label: str) -> list[str]:
    if not values:
        return default
    if len(values) == 1 and values[0].lower() == "all":
        return default
    unknown = [v for v in values if v not in choices]
    if unknown:
        raise SystemExit(f"Unknown {label}: {unknown}. Available: {sorted(choices)}")
    return values


def _build_cells(
    synths: list[str],
    losses: list[str],
    methods: list[str],
    skip_unsupported_gd: bool,
) -> tuple[list[Cell], list[Cell]]:
    runnable: list[Cell] = []
    skipped: list[Cell] = []
    for synth in synths:
        for loss in losses:
            for method in methods:
                is_gd = METHODS[method][0]
                if is_gd and loss not in GD_SUPPORTED_LOSSES and skip_unsupported_gd:
                    skipped.append(Cell(synth, loss, method))
                    continue
                runnable.append(Cell(synth, loss, method))
    return runnable, skipped


def _cell_command(cell: Cell, trials: int, budget: int) -> list[str]:
    return [
        sys.executable,
        "paper_experiments/run_paper.py",
        "--synth",
        cell.synth,
        "--loss",
        cell.loss,
        "--method",
        cell.method,
        "--trials",
        str(trials),
        "--budget",
        str(budget),
    ]


def _run_cell(cell: Cell, trials: int, budget: int, threads: int) -> tuple[Cell, int, float]:
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = str(threads)
    env["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=true "
        f"intra_op_parallelism_threads={threads}"
    )

    cmd = _cell_command(cell, trials, budget)
    started = time.perf_counter()
    print(f"==> {cell.synth} | {cell.loss} | {cell.method}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO_DIR, env=env)
    elapsed = time.perf_counter() - started
    return cell, proc.returncode, elapsed


def _write_timing(path: Path, cell: Cell, elapsed: float, status: int) -> None:
    with path.open("a") as f:
        f.write(f"{cell.synth}\t{cell.loss}\t{cell.method}\t{elapsed:.3f}\t{status}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--synths", nargs="+", default=None, help='Synths, or "all".')
    parser.add_argument("--losses", nargs="+", default=None, help='Losses, or "all". Quote names with spaces.')
    parser.add_argument("--methods", nargs="+", default=None, help='Methods, or "all".')
    parser.add_argument("--timing-file", default=None)
    parser.add_argument(
        "--run-unsupported-gd",
        action="store_true",
        help="Try GD for losses outside its differentiable loss set instead of skipping them.",
    )
    args = parser.parse_args()

    synths = _parse_list(args.synths, list(SYNTHS), set(SYNTHS), "synths")
    losses = _parse_list(args.losses, list(ALL_LOSSES), set(ALL_LOSSES), "losses")
    methods = _parse_list(args.methods, list(METHODS), set(METHODS), "methods")

    cells, skipped = _build_cells(
        synths=synths,
        losses=losses,
        methods=methods,
        skip_unsupported_gd=not args.run_unsupported_gd,
    )

    results_dir = REPO_DIR / "paper_experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timing_file = (
        Path(args.timing_file)
        if args.timing_file
        else results_dir / f"loss_grid_timings_{time.strftime('%Y%m%d_%H%M%S')}.tsv"
    )
    timing_file.parent.mkdir(parents=True, exist_ok=True)
    timing_file.write_text("synth\tloss\tmethod\telapsed_s\tstatus\n")

    print(f"Synths:  {' '.join(synths)}")
    print(f"Losses:  {', '.join(losses)}")
    print(f"Methods: {' '.join(methods)}")
    print(f"Cells:   {len(cells)} runnable, {len(skipped)} skipped")
    print(f"Trials:  {args.trials}")
    print(f"Budget:  {args.budget}")
    print(f"Jobs:    {args.jobs}")
    print(f"Threads: {args.threads} per job")
    print(f"Timing:  {timing_file}")
    if skipped:
        print("Skipped unsupported GD cells:")
        for cell in skipped:
            print(f"  {cell.synth} | {cell.loss} | {cell.method}")
    print("", flush=True)

    failed = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {
            pool.submit(_run_cell, cell, args.trials, args.budget, args.threads): cell
            for cell in cells
        }
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                cell, status, elapsed = future.result()
                _write_timing(timing_file, cell, elapsed, status)
                if status != 0:
                    failed += 1
                print(
                    f"<== {cell.synth} | {cell.loss} | {cell.method} "
                    f"| status={status} | {elapsed:.1f}s",
                    flush=True,
                )

    print(f"\nFinished {len(cells)} cells with {failed} failures.")
    print(f"Timing details: {timing_file}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
