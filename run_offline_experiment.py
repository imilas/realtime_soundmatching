#!/usr/bin/env python
"""
Run one offline render-only experiment from a JSON config.

Usage:
    python run_offline_experiment.py --config experiments/examples/offline_config.example.json --output-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from agents.hillclimber import HillClimberAgent
from agents.q_learning import QLearningAgent
from agents.random_search import RandomSearchAgent
from experiments.runner import ExperimentConfig, ExperimentRunner, TARGET_RENDERED, TARGET_WAV
from synths import get_program


MIN_DISTANCE_FRAC = 0.10
MAX_RESAMPLE_ATTEMPTS = 100


@dataclass
class ResolvedParam:
    name: str
    frozen: bool
    init: float
    target: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-").lower()


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _serialize_experiment_config(config: ExperimentConfig) -> dict:
    return {
        "program_name": config.program_name,
        "init_params": dict(config.init_params),
        "target_params": dict(config.target_params),
        "frozen_params": sorted(config.frozen_params),
        "target_source": config.target_source,
        "target_wav_path": config.target_wav_path,
        "loss_name": config.loss_name,
        "sample_rate": config.sample_rate,
        "eval_blocks": config.eval_blocks,
        "block_size": config.block_size,
        "settle_time": config.settle_time,
        "landscape_steps": config.landscape_steps,
        "osc_host": config.osc_host,
        "osc_port": config.osc_port,
    }


def _parse_step(program, param_name: str) -> float:
    pattern = (
        r'hslider\s*\(\s*"' + re.escape(param_name)
        + r'"\s*,\s*\{[^}]*\}\s*,\s*[^,]+\s*,\s*[^,]+\s*,\s*([^\)]+)\)'
    )
    match = re.search(pattern, program.faust_template)
    if not match:
        return 0.0
    try:
        return float(match.group(1).strip())
    except ValueError:
        return 0.0


def _quantize(value: float, min_val: float, max_val: float, step: float) -> float:
    value = max(min_val, min(max_val, value))
    if step > 0:
        value = round((value - min_val) / step) * step + min_val
        value = max(min_val, min(max_val, value))
    return float(value)


def _resolve_params(config: dict) -> tuple[dict[str, float], dict[str, float], set[str], dict[str, dict], int, list[str]]:
    if "init_params" in config and "target_params" in config:
        init = {k: float(v) for k, v in config["init_params"].items()}
        target = {k: float(v) for k, v in config["target_params"].items()}
        frozen = set(config.get("frozen_params", []))
        details = {
            name: {
                "frozen": name in frozen,
                "init": init[name],
                "target": target[name],
                "source": "resolved",
            }
            for name in init
        }
        return init, target, frozen, details, 1, []

    program = get_program(config["program_name"])
    spec = config.get("parameter_config", {})
    seed = config.get("seed")
    rng = random.Random(seed)

    init: dict[str, float] = {}
    target: dict[str, float] = {}
    frozen: set[str] = set()
    details: dict[str, dict] = {}
    max_attempts = 0
    violations: list[str] = []

    for name, (min_val, max_val) in program.param_ranges.items():
        row = spec.get(name, {})
        step = _parse_step(program, name)
        is_frozen = bool(row.get("freeze", False))
        if is_frozen:
            frozen_value = row.get("frozen_value_override")
            if frozen_value is None:
                value = _quantize(rng.uniform(min_val, max_val), min_val, max_val, step)
                source = "random"
            else:
                value = _quantize(float(frozen_value), min_val, max_val, step)
                source = "override"
            init[name] = value
            target[name] = value
            frozen.add(name)
            details[name] = {"frozen": True, "init": value, "target": value, "source": source}
            max_attempts = max(max_attempts, 1)
            continue

        init_override = row.get("init_override")
        target_override = row.get("target_override")
        min_dist = MIN_DISTANCE_FRAC * (max_val - min_val)
        attempts = 0

        if init_override is not None and target_override is not None:
            init_value = _quantize(float(init_override), min_val, max_val, step)
            target_value = _quantize(float(target_override), min_val, max_val, step)
            attempts = 1
            violated = abs(init_value - target_value) < min_dist
        else:
            init_value = target_value = 0.0
            for i in range(MAX_RESAMPLE_ATTEMPTS):
                attempts = i + 1
                init_value = (
                    _quantize(float(init_override), min_val, max_val, step)
                    if init_override is not None
                    else _quantize(rng.uniform(min_val, max_val), min_val, max_val, step)
                )
                target_value = (
                    _quantize(float(target_override), min_val, max_val, step)
                    if target_override is not None
                    else _quantize(rng.uniform(min_val, max_val), min_val, max_val, step)
                )
                if abs(init_value - target_value) >= min_dist:
                    break
            violated = abs(init_value - target_value) < min_dist

        if violated:
            violations.append(name)
        init[name] = init_value
        target[name] = target_value
        details[name] = {
            "frozen": False,
            "init": init_value,
            "target": target_value,
            "source": {
                "init": "override" if init_override is not None else "random",
                "target": "override" if target_override is not None else "random",
            },
        }
        max_attempts = max(max_attempts, attempts)

    return init, target, frozen, details, max_attempts, violations


def _build_agent(agent_type: str, step_percent: float, agent_options: dict):
    if agent_type == "hillclimber":
        return HillClimberAgent(step_percent=step_percent)
    if agent_type == "random":
        return RandomSearchAgent(step_percent=step_percent)
    if agent_type == "q_learning":
        return QLearningAgent(step_percent=step_percent, **agent_options)
    raise ValueError(f"Unknown agent_type: {agent_type!r}")


def _best_params(snapshot, sweep_param: str, init_params: dict[str, float]) -> dict[str, float]:
    params = dict(init_params)
    params[sweep_param] = float(snapshot.best_value)
    return params


def _write_history_csv(path: Path, snapshots):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["iteration", "current_value", "current_loss", "best_value", "best_loss"],
        )
        writer.writeheader()
        for snap in snapshots:
            writer.writerow(
                {
                    "iteration": snap.iteration,
                    "current_value": snap.current_value,
                    "current_loss": snap.current_loss,
                    "best_value": snap.best_value,
                    "best_loss": snap.best_loss,
                }
            )


def _write_landscape_csv(path: Path, points):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "value", "loss"])
        writer.writeheader()
        for point in points:
            writer.writerow({"index": point.index, "value": point.value, "loss": point.loss})


def parse_args():
    parser = argparse.ArgumentParser(description="Run one offline synth experiment from JSON config")
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    parser.add_argument("--output-dir", required=True, help="Directory where run artifacts will be written")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).parent.resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    raw = _load_json(config_path)

    seed = raw.get("seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    init_params, target_params, frozen_params, resolution_details, max_attempts, violations = _resolve_params(raw)

    experiment_config = ExperimentConfig(
        program_name=raw["program_name"],
        init_params=init_params,
        target_params=target_params,
        frozen_params=frozen_params,
        target_source=raw.get("target_source", TARGET_RENDERED),
        target_wav_path=raw.get("target_wav_path"),
        loss_name=raw.get("loss_name", "Multi-Res Spectral"),
        sample_rate=int(raw.get("sample_rate", 44100)),
        eval_blocks=int(raw.get("eval_blocks", 32)),
        block_size=int(raw.get("block_size", 1024)),
        settle_time=float(raw.get("settle_time", 0.08)),
        landscape_steps=int(raw.get("landscape_steps", 80)),
        osc_host=str(raw.get("osc_host", "127.0.0.1")),
        osc_port=int(raw.get("osc_port", 5510)),
    )

    agent_type = raw["agent_type"]
    step_percent = float(raw.get("step_percent", 5.0))
    agent_options = dict(raw.get("agent_options", {}))
    max_iterations = int(raw.get("max_iterations", 100))
    compute_landscape = bool(raw.get("compute_landscape", True))

    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        f"_{_slug(raw['program_name'])}_{_slug(agent_type)}_{uuid.uuid4().hex[:8]}"
    )
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    started_at = _utc_now()
    runner = ExperimentRunner(experiment_config, repo_root)
    agent = _build_agent(agent_type, step_percent, agent_options)

    snapshots = list(runner.run_offline_optimization(agent, max_iterations=max_iterations))
    if not snapshots:
        raise RuntimeError("Offline optimization produced no snapshots")

    landscape_points = list(runner.compute_landscape()) if compute_landscape else []
    final_snapshot = snapshots[-1]
    initial_snapshot = snapshots[0]
    best_params = _best_params(final_snapshot, runner.sweep_param, experiment_config.init_params)

    _write_history_csv(run_dir / "history.csv", snapshots)
    if landscape_points:
        _write_landscape_csv(run_dir / "landscape.csv", landscape_points)

    summary = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "config_path": str(config_path),
        "mode": "offline",
        "agent_type": agent_type,
        "agent_options": agent_options,
        "step_percent": step_percent,
        "max_iterations": max_iterations,
        "compute_landscape": compute_landscape,
        "config": _serialize_experiment_config(experiment_config),
        "sweep_param": runner.sweep_param,
        "frozen_params": sorted(experiment_config.frozen_params),
        "resolution": {
            "seed": seed,
            "max_resampling_attempts": max_attempts,
            "min_distance_frac": MIN_DISTANCE_FRAC,
            "violations": violations,
            "parameter_details": resolution_details,
        },
        "initial_params": experiment_config.init_params,
        "target_params": experiment_config.target_params,
        "final_params": final_snapshot.current_params,
        "best_params": best_params,
        "metrics": {
            "iterations_recorded": len(snapshots),
            "initial_loss": initial_snapshot.current_loss,
            "final_loss": final_snapshot.current_loss,
            "best_loss": final_snapshot.best_loss,
            "final_minus_initial": final_snapshot.current_loss - initial_snapshot.current_loss,
            "best_minus_initial": final_snapshot.best_loss - initial_snapshot.current_loss,
            "improved_over_initial": final_snapshot.best_loss < initial_snapshot.current_loss,
        },
        "artifacts": {
            "history_csv": "history.csv",
            "landscape_csv": "landscape.csv" if landscape_points else None,
        },
    }
    _write_json(run_dir / "summary.json", summary)

    print(f"run_dir={run_dir}")
    print(f"program={experiment_config.program_name}")
    print(f"agent={agent_type}")
    print(f"loss={experiment_config.loss_name}")
    print(f"sweep_param={runner.sweep_param}")
    print(f"initial_loss={initial_snapshot.current_loss:.6f}")
    print(f"final_loss={final_snapshot.current_loss:.6f}")
    print(f"best_loss={final_snapshot.best_loss:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
