"""
Pure Python orchestration for optimization experiments.

ExperimentRunner consumes an ExperimentConfig (program name + per-param spec +
target source) and exposes:
  - compute_landscape() — offline render-based loss curve over the sweep param
  - run_optimization() — realtime JACK+OSC loop, launches the synth itself

No Qt or GUI dependencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from scipy.signal import resample_poly

from agent.capture import JackCapture
from agent.controller import OSCController
from agent.params import FaustParams
from agents.base import AgentBase
from synths.build import SynthBuild, prepare, running_synth
from utils.faust_renderer import render
from utils.io import load_audio
from utils.loss_functions import ALL_LOSSES, multi_resolution_spectral_loss


TARGET_RENDERED = "rendered"
TARGET_WAV = "wav"


@dataclass
class ExperimentConfig:
    """Complete spec of one experiment.

    init_params and target_params are concrete values (post-resolve). For frozen
    params, init == target == frozen value. For non-frozen params, init and
    target are independent.

    target_source:
      "rendered" → target audio is rendered offline from the same program at
                   target_params (in-domain experiment).
      "wav"      → target audio is loaded from target_wav_path (out-of-domain).
    """
    program_name: str
    init_params: dict[str, float]
    target_params: dict[str, float]
    frozen_params: set[str] = field(default_factory=set)
    target_source: str = TARGET_RENDERED
    target_wav_path: str | None = None
    loss_name: str = "Multi-Res Spectral"

    sample_rate: int = 44100
    eval_blocks: int = 32
    block_size: int = 1024
    settle_time: float = 0.08
    landscape_steps: int = 80
    osc_host: str = "127.0.0.1"
    osc_port: int = 5510

    def __post_init__(self):
        if self.target_source not in (TARGET_RENDERED, TARGET_WAV):
            raise ValueError(
                f"target_source must be {TARGET_RENDERED!r} or {TARGET_WAV!r}, "
                f"got {self.target_source!r}"
            )
        if self.target_source == TARGET_WAV and not self.target_wav_path:
            raise ValueError("target_source='wav' requires target_wav_path")
        if self.loss_name not in ALL_LOSSES:
            raise ValueError(
                f"Unknown loss_name {self.loss_name!r}. "
                f"Available: {sorted(ALL_LOSSES)}"
            )


@dataclass
class LandscapePoint:
    index: int
    value: float
    loss: float


@dataclass
class OptimizationSnapshot:
    iteration: int
    current_value: float
    current_loss: float
    best_value: float
    best_loss: float
    history_values: list[float]
    history_losses: list[float]
    current_params: dict[str, float]


class ExperimentRunner:
    """Orchestrates one experiment configuration."""

    def __init__(self, config: ExperimentConfig, repo_root: Path | None = None):
        self.config = config
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.build: SynthBuild = prepare(config.program_name)
        self.params = FaustParams(str(self.build.json_path))

        non_frozen = [n for n in self.params.names() if n not in config.frozen_params]
        if not non_frozen:
            raise ValueError("All parameters are frozen — nothing to optimize.")
        if len(non_frozen) > 1:
            raise ValueError(
                "Current agents support a single optimization variable. "
                f"Freeze all but one parameter. Non-frozen: {non_frozen}"
            )
        self.sweep_param: str = non_frozen[0]

        for required in (config.init_params, config.target_params):
            missing = set(self.params.names()) - set(required.keys())
            if missing:
                raise ValueError(f"Missing values for parameters: {sorted(missing)}")
        self.loss_fn = ALL_LOSSES.get(config.loss_name, multi_resolution_spectral_loss)

    @property
    def jack_port(self) -> str:
        return f"{self.build.jack_client_name}:out_0"

    def _n_samples(self) -> int:
        return self.config.eval_blocks * self.config.block_size

    def _resample(self, audio: np.ndarray, src_sr: int) -> np.ndarray:
        if src_sr == self.config.sample_rate:
            return audio
        g = gcd(self.config.sample_rate, src_sr)
        return resample_poly(audio, self.config.sample_rate // g, src_sr // g)

    def _trim_or_pad(self, audio: np.ndarray, n: int) -> np.ndarray:
        if len(audio) < n:
            out = np.zeros(n, dtype=np.float64)
            out[: len(audio)] = audio
            return out
        return audio[:n].astype(np.float64)

    def _load_target_audio(self) -> np.ndarray:
        n = self._n_samples()
        if self.config.target_source == TARGET_RENDERED:
            duration_s = n / self.config.sample_rate
            audio = render(
                str(self.build.dsp_path),
                params=self.config.target_params,
                duration_s=duration_s,
                sample_rate=self.config.sample_rate,
            )
        else:
            audio, sr = load_audio(self.config.target_wav_path)
            audio = self._resample(audio, sr)
        return self._trim_or_pad(audio, n)

    def _compute_audio_loss(self, target: np.ndarray, audio: np.ndarray) -> float:
        m = min(len(target), len(audio))
        return self.loss_fn(
            target[:m], audio[:m], sample_rate=self.config.sample_rate
        )

    def _render_params_audio(self, params: dict[str, float], n_samples: int) -> np.ndarray:
        duration_s = n_samples / self.config.sample_rate
        audio = render(
            str(self.build.dsp_path),
            params=params,
            duration_s=duration_s,
            sample_rate=self.config.sample_rate,
        )
        return self._trim_or_pad(audio, n_samples)

    def _measure_params_loss(self, target_audio: np.ndarray, params: dict[str, float]) -> float:
        audio = self._render_params_audio(params, len(target_audio))
        return self._compute_audio_loss(target_audio, audio)

    def compute_landscape(self, n_steps: int | None = None) -> Iterator[LandscapePoint]:
        """Sweep the non-frozen param across its range and emit (value, loss) points.

        Uses offline rendering — does not require the synth process to be running.
        """
        if n_steps is None:
            n_steps = self.config.landscape_steps

        target_audio = self._load_target_audio()
        sweep_meta = self.params[self.sweep_param]
        sweep_values = np.linspace(sweep_meta.min_val, sweep_meta.max_val, n_steps)
        duration_s = len(target_audio) / self.config.sample_rate

        # Frozen params held at their init value (== target for frozen).
        frozen = {n: self.config.init_params[n] for n in self.config.frozen_params}

        for idx, value in enumerate(sweep_values):
            params = {**frozen, self.sweep_param: float(value)}
            audio = render(
                str(self.build.dsp_path),
                params=params,
                duration_s=duration_s,
                sample_rate=self.config.sample_rate,
            )
            loss = self._compute_audio_loss(target_audio, audio)
            yield LandscapePoint(idx, float(value), float(loss))

    def run_optimization(
        self,
        agent: AgentBase,
        stop_check: Callable[[], bool] | None = None,
    ) -> Iterator[OptimizationSnapshot]:
        """Realtime optimization loop. Launches the synth and tears it down on exit."""
        if stop_check is None:
            stop_check = lambda: False

        target_audio = self._load_target_audio()
        capture: JackCapture | None = None

        with running_synth(self.config.program_name) as build:
            controller = OSCController(
                self.params,
                host=self.config.osc_host,
                port=self.config.osc_port,
            )
            try:
                capture = JackCapture(client_name="experiment_runner")
                capture.start(self.jack_port)

                # Send all params (frozen + initial sweep) before measuring.
                current_params = dict(self.config.init_params)
                controller.send(current_params)
                capture.flush()
                if self.config.settle_time:
                    time.sleep(self.config.settle_time)

                audio = capture.get_n_blocks(self.config.eval_blocks)
                current_loss = self._compute_audio_loss(target_audio, audio)
                agent.update(current_params[self.sweep_param], current_loss)

                yield self._snapshot(current_params, current_loss, agent)

                sweep_meta = self.params[self.sweep_param]
                param_range = (sweep_meta.min_val, sweep_meta.max_val)

                while not stop_check():
                    candidates = agent.step(param_range)
                    if not isinstance(candidates, tuple):
                        candidates = (candidates,)

                    trial_losses = []
                    for candidate in candidates:
                        if stop_check():
                            break
                        controller.send({self.sweep_param: candidate})
                        capture.flush()
                        if self.config.settle_time:
                            time.sleep(self.config.settle_time)
                        audio = capture.get_n_blocks(self.config.eval_blocks)
                        trial_losses.append(self._compute_audio_loss(target_audio, audio))

                    if stop_check():
                        break

                    if hasattr(agent, "accept_trial") and len(trial_losses) >= 2:
                        choice = agent.accept_trial(trial_losses[0], trial_losses[1])
                        if choice == 1:
                            current_params[self.sweep_param] = candidates[0]
                            current_loss = trial_losses[0]
                        elif choice == 2:
                            current_params[self.sweep_param] = candidates[1]
                            current_loss = trial_losses[1]
                    elif trial_losses:
                        current_params[self.sweep_param] = candidates[0]
                        current_loss = trial_losses[0]

                    agent.update(current_params[self.sweep_param], current_loss)
                    yield self._snapshot(current_params, current_loss, agent)
            finally:
                if capture is not None:
                    try:
                        capture.stop()
                    except Exception:
                        pass

    def run_offline_optimization(
        self,
        agent: AgentBase,
        max_iterations: int,
        stop_check: Callable[[], bool] | None = None,
    ) -> Iterator[OptimizationSnapshot]:
        """Offline render-only optimization loop.

        No JACK or OSC required. Each candidate is rendered directly and scored
        against the target audio with the configured loss function.
        """
        if stop_check is None:
            stop_check = lambda: False

        target_audio = self._load_target_audio()
        current_params = dict(self.config.init_params)
        current_loss = self._measure_params_loss(target_audio, current_params)
        agent.update(current_params[self.sweep_param], current_loss)
        yield self._snapshot(current_params, current_loss, agent)

        sweep_meta = self.params[self.sweep_param]
        param_range = (sweep_meta.min_val, sweep_meta.max_val)

        for _ in range(max_iterations):
            if stop_check():
                break

            candidates = agent.step(param_range)
            if not isinstance(candidates, tuple):
                candidates = (candidates,)

            trial_losses = []
            for candidate in candidates:
                if stop_check():
                    break
                trial_params = dict(current_params)
                trial_params[self.sweep_param] = candidate
                trial_losses.append(self._measure_params_loss(target_audio, trial_params))

            if stop_check():
                break

            if hasattr(agent, "accept_trial") and len(trial_losses) >= 2:
                choice = agent.accept_trial(trial_losses[0], trial_losses[1])
                if choice == 1:
                    current_params[self.sweep_param] = candidates[0]
                    current_loss = trial_losses[0]
                elif choice == 2:
                    current_params[self.sweep_param] = candidates[1]
                    current_loss = trial_losses[1]
            elif trial_losses:
                current_params[self.sweep_param] = candidates[0]
                current_loss = trial_losses[0]

            agent.update(current_params[self.sweep_param], current_loss)
            yield self._snapshot(current_params, current_loss, agent)

    def _snapshot(
        self, current_params: dict[str, float], current_loss: float, agent: AgentBase
    ) -> OptimizationSnapshot:
        return OptimizationSnapshot(
            iteration=agent.iteration,
            current_value=current_params[self.sweep_param],
            current_loss=current_loss,
            best_value=agent.best_value,
            best_loss=agent.best_loss,
            history_values=list(agent.history_values),
            history_losses=list(agent.history_losses),
            current_params=dict(current_params),
        )
