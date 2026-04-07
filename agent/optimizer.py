"""
agent/optimizer.py

Realtime parameter optimizer.

Strategy: adaptive random-restart hillclimbing.
  - Maintain a current parameter vector and current loss
  - Each step: perturb → send → wait for audio to settle → evaluate
  - Accept if better (greedy), shrink step on failure, grow on success
  - Periodically restart from random point to escape local minima

To upgrade to CMA-ES, install `pip install cma` and switch to
CMAOptimizer (bottom of file).  The interface is identical.
"""

import time
import threading
import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass, field

from .params import FaustParams
from .capture import JackCapture
from .controller import OSCController
from .features import compute_features, spectral_loss


# ------------------------------------------------------------------
# Shared state (thread-safe enough for a single writer)
# ------------------------------------------------------------------

@dataclass
class AgentState:
    iteration: int = 0
    best_loss: float = float("inf")
    current_params: dict = field(default_factory=dict)
    history: list = field(default_factory=list)   # list of (iter, loss, params)


# ------------------------------------------------------------------
# Core hillclimbing optimizer
# ------------------------------------------------------------------

class HillClimbOptimizer:
    """
    Greedy adaptive hillclimbing over the Faust parameter space.

    Parameters
    ----------
    params           : FaustParams — action space description
    capture          : JackCapture — live audio source (the synth being tuned)
    controller       : OSCController — sends params to the synth
    target_features  : np.ndarray — reference log-PSD to match
    sample_rate      : int
    settle_time      : float — seconds to wait after sending params (audio settling)
    eval_blocks      : int — JACK blocks to capture for each evaluation
    initial_sigma    : float — initial perturbation as fraction of param range
    sigma_grow       : float — multiply sigma by this on accept
    sigma_shrink     : float — multiply sigma by this on reject
    sigma_min        : float — lower bound on sigma
    restart_every    : int — force random restart every N iterations
    callback         : optional fn(state) called each iteration
    """

    def __init__(
        self,
        params: FaustParams,
        capture: JackCapture,
        controller: OSCController,
        target_features: np.ndarray,
        sample_rate: int = 44100,
        settle_time: float = 0.08,
        eval_blocks: int = 8,
        initial_sigma: float = 0.15,
        sigma_grow: float = 1.10,
        sigma_shrink: float = 0.92,
        sigma_min: float = 0.001,
        restart_every: int = 200,
        callback: Optional[Callable] = None,
    ):
        self.params = params
        self.capture = capture
        self.controller = controller
        self.target = target_features
        self.sr = sample_rate
        self.settle_time = settle_time
        self.eval_blocks = eval_blocks
        self.sigma = initial_sigma
        self.sigma_grow = sigma_grow
        self.sigma_shrink = sigma_shrink
        self.sigma_min = sigma_min
        self.restart_every = restart_every
        self.callback = callback

        self.state = AgentState()
        self._stop_event = threading.Event()

        lowers, uppers = params.bounds()
        self._lowers = np.array(lowers)
        self._uppers = np.array(uppers)
        self._ranges = self._uppers - self._lowers

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate(self, vec: np.ndarray) -> float:
        """Send params, wait, capture audio, return loss."""
        self.controller.send_vector(vec)
        time.sleep(self.settle_time)
        self.capture.flush()                        # discard pre-settle audio
        audio = self.capture.get_n_blocks(self.eval_blocks)
        feats = compute_features(audio, self.sr)
        return spectral_loss(feats, self.target)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, max_iterations: int = 10_000):
        """Blocking optimization loop.  Call stop() from another thread to exit."""
        state = self.state

        # Start from defaults
        current_vec = self.params.defaults_vector()
        current_loss = self._evaluate(current_vec)
        state.current_params = self.params.vector_to_dict(current_vec)
        state.best_loss = current_loss

        print(f"[Optimizer] Starting loss: {current_loss:.5f}")
        print(f"[Optimizer] Running for up to {max_iterations} iterations…")

        for i in range(max_iterations):
            if self._stop_event.is_set():
                break

            state.iteration = i

            # Periodic restart from random point
            if i > 0 and i % self.restart_every == 0:
                print(f"[Optimizer] Restart at iter {i}, best_loss={state.best_loss:.5f}")
                current_vec = self._random_params()
                current_loss = self._evaluate(current_vec)

            # Propose candidate: current + Gaussian noise scaled by sigma * range
            noise = np.random.randn(len(current_vec)) * self.sigma * self._ranges
            candidate = self.params.clamp_vector(current_vec + noise)
            candidate_loss = self._evaluate(candidate)

            if candidate_loss < current_loss:
                current_vec = candidate
                current_loss = candidate_loss
                self.sigma = min(self.sigma * self.sigma_grow, 1.0)
                marker = "✓"
            else:
                self.sigma = max(self.sigma * self.sigma_shrink, self.sigma_min)
                marker = "✗"

            # Track best ever
            if current_loss < state.best_loss:
                state.best_loss = current_loss
                state.current_params = self.params.vector_to_dict(current_vec)

            state.history.append((i, current_loss, state.current_params.copy()))

            # Console log
            param_str = "  ".join(f"{k}={v:.1f}" for k, v in state.current_params.items())
            print(f"[{i:5d}] {marker} loss={current_loss:.5f}  σ={self.sigma:.4f}  {param_str}")

            if self.callback:
                self.callback(state)

        # Apply best found
        self.controller.send_vector(
            np.array([state.current_params[n] for n in self.params.names()])
        )
        print(f"\n[Optimizer] Done. Best loss={state.best_loss:.5f}")
        print(f"[Optimizer] Best params: {state.current_params}")

    def stop(self):
        self._stop_event.set()

    def _random_params(self) -> np.ndarray:
        r = np.random.rand(len(self._lowers))
        return self._lowers + r * self._ranges


# ------------------------------------------------------------------
# CMA-ES upgrade (requires: pip install cma)
# ------------------------------------------------------------------

class CMAOptimizer:
    """
    Drop-in replacement for HillClimbOptimizer using CMA-ES.
    More sample-efficient but evaluates one candidate at a time
    (we can't parallelize because each eval takes real time).

    Install: pip install cma
    """

    def __init__(self, params, capture, controller, target_features,
                 sample_rate=44100, settle_time=0.08, eval_blocks=8,
                 initial_sigma=0.3, **kwargs):
        self.params = params
        self.capture = capture
        self.controller = controller
        self.target = target_features
        self.sr = sample_rate
        self.settle_time = settle_time
        self.eval_blocks = eval_blocks
        self.initial_sigma = initial_sigma
        self.state = AgentState()
        self._stop_event = threading.Event()

    def _evaluate(self, vec):
        self.controller.send_vector(vec)
        time.sleep(self.settle_time)
        self.capture.flush()
        audio = self.capture.get_n_blocks(self.eval_blocks)
        feats = compute_features(audio, self.sr)
        return spectral_loss(feats, self.target)

    def run(self, max_iterations=10_000):
        import cma

        lowers, uppers = self.params.bounds()
        x0 = self.params.defaults_vector().tolist()
        opts = {
            "bounds": [lowers, uppers],
            "maxiter": max_iterations,
            "verbose": -9,   # suppress CMA stdout; we log ourselves
            "popsize": 1,    # sequential evaluation (realtime constraint)
        }
        es = cma.CMAEvolutionStrategy(x0, self.initial_sigma, opts)

        i = 0
        while not es.stop() and not self._stop_event.is_set():
            solutions = es.ask()
            losses = []
            for sol in solutions:
                clamped = self.params.clamp_vector(np.array(sol))
                loss = self._evaluate(clamped)
                losses.append(loss)
                self.state.iteration = i
                if loss < self.state.best_loss:
                    self.state.best_loss = loss
                    self.state.current_params = self.params.vector_to_dict(clamped)
                param_str = "  ".join(f"{k}={v:.1f}" for k, v in
                                       self.params.vector_to_dict(clamped).items())
                print(f"[{i:5d}] loss={loss:.5f}  {param_str}")
                i += 1
            es.tell(solutions, losses)

        print(f"\n[CMAOptimizer] Done. Best: {self.state.current_params}")

    def stop(self):
        self._stop_event.set()
