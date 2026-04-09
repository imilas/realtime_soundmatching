"""
agent/optimizer.py
"""

import time
import threading
import numpy as np
from dataclasses import dataclass, field

from .params import FaustParams
from .capture import JackCapture
from .controller import OSCController
from .features import compute_features, spectral_loss


@dataclass
class AgentState:
    iteration: int = 0
    best_loss: float = float("inf")
    current_params: dict = field(default_factory=dict)
    history: list = field(default_factory=list)


class BoilerplateOptimizer:
    """
    Minimal starting point for your own algorithm.

    Every `interval_blocks` JACK blocks:
      1. capture audio
      2. compute spectrogram + loss
      3. call step() — fill this in with your logic
      4. send returned param vector to the synth via OSC
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
        interval_blocks: int = 8,
        **kwargs,
    ):
        self.params = params
        self.capture = capture
        self.controller = controller
        self.target = target_features
        self.sr = sample_rate
        self.settle_time = settle_time
        self.eval_blocks = eval_blocks
        self.interval_blocks = interval_blocks
        self.state = AgentState()
        self._stop_event = threading.Event()

        self._current_vec = params.defaults_vector()

    # ------------------------------------------------------------------
    # TODO: implement your algorithm here
    # ------------------------------------------------------------------
    def step(self, spectrogram: np.ndarray, loss: float) -> np.ndarray:
        """
        Called every `interval_blocks` blocks with the latest spectrogram.

        Parameters
        ----------
        spectrogram : 1-D feature vector from compute_features()
        loss        : spectral_loss(spectrogram, self.target)

        Returns
        -------
        new_vec : parameter vector to send to the synth
                  (same shape as self._current_vec)
        """
        # --- replace this with your own update rule ---
        return self._current_vec   # no-op: just return current params

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, max_iterations: int = 10_000):
        print(f"[Boilerplate] Starting. interval={self.interval_blocks} blocks, "
              f"max_iters={max_iterations}")

        for i in range(max_iterations):
            if self._stop_event.is_set():
                break

            self.capture.flush()
            audio = self.capture.get_n_blocks(self.interval_blocks)

            feats = compute_features(audio, self.sr)
            loss = spectral_loss(feats, self.target)

            self.state.iteration = i
            if loss < self.state.best_loss:
                self.state.best_loss = loss
                self.state.current_params = self.params.vector_to_dict(self._current_vec)

            print(f"[{i:5d}] loss={loss:.5f}  params={self.state.current_params}")

            new_vec = self.step(feats, loss)
            self._current_vec = self.params.clamp_vector(new_vec)
            self.controller.send_vector(self._current_vec)
            time.sleep(self.settle_time)

        print(f"\n[Boilerplate] Done. Best loss={self.state.best_loss:.5f}")

    def stop(self):
        self._stop_event.set()
