"""
agent/capture.py

Minimal JACK client that copies audio blocks into a thread-safe queue.
The realtime callback does *nothing* except memcopy + queue.put_nowait.
All heavy work (FFT, loss, OSC) happens in normal Python threads.
"""

import jack
import numpy as np
import queue
import threading
import time


class JackCapture:
    """
    Connects to one or two JACK output ports and streams audio blocks
    into a queue.Queue that any thread can read from.
    """

    def __init__(self, client_name: str = "agent_capture", queue_maxsize: int = 200):
        self.client = jack.Client(client_name)
        self.sample_rate: int = self.client.samplerate
        self.blocksize: int = self.client.blocksize

        self._queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._dropped = 0

        # Register one mono input port (we'll mix stereo down if needed)
        self._inport = self.client.inports.register("in_L")

        self.client.set_process_callback(self._rt_callback)
        self.client.set_shutdown_callback(self._shutdown)

    # ------------------------------------------------------------------
    # Realtime callback — must be allocation-free, no Python heavy work
    # ------------------------------------------------------------------
    def _rt_callback(self, frames: int):
        buf = self._inport.get_array().copy()   # float32 numpy view → copy
        try:
            self._queue.put_nowait(buf)
        except queue.Full:
            self._dropped += 1                  # silently drop; agent is slow

    def _shutdown(self, status, reason):
        print(f"[JackCapture] JACK shutdown: {reason}")

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def start(self, source_port: str):
        """
        Activate and connect to a Faust output port.

        source_port example: "bandpass_noise:output_0"
        """
        self.client.activate()
        self.client.connect(source_port, self._inport)
        print(f"[JackCapture] Connected {source_port} → {self._inport.name}")

    def stop(self):
        self.client.deactivate()
        self.client.close()

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------
    def get_block(self, timeout: float = 2.0) -> np.ndarray:
        """Block until next audio block is available, return float32 array."""
        return self._queue.get(timeout=timeout)

    def get_n_blocks(self, n: int, timeout: float = 5.0) -> np.ndarray:
        """Accumulate n consecutive blocks and return as one flat array."""
        chunks = []
        deadline = time.time() + timeout
        while len(chunks) < n:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("JackCapture.get_n_blocks timed out")
            chunks.append(self.get_block(timeout=remaining))
        return np.concatenate(chunks)

    def flush(self):
        """Discard all buffered blocks (call after sending new params)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
