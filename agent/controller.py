"""
agent/controller.py

Thin wrapper around python-osc that sends parameter values to a
running Faust binary.  The binary must have been launched with -osc
(faust2jack adds this automatically).

Faust OSC defaults:
  receive port : 5510   (binary listens here)
  send port    : 5511   (binary sends UI updates here — not used by us)
"""

import time
from typing import Dict

from pythonosc import udp_client

from .params import FaustParams


class OSCController:
    """
    Sends named parameter values to a Faust process over OSC.

    controller = OSCController(params, host="127.0.0.1", port=5510)
    controller.send({"hp_freq": 300.0, "lp_freq": 1200.0})
    """

    def __init__(
        self,
        params: FaustParams,
        host: str = "127.0.0.1",
        port: int = 5510,
        inter_message_delay: float = 0.002,   # small gap avoids UDP loss
    ):
        self._params = params
        self._client = udp_client.SimpleUDPClient(host, port)
        self._delay = inter_message_delay
        print(f"[OSCController] Targeting Faust at {host}:{port}")

    def send(self, values: Dict[str, float]):
        """
        Send a dict of {param_name: value} to the synth.
        Values are clamped to the parameter's declared range.
        """
        for name, value in values.items():
            p = self._params[name]
            clamped = p.clamp(value)
            self._client.send_message(p.osc_address, clamped)
            if self._delay:
                time.sleep(self._delay)

    def send_vector(self, vec, names=None):
        """Send a numpy vector.  Uses FaustParams ordering by default."""
        if names is None:
            names = self._params.names()
        self.send(dict(zip(names, vec.tolist())))

    def reset_to_defaults(self):
        defaults = {name: p.default for name, p in self._params.items()}
        self.send(defaults)
        print("[OSCController] Reset to defaults")
