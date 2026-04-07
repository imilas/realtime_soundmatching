"""
agent/params.py

Parses the JSON emitted by `faust -json my_synth.dsp`.
Gives the agent a clean view of the action space without knowing
anything specific about the synth being controlled.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Param:
    name: str
    osc_address: str
    min_val: float
    max_val: float
    default: float
    step: float

    def clamp(self, v: float) -> float:
        return float(np.clip(v, self.min_val, self.max_val))

    def normalize(self, v: float) -> float:
        """Map real value → [0, 1]"""
        r = self.max_val - self.min_val
        return (v - self.min_val) / r if r else 0.0

    def denormalize(self, v: float) -> float:
        """Map [0, 1] → real value"""
        return self.min_val + v * (self.max_val - self.min_val)


import numpy as np   # imported here to avoid circular issue above


class FaustParams:
    """
    Load a Faust JSON metadata file and expose the full parameter space.

    Usage
    -----
    params = FaustParams("bandpass_noise.json")
    for name, p in params.items():
        print(name, p.osc_address, p.min_val, p.max_val)
    """

    def __init__(self, json_path: str):
        with open(json_path) as f:
            meta = json.load(f)
        self._params: Dict[str, Param] = {}
        self._parse_ui(meta.get("ui", []))

        if not self._params:
            raise ValueError(f"No slider/nentry parameters found in {json_path}")

        print(f"[FaustParams] Loaded {len(self._params)} parameter(s) from {json_path}")
        for p in self._params.values():
            print(f"  {p.name:20s}  {p.osc_address:40s}  [{p.min_val}, {p.max_val}]  default={p.default}")

    # ------------------------------------------------------------------
    # Internal JSON walker
    # ------------------------------------------------------------------
    def _parse_ui(self, items: list):
        for item in items:
            t = item.get("type", "")
            if t in ("hslider", "vslider", "nentry"):
                addr = item.get("address", f"/{item['label']}")
                p = Param(
                    name=item["label"],
                    osc_address=addr,
                    min_val=float(item["min"]),
                    max_val=float(item["max"]),
                    default=float(item["init"]),
                    step=float(item["step"]),
                )
                self._params[p.name] = p
            elif "items" in item:
                self._parse_ui(item["items"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def items(self):
        return self._params.items()

    def __getitem__(self, name: str) -> Param:
        return self._params[name]

    def names(self) -> List[str]:
        return list(self._params.keys())

    def bounds(self) -> Tuple[List[float], List[float]]:
        """Returns (lowers, uppers) parallel lists for scipy/CMA."""
        lowers = [p.min_val for p in self._params.values()]
        uppers = [p.max_val for p in self._params.values()]
        return lowers, uppers

    def defaults_vector(self) -> np.ndarray:
        return np.array([p.default for p in self._params.values()])

    def vector_to_dict(self, vec: np.ndarray) -> Dict[str, float]:
        return {name: float(vec[i]) for i, name in enumerate(self._params)}

    def clamp_vector(self, vec: np.ndarray) -> np.ndarray:
        lowers, uppers = self.bounds()
        return np.clip(vec, lowers, uppers)
