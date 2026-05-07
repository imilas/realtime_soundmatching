"""
synths/program.py

SynthProgram: a Faust DSP program defined by a .dsp template file.
Parameter bounds are parsed from the hslider definitions in the code.
Target and init params are set at experiment time, not here.
"""

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PROGRAMS_DIR = Path(__file__).parent / "programs"


@dataclass
class SynthProgram:
    name: str
    faust_template: str  # Faust code with {param_name} placeholders for default values

    @property
    def param_ranges(self) -> dict[str, tuple[float, float]]:
        """Parse (min, max) bounds for each parameter from hslider definitions."""
        ranges = {}
        pattern = r'hslider\s*\(\s*"([^"]+)"\s*,\s*\{[^}]*\}\s*,\s*([^,]+),\s*([^,]+)\s*,'
        for match in re.finditer(pattern, self.faust_template):
            name = match.group(1)
            ranges[name] = (float(match.group(2)), float(match.group(3)))
        return ranges

    def random_params(self) -> dict[str, float]:
        """Sample parameter values uniformly at random within their bounds."""
        return {
            name: float(np.random.uniform(lo, hi))
            for name, (lo, hi) in self.param_ranges.items()
        }

    def instantiate(self, params: dict[str, float]) -> str:
        """Return Faust code with {param} placeholders filled in."""
        code = self.faust_template
        for name, value in params.items():
            code = code.replace(f"{{{name}}}", str(value))
        return code

    def write_dsp(self, params: dict[str, float], path: str | None = None) -> str:
        """Write instantiated Faust code to a .dsp file. Returns the file path."""
        code = self.instantiate(params)
        if path is None:
            f = tempfile.NamedTemporaryFile(suffix=".dsp", delete=False, mode="w")
            f.write(code)
            f.close()
            return f.name
        Path(path).write_text(code)
        return path


def load_program(dsp_path: str | Path) -> SynthProgram:
    """Load a SynthProgram from a .dsp template file."""
    dsp_path = Path(dsp_path)
    return SynthProgram(
        name=dsp_path.stem,
        faust_template=dsp_path.read_text(),
    )


def list_programs() -> list[str]:
    """List available program names (stems of .dsp files in synths/programs/)."""
    return sorted(p.stem for p in _PROGRAMS_DIR.glob("*.dsp"))


def get_program(name: str) -> SynthProgram:
    """Load a program by name."""
    path = _PROGRAMS_DIR / f"{name}.dsp"
    if not path.exists():
        raise ValueError(f"Unknown program: {name!r}. Available: {list_programs()}")
    return load_program(path)


# Pre-loaded registry
PROGRAMS: dict[str, SynthProgram] = {
    name: get_program(name) for name in list_programs()
}
