"""
Synth programs and build helpers.

Programs are .dsp templates in synths/programs/ — each declares parameter
bounds via hslider definitions. The build helper compiles a program to a
JACK + OSC binary on demand and caches the result by content hash.

Public API:
  list_programs()        — names of available .dsp templates
  get_program(name)      — load a SynthProgram by name
  prepare(name)          — compile + emit JSON; returns SynthBuild
  running_synth(name)    — context manager: prepare + launch + cleanup
"""

from .build import SynthBuild, prepare, launch, running_synth
from .program import SynthProgram, get_program, list_programs

__all__ = [
    "SynthBuild",
    "SynthProgram",
    "get_program",
    "list_programs",
    "prepare",
    "launch",
    "running_synth",
]
