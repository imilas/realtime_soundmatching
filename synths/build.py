"""
synths/build.py

One-stop helper to compile a Faust synth template into a runnable
JACK + OSC binary, generate parameter JSON metadata, and (optionally)
launch it as a subprocess.

Compilation results are cached under synths/build/ keyed by template
content hash, so repeat calls are no-ops.

Typical usage:

    from synths.build import prepare, running_synth

    # Compile + JSON; idempotent.
    build = prepare("bandpass_noise")
    print(build.binary_path, build.json_path)

    # Launch in a context manager so the process is cleaned up.
    with running_synth("bandpass_noise") as build:
        # build.binary_path is running on JACK + OSC
        # connect with: jack_client = build.jack_client_name
        ...
"""

from __future__ import annotations

import hashlib
import fcntl
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .program import SynthProgram, get_program

_BUILD_DIR = Path(__file__).parent / "build"


@dataclass
class SynthBuild:
    """A compiled and ready-to-run synth program."""
    program: SynthProgram
    dsp_path: Path
    binary_path: Path
    json_path: Path

    @property
    def jack_client_name(self) -> str:
        return self.program.name

    @property
    def param_ranges(self) -> dict[str, tuple[float, float]]:
        return self.program.param_ranges


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _midpoint_defaults(program: SynthProgram) -> dict[str, float]:
    """Compile-time defaults for slider placeholders. Overridden via OSC at runtime."""
    return {name: (lo + hi) / 2 for name, (lo, hi) in program.param_ranges.items()}


def prepare(program: SynthProgram | str, force: bool = False) -> SynthBuild:
    """Compile + emit JSON if needed. Returns a SynthBuild handle.

    Idempotent: cached by template content hash. Pass force=True to recompile.
    """
    if isinstance(program, str):
        program = get_program(program)

    instantiated = program.instantiate(_midpoint_defaults(program))
    key = _content_hash(instantiated)
    out_dir = _BUILD_DIR / f"{program.name}_{key}"
    dsp_path = out_dir / f"{program.name}.dsp"
    binary_path = out_dir / program.name
    json_path = out_dir / f"{program.name}.dsp.json"

    if not force and binary_path.is_file() and json_path.is_file():
        return SynthBuild(program, dsp_path, binary_path, json_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".build.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if not force and binary_path.is_file() and json_path.is_file():
            return SynthBuild(program, dsp_path, binary_path, json_path)

        dsp_path.write_text(instantiated)

        # Compile sndfile binary (no JACK required). faust2sndfile drops it in cwd.
        # faust2jaqt is only needed for the interactive GUI — skip it on headless servers.
        env = os.environ.copy()
        env["CXXFLAGS"] = f"{env.get('CXXFLAGS', '')} -pthread".strip()
        res = subprocess.run(
            ["faust2sndfile", dsp_path.name],
            cwd=out_dir, capture_output=True, text=True, env=env,
        )
        if res.returncode != 0 or not binary_path.is_file():
            raise RuntimeError(
                f"faust2sndfile failed for {program.name}:\n"
                f"stdout: {res.stdout}\nstderr: {res.stderr}"
            )

        # Emit parameter JSON next to the dsp file.
        res = subprocess.run(
            ["faust", "-json", dsp_path.name],
            cwd=out_dir, capture_output=True, text=True,
        )
        if res.returncode != 0 or not json_path.is_file():
            raise RuntimeError(
                f"faust -json failed for {program.name}:\n"
                f"stdout: {res.stdout}\nstderr: {res.stderr}"
            )

    return SynthBuild(program, dsp_path, binary_path, json_path)


def launch(build: SynthBuild, settle: float = 0.5) -> subprocess.Popen:
    """Start the synth as a JACK client. Caller is responsible for terminating."""
    proc = subprocess.Popen(
        [str(build.binary_path), "--nogui"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(settle)
    if proc.poll() is not None:
        raise RuntimeError(
            f"{build.program.name} exited immediately with code {proc.returncode}"
        )
    return proc


@contextmanager
def running_synth(program: SynthProgram | str, settle: float = 0.5):
    """prepare() + launch() + guaranteed cleanup."""
    build = prepare(program)
    proc = launch(build, settle=settle)
    try:
        yield build
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
