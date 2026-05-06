"""
Synth configuration definitions.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SynthConfig:
    """Configuration for a synthesizer."""

    name: str
    """Display name (e.g. "Bandpass Noise")"""

    synth_json: str
    """Path to synth.dsp.json file"""

    dsp_path: str
    """Path to synth.dsp file"""

    target_wav: str
    """Path to target audio file"""

    jack_port: str
    """JACK port to connect to (e.g. "bandpass_noise:out_0")"""

    sweep_param: str
    """Parameter to optimize (e.g. "lp_freq")"""

    fixed_params: dict[str, float] = field(default_factory=dict)
    """Parameters to hold constant during sweep"""

    osc_host: str = "127.0.0.1"
    """OSC host"""

    osc_port: int = 5510
    """OSC port"""

    sample_rate: int = 44100
    """Sample rate in Hz"""

    eval_blocks: int = 32
    """JACK blocks to capture per evaluation"""

    block_size: int = 1024
    """JACK block size in samples"""

    settle_time: float = 0.08
    """Time to wait after parameter change before capturing (seconds)"""

    landscape_steps: int = 80
    """Number of points to compute in loss landscape"""

    def resolve_paths(self, repo_root: Path) -> "SynthConfig":
        """Resolve relative paths against repo root."""
        return SynthConfig(
            name=self.name,
            synth_json=str(repo_root / self.synth_json),
            dsp_path=str(repo_root / self.dsp_path),
            target_wav=str(repo_root / self.target_wav),
            jack_port=self.jack_port,
            sweep_param=self.sweep_param,
            fixed_params=self.fixed_params,
            osc_host=self.osc_host,
            osc_port=self.osc_port,
            sample_rate=self.sample_rate,
            eval_blocks=self.eval_blocks,
            block_size=self.block_size,
            settle_time=self.settle_time,
            landscape_steps=self.landscape_steps,
        )
