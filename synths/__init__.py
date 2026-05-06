"""
Synth configurations registry.
"""

from .config import SynthConfig

# Define available synths
BANDPASS_NOISE = SynthConfig(
    name="Bandpass Noise",
    synth_json="synths/bandpass_noise.dsp.json",
    dsp_path="synths/bandpass_noise.dsp",
    target_wav="targets/bp_100-901.wav",
    jack_port="bandpass_noise:out_0",
    sweep_param="lp_freq",
    fixed_params={"hp_freq": 400},
)

SINE = SynthConfig(
    name="Sine",
    synth_json="synths/sine.dsp.json",
    dsp_path="synths/sine.dsp",
    target_wav="targets/50hz_sine.wav",
    jack_port="sine:out_0",
    sweep_param="freq",
    fixed_params={},
)

# Registry: name -> config
SYNTHS = {
    "bandpass_noise": BANDPASS_NOISE,
    "sine": SINE,
}


def get_synth(name: str) -> SynthConfig:
    """Get synth config by name."""
    if name not in SYNTHS:
        raise ValueError(f"Unknown synth: {name}. Available: {list(SYNTHS.keys())}")
    return SYNTHS[name]


def list_synths() -> list[str]:
    """List available synth names."""
    return list(SYNTHS.keys())
