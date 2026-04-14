import numpy as np
import soundfile as sf


def save_audio(path: str, audio: np.ndarray, samplerate: int) -> None:
    """Save audio array to a file (format inferred from extension)."""
    sf.write(path, audio, samplerate)


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio from a file. Returns (audio_array, samplerate)."""
    audio, samplerate = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio,axis=1) 
    return audio, samplerate
