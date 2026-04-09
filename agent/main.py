"""
agent/main.py

Entry point for the realtime synth agent.

Usage
-----
python -m agent.main --synth-json synths/bandpass_noise.json \
                     --jack-port "bandpass_noise:output_0" \
                     --target-wav target.wav
"""

import argparse
import signal
import sys
import time

import numpy as np

from .capture import JackCapture
from .params import FaustParams
from .controller import OSCController
from .features import load_target_from_wav, load_target_from_capture
from .optimizer import BoilerplateOptimizer


def parse_args():
    p = argparse.ArgumentParser(description="Realtime Faust synth agent")
    p.add_argument("--synth-json",   required=True,
                   help="Path to faust -json output for the synth being controlled")
    p.add_argument("--jack-port",    required=True,
                   help="JACK port of the synth, e.g. 'bandpass_noise:output_0'")
    p.add_argument("--osc-host",     default="127.0.0.1")
    p.add_argument("--osc-port",     default=5510, type=int)
    p.add_argument("--sample-rate",  default=44100, type=int)

    # Target: WAV file OR a live JACK port
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-wav",        help="Pre-recorded target WAV")
    target.add_argument("--target-jack-port",  help="JACK port of target synth")

    p.add_argument("--record-target-blocks", default=64, type=int,
                   help="How many JACK blocks to record for live target capture")
    p.add_argument("--optimizer", choices=["boilerplate"], default="boilerplate")
    p.add_argument("--settle-time",   default=0.08,  type=float)
    p.add_argument("--eval-duration", default=None,  type=float,
                   help="Seconds of audio to capture per evaluation (e.g. 0.5, 1.0). "
                        "Overrides --eval-blocks when provided.")
    p.add_argument("--eval-blocks",   default=8,     type=int,
                   help="JACK blocks to capture per evaluation (used if --eval-duration is not set)")
    p.add_argument("--max-iters",    default=5_000, type=int)
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load parameter space from Faust JSON
    # ------------------------------------------------------------------
    params = FaustParams(args.synth_json)

    # ------------------------------------------------------------------
    # 2. Start JACK capture of the *synth* being controlled
    # ------------------------------------------------------------------
    synth_capture = JackCapture(client_name="agent_synth_capture")
    synth_capture.start(args.jack_port)

    # ------------------------------------------------------------------
    # 3. Get target features
    # ------------------------------------------------------------------
    if args.target_wav:
        print(f"[main] Loading target from WAV: {args.target_wav}")
        target_features = load_target_from_wav(args.target_wav, args.sample_rate)
    else:
        print(f"[main] Recording target from JACK port: {args.target_jack_port}")
        target_capture = JackCapture(client_name="agent_target_capture")
        target_capture.start(args.target_jack_port)
        time.sleep(0.5)   # let buffer fill
        target_features = load_target_from_capture(
            target_capture,
            n_blocks=args.record_target_blocks,
            sample_rate=args.sample_rate,
        )
        target_capture.stop()

    # ------------------------------------------------------------------
    # 4. Resolve eval_blocks from duration if requested
    # ------------------------------------------------------------------
    if args.eval_duration is not None:
        eval_blocks = max(1, round(args.eval_duration * synth_capture.sample_rate / synth_capture.blocksize))
        print(f"[main] eval_duration={args.eval_duration}s → {eval_blocks} blocks "
              f"(blocksize={synth_capture.blocksize}, sr={synth_capture.sample_rate})")
    else:
        eval_blocks = args.eval_blocks

    # ------------------------------------------------------------------
    # 5. OSC controller
    # ------------------------------------------------------------------
    controller = OSCController(params, host=args.osc_host, port=args.osc_port)

    # ------------------------------------------------------------------
    # 6. Build optimizer
    # ------------------------------------------------------------------
    opt_kwargs = dict(
        params=params,
        capture=synth_capture,
        controller=controller,
        target_features=target_features,
        sample_rate=args.sample_rate,
        settle_time=args.settle_time,
        eval_blocks=eval_blocks,
    )

    optimizer = BoilerplateOptimizer(**opt_kwargs)

    # ------------------------------------------------------------------
    # 7. Graceful shutdown on Ctrl-C
    # ------------------------------------------------------------------
    def _signal_handler(sig, frame):
        print("\n[main] Stopping…")
        optimizer.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ------------------------------------------------------------------
    # 8. Run
    # ------------------------------------------------------------------
    try:
        optimizer.run(max_iterations=args.max_iters)
    finally:
        synth_capture.stop()
        print("[main] JACK client closed.")


if __name__ == "__main__":
    main()
