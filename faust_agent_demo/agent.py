#!/usr/bin/env python3
import argparse
import queue
import signal
import sys
import time

import jack
import numpy as np
from pythonosc.udp_client import SimpleUDPClient


def spectral_centroid_hz(x, sr):
    x = np.asarray(x, dtype=np.float32)
    if np.allclose(x, 0):
        return 0.0
    win = np.hanning(len(x)).astype(np.float32)
    spec = np.fft.rfft(x * win)
    mag = np.abs(spec) + 1e-12
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    return float((freqs * mag).sum() / mag.sum())


class LiveAgent:
    def __init__(self, source_client: str, osc_port: int, target_centroid: float):
        self.source_client = source_client
        self.osc_port = osc_port
        self.target_centroid = float(target_centroid)

        self.client = jack.Client("python_agent")
        self.in_l = self.client.inports.register("in_1")
        self.in_r = self.client.inports.register("in_2")

        self.sample_rate = self.client.samplerate
        self.block_size = self.client.blocksize
        self.q = queue.Queue(maxsize=512)
        self.running = True

        # Control state
        self.bandwidth = 1200.0
        self.center = 1000.0

        self.osc = SimpleUDPClient("127.0.0.1", self.osc_port)
        self.osc_root = f"/{self.source_client}"

        @self.client.set_process_callback
        def process(frames):
            # get_array() is only used inside the process callback
            left = self.in_l.get_array().copy()
            right = self.in_r.get_array().copy()
            mono = 0.5 * (left + right)
            try:
                self.q.put_nowait(mono)
            except queue.Full:
                pass

        @self.client.set_shutdown_callback
        def shutdown(status, reason):
            print(f"JACK shutdown: {reason} ({status})", file=sys.stderr)
            self.running = False

    def find_source_ports(self):
        ports = list(self.client.get_ports(is_audio=True, is_output=True))
        ports = [p for p in ports if p.name.startswith(f"{self.source_client}:")]
        ports.sort(key=lambda p: p.name)
        return ports

    def find_playback_ports(self):
        ports = list(self.client.get_ports(is_audio=True, is_input=True, is_physical=True))
        ports.sort(key=lambda p: p.name)
        return ports

    def safe_connect(self, src, dst):
        try:
            self.client.connect(src, dst)
            print(f"connected {src} -> {dst}")
        except jack.JackError:
            pass

    def connect_graph(self):
        src_ports = self.find_source_ports()
        if not src_ports:
            raise RuntimeError(
                f"Could not find JACK output ports for source client '{self.source_client}'. "
                f"Start ./bp_noise first."
            )

        playback_ports = self.find_playback_ports()
        if not playback_ports:
            raise RuntimeError("Could not find physical JACK playback ports.")

        # Connect Faust -> speakers
        if len(src_ports) == 1 and len(playback_ports) >= 2:
            self.safe_connect(src_ports[0].name, playback_ports[0].name)
            self.safe_connect(src_ports[0].name, playback_ports[1].name)
        else:
            for src, dst in zip(src_ports, playback_ports):
                self.safe_connect(src.name, dst.name)

        # Connect Faust -> Python agent
        if len(src_ports) == 1:
            self.safe_connect(src_ports[0].name, self.in_l.name)
            self.safe_connect(src_ports[0].name, self.in_r.name)
        else:
            self.safe_connect(src_ports[0].name, self.in_l.name)
            if len(src_ports) > 1:
                self.safe_connect(src_ports[1].name, self.in_r.name)
            else:
                self.safe_connect(src_ports[0].name, self.in_r.name)

    def set_params(self, hp_cut, lp_cut):
        hp_cut = float(np.clip(hp_cut, 20.0, 3000.0))
        lp_cut = float(np.clip(lp_cut, hp_cut + 50.0, 8000.0))
        self.osc.send_message(f"{self.osc_root}/hp_cut", hp_cut)
        self.osc.send_message(f"{self.osc_root}/lp_cut", lp_cut)

    def run(self):
        self.client.activate()
        self.connect_graph()

        # Seed the synth
        self.set_params(self.center - self.bandwidth / 2, self.center + self.bandwidth / 2)

        analysis_horizon_s = 0.25
        target_samples = int(self.sample_rate * analysis_horizon_s)
        buf = np.zeros(0, dtype=np.float32)

        print(f"sample_rate={self.sample_rate}, block_size={self.block_size}")
        print(f"listening to '{self.source_client}', sending OSC to 127.0.0.1:{self.osc_port}")
        print(f"target centroid = {self.target_centroid:.1f} Hz")
        print("Ctrl-C to stop.")

        while self.running:
            try:
                block = self.q.get(timeout=1.0)
            except queue.Empty:
                continue

            buf = np.concatenate([buf, block])
            if len(buf) < target_samples:
                continue

            x = buf[-target_samples:]
            centroid = spectral_centroid_hz(x, self.sample_rate)

            # very simple proportional controller
            err = self.target_centroid - centroid
            self.center += 0.15 * err
            self.center = float(np.clip(self.center, 150.0, 5000.0))

            hp = self.center - self.bandwidth / 2
            lp = self.center + self.bandwidth / 2
            self.set_params(hp, lp)

            print(
                f"centroid={centroid:8.1f} Hz | "
                f"target={self.target_centroid:8.1f} Hz | "
                f"hp={max(20.0, hp):7.1f} | lp={min(8000.0, lp):7.1f}"
            )

            # keep only recent audio
            buf = buf[-target_samples:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-client", default="bp_noise",
                        help="JACK client name of the Faust synth")
    parser.add_argument("--osc-port", type=int, default=5510,
                        help="OSC port the Faust app listens on")
    parser.add_argument("--target-centroid", type=float, default=1200.0,
                        help="Target spectral centroid in Hz")
    args = parser.parse_args()

    agent = LiveAgent(
        source_client=args.source_client,
        osc_port=args.osc_port,
        target_centroid=args.target_centroid,
    )

    def stop_handler(signum, frame):
        agent.running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    try:
        agent.run()
    finally:
        try:
            agent.client.deactivate()
        except Exception:
            pass
        try:
            agent.client.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
