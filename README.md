# Realtime Faust Synth Agent — Setup & Commands

## Prerequisites

```bash
# Ubuntu/Debian
sudo apt install faust jackd2 jack-tools qjackctl

# macOS (Homebrew)
brew install faust jack
```

---

## Step 1 — Python environment

```bash
cd synth_agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Step 2 — Start JACK

JACK must be running before anything else.

```bash
# Option A: GUI
qjackctl &

# Option B: headless (48kHz, 512-sample blocks, ALSA hw:0)
jackd -d alsa -d hw:0 -r 44100 -p 512 -n 2 &

# macOS
jackd -d coreaudio -r 44100 -p 512 &
```

---

## Step 3 — Compile the Faust synths

```bash
# Compile the synth-under-control
faust2jack -osc synths/bandpass_noise.dsp
# → produces ./bandpass_noise binary

# Compile the target synth (separate binary)
faust2jack -osc synths/target.dsp
# → produces ./target binary

# Generate JSON metadata (needed by the Python agent)
faust -json synths/bandpass_noise.dsp
# → produces ./synths/bandpass_noise.json
```

If `faust2jack` is not on your path, try:
```bash
# Find it
find /usr -name "faust2jack" 2>/dev/null
# Or compile via faust2jackinternal for embedded OSC
faust2jaqt -osc synths/bandpass_noise.dsp   # Qt GUI variant
```

---

## Step 4 — Run the synths

Open two terminal tabs.

**Tab 1 — synth to be controlled (starts far from target):**
```bash
./bandpass_noise
# JACK ports: bandpass_noise:output_0, bandpass_noise:output_1
# OSC listen: localhost:5510
```

**Tab 2 — target synth (hp=200, lp=1000):**
```bash
./target
# JACK ports: target:output_0, target:output_1
# OSC listen: localhost:5511  (different port — change in target.dsp if needed)
```

Verify JACK connections:
```bash
jack_lsp                      # list all ports
jack_connect bandpass_noise:output_0 system:playback_1   # optional: hear it
```

---

## Step 5 — Run the agent

**Option A — live target (two running synths):**
```bash
python -m agent.main \
    --synth-json   synths/bandpass_noise.json \
    --jack-port    "bandpass_noise:output_0" \
    --target-jack-port "target:output_0" \
    --record-target-blocks 64 \
    --optimizer    hill \
    --settle-time  0.08 \
    --eval-blocks  8 \
    --max-iters    2000
```

**Option B — pre-recorded WAV target:**
```bash
# Record the target sound first (5 seconds)
faust2sndfile -d 5 synths/target.dsp target.wav

# Run agent against the recording
python -m agent.main \
    --synth-json synths/bandpass_noise.json \
    --jack-port  "bandpass_noise:output_0" \
    --target-wav target.wav
```

**Option C — CMA-ES optimizer:**
```bash
python -m agent.main \
    --synth-json   synths/bandpass_noise.json \
    --jack-port    "bandpass_noise:output_0" \
    --target-jack-port "target:output_0" \
    --optimizer    cma
```

---

## What you should see

```
[FaustParams] Loaded 3 parameter(s) from synths/bandpass_noise.json
  hp_freq    /bandpass_noise/hp_freq    [20, 8000]  default=20.0
  lp_freq    /bandpass_noise/lp_freq    [20, 8000]  default=5000.0
  gain       /bandpass_noise/gain       [0, 1]      default=0.5

[OSCController] Targeting Faust at 127.0.0.1:5510
[Optimizer] Starting loss: 0.84231
[    0] ✓ loss=0.81004  σ=0.1650  hp_freq=47.2  lp_freq=4203.1  gain=0.49
[    1] ✗ loss=0.81004  σ=0.1518  hp_freq=47.2  lp_freq=4203.1  gain=0.49
[    2] ✓ loss=0.76443  σ=0.1670  hp_freq=98.1  lp_freq=3100.2  gain=0.51
...converging toward hp=200, lp=1000...
```

---

## Swapping synths

To control a completely different Faust synth:

```bash
# 1. Write your new synth DSP
# 2. Compile
faust2jack -osc synths/my_new_synth.dsp
faust -json synths/my_new_synth.dsp

# 3. Run the new binary
./my_new_synth

# 4. Run agent — only the --synth-json and --jack-port change
python -m agent.main \
    --synth-json synths/my_new_synth.json \
    --jack-port  "my_new_synth:output_0" \
    --target-wav target.wav
```

The agent code is unchanged.  It discovers the new parameter names/ranges
automatically from the JSON file.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot connect to server socket` | JACK not running — do Step 2 |
| `error: 'PRESETDIR' was not declared` | Add `-DPRESETDIR=\"auto\"` to CXXFLAGS in Makefile |
| OSC not reaching Faust | Confirm Faust binary started with `-osc` flag; check port with `faust -osc -port 5510` |
| `queue.Empty` / timeout | Increase `--settle-time`; ensure JACK ports are connected |
| Low-latency audio clicks | Increase JACK buffer: `jackd ... -p 1024` |
