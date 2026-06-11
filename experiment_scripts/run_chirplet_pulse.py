"""Rerun chirplet_pulse × 4 losses × GD, 300 trials, with corrected * 526 DSP."""
import subprocess, os

env_base = os.environ.copy()
env_base.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "GOMP_SPINCOUNT": "0",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
})

PY = "/cshome/asalimi/.conda/envs/soundmatch/bin/python"
REPO = "/cshome/asalimi/code/realtime_soundmatching"
LOSSES = ["SIMSE_Spec", "L1_Spec", "JTFS", "DTW_Envelope"]

procs = []
for loss in LOSSES:
    logpath = f"{REPO}/paper_experiments/results/chirplet_pulse_{loss}_GD.log"
    cmd = [PY, f"{REPO}/paper_experiments/run_paper.py",
           "--synth", "chirplet_pulse", "--loss", loss,
           "--method", "GD", "--trials", "300", "--budget", "200"]
    with open(logpath, "w") as lf:
        p = subprocess.Popen(cmd, env=env_base, stdout=lf, stderr=lf, cwd=REPO)
    procs.append((loss, p))
    print(f"PID {p.pid} → chirplet_pulse / {loss}", flush=True)

print(f"Waiting for {len(procs)} jobs...", flush=True)
for loss, p in procs:
    p.wait()
    print(f"Done: chirplet_pulse / {loss} (exit {p.returncode})", flush=True)
print("All complete.", flush=True)
