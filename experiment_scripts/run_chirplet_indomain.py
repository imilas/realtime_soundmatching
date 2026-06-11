"""Launch chirplet × 4 losses + chirplet_pulse × 4 losses × GD, 300 trials each."""
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
SYNTHS = ["chirplet", "chirplet_pulse"]
LOSSES = ["SIMSE_Spec", "L1_Spec", "JTFS", "DTW_Envelope"]
TRIALS = 300

procs = []
for synth in SYNTHS:
    for loss in LOSSES:
        logpath = f"{REPO}/paper_experiments/results/{synth}_{loss}_GD.log"
        cmd = [PY, f"{REPO}/paper_experiments/run_paper.py",
               "--synth", synth, "--loss", loss,
               "--method", "GD", "--trials", str(TRIALS), "--budget", "200"]
        with open(logpath, "a") as lf:
            p = subprocess.Popen(cmd, env=env_base, stdout=lf, stderr=lf, cwd=REPO)
        procs.append((synth, loss, p))
        print(f"PID {p.pid} → {synth} / {loss}", flush=True)

print(f"Waiting for {len(procs)} jobs...", flush=True)
for synth, loss, p in procs:
    p.wait()
    print(f"Done: {synth} / {loss} (exit {p.returncode})", flush=True)
print("All complete.", flush=True)
