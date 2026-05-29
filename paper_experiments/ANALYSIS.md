# Results analysis — 2026-05-29

Analysis of `paper_experiments/results/*.pkl`. All P-loss figures are Euclidean
distance in normalized [0,1]^d parameter space (lower = better). Budget = 200
evals/trial (note: `config.EVAL_BUDGET = 400` is **unused** by the runs that
produced these pkls — see "Caveats").

## TL;DR

- **CMA-ES is the best overall method** (best final accuracy and sample
  efficiency on `bandpass_noise` and `am_noise`); **BO is best on the
  multimodal `add_sinesaw`**.
- **BO's huge jumps are real but not a bug-in-outcome.** It is global GP
  exploration: the best point is found *late* (47–74% of trials after eval 100,
  only 2–4% in the initial random phase). It works, but never switches to a
  local exploitation/refinement phase — hence the random-search *look*.
- **`add_sinesaw` is a loss/identifiability problem, not a method problem.**
  At the point of lowest *audio* loss, median *P*-loss is still 0.20–0.37, and
  corr(audio_loss, P_loss) ≈ 0.04–0.34. The optimizers DO visit near-perfect
  points (best P-loss 0.02–0.04) but the loss doesn't reward them. No optimizer
  change fixes this.
- **QL does not learn across trials.** With a persistent Q-table over 5000
  trials, last-20% vs first-20% median P-loss is flat (Δ ≈ +0.001 on bandpass,
  −0.038 on am_noise). The MDP formulation can't localize a target that changes
  every trial and isn't encoded in the state.
- **RandomSearch ignores the objective.** Its best-so-far curve is *identical*
  across all three synths (0.260 / 0.168 / 0.082 / 0.049 at evals 25/50/100/200)
  because it is a seed-determined random *walk*, independent of audio loss.

## Final accuracy — best_p_loss (median across trials)

| synth | GD | HillClimber | RandomSearch | CMA-ES | BO | QL |
|---|---|---|---|---|---|---|
| bandpass_noise | 0.029 | 0.012 | 0.098 | **0.001** | 0.041 | 0.258 |
| am_noise | 0.402 | 0.038 | 0.085 | **0.008** | 0.014 | 0.309 |
| add_sinesaw | 0.360 | 0.373 | 0.324 | 0.336 | **0.197** | 0.376 |

n: GD = 10 (bandpass) / 60 (others); HC/RS/CMA/BO = 200; QL = 5000 / 444 / 2375.

## Sample efficiency — best-so-far P-loss (median) at eval budget

| synth | method | @25 | @50 | @100 | @200 |
|---|---|---|---|---|---|
| bandpass_noise | CMA-ES | 0.083 | 0.041 | 0.012 | **0.000** |
| bandpass_noise | BO | 0.076 | 0.050 | 0.035 | 0.025 |
| bandpass_noise | HillClimber | 0.171 | 0.046 | 0.011 | 0.007 |
| am_noise | CMA-ES | 0.082 | 0.046 | 0.017 | **0.004** |
| am_noise | BO | 0.084 | 0.054 | 0.021 | 0.007 |
| am_noise | GD | 0.429 | 0.417 | 0.406 | 0.402 |
| add_sinesaw | BO | 0.087 | 0.059 | 0.032 | **0.021** |
| add_sinesaw | CMA-ES | 0.096 | 0.063 | 0.044 | 0.038 |

## Step-size analysis (normalized; max possible in 2D box = √2 ≈ 1.414)

Mean / median per-step distance ||x_t − x_{t-1}||:

| method | mean | median | p90 | interpretation |
|---|---|---|---|---|
| GD | 0.015–0.026 | small | — | smooth, tiny steps; barely moves on am/add |
| QL | 0.035–0.058 | ~0.00 | 0.11 | mostly "stay"/boundary actions; coarse 0.1 grid |
| HillClimber | 0.082 | 0.076 | 0.143 | Gaussian σ=0.05 + greedy accept |
| RandomSearch | 0.112 | 0.104 | 0.199 | random walk σ=0.1, no acceptance |
| CMA-ES | 0.13–0.15 | 0.05–0.08 | 0.37–0.42 | adaptive: small steps + occasional big |
| **BO** | **0.34–0.54** | **0.5** | **0.8–0.95** | **corner-to-corner; pure global EI exploration** |

## Per-method assessment & best-practice issues

### CMA-ES — best overall, mostly sound
- Operates in normalized [0,1]^2, sigma0=0.3, library defaults otherwise.
- Failure rate climbs on harder problems: %trials with best_p_loss>0.2 is
  4% (bandpass) → 20% (am_noise) → 67% (add_sinesaw, but that's the loss).
- **Improvement:** restarts (IPOP/BIPOP-CMA-ES) to escape local optima on
  am_noise; the `cma` library supports `restarts`.

### BO — strong best-found, no exploitation phase
- `skopt.Optimizer(base_estimator="GP", acq_func="EI", n_initial_points=10)`.
- skopt's default GP **estimates Gaussian observation noise**; on a
  near-deterministic objective this inflates posterior uncertainty and keeps EI
  exploring globally forever (explains the √2 jumps).
- **Improvement (test first):** set GP `noise` ≈ 1e-10 (near-deterministic);
  optionally lower EI `xi` to exploit; consider a short local-refinement phase
  on the incumbent at the end. Also: GP at 200 points is O(n³) slow.

### HillClimber — good median, gets stuck
- Fixed σ=0.05, greedy accept, no restart. mean ≫ median (0.12 vs 0.012 on
  bandpass); %stuck (>0.2) = 15% / 44% / 71%.
- **Improvement:** random restarts on stagnation; adaptive step size (1/5
  success rule) — both standard and cheap.

### RandomSearch — not a standard baseline
- It is a Gaussian random *walk* (σ=0.1) that resets to the last point each
  step and never uses the loss. Hence synth-independent best-so-far.
- **Improvement:** replace with canonical **uniform** random sampling over the
  box (the standard "random search" floor). A σ=0.1 walk actually explores
  *less* than uniform sampling, so it isn't even a valid lower bound.

### GD — weak on am_noise / add_sinesaw (low n, treat as provisional)
- RMSProp, lr=0.045, grad-clip 1.0, 200 steps, audio-loss gradients (SIMSE_Spec
  / DTW_Envelope / JTFS). Step sizes tiny (~0.02).
- On am_noise it barely moves (0.429→0.402 over 200 steps) — weak/poor gradient
  signal from DTW_Envelope, or lr too low.
- **Caveat:** GD n is only 10 (bandpass) / 60 (others); the 300-trial run
  (`run_gd_300x200.sh`) errored and never completed. Conclusions provisional
  until GD has matched n.
- **Improvement:** multi-start GD; per-loss lr tuning; check gradient norms are
  non-trivial for DTW/JTFS before/after clipping.

### QL — does not learn; MDP mismatch
- State = (position bins, loss bucket); actions = 3^d grid steps; reward = Δloss.
- The target changes every trial and is **not** in the state, so no stationary
  policy can point toward it. The agent can only learn a weak "keep moving if
  improving" heuristic. Last-20% vs first-20% performance is flat over 5000
  trials → the persistent Q-table adds nothing.
- **Improvement (research-level):** encode target-relative information in the
  state (e.g. recent loss gradient / direction), or reframe as contextual
  bandit / learned policy with the target spectrum as context. As a plain
  tabular grid-walker it cannot win.

## Caveats for the write-up (fairness)

1. **Unequal n.** CMA-ES vs BO is a clean comparison (both n=200, matched
   seeds). GD (n=10–60) and QL (trained, n=thousands) are **not** matched —
   restrict to common seeds (trial index = seed) before ranking them.
2. **Budget.** These pkls are 200 evals; `config.EVAL_BUDGET = 400`. State which
   the paper uses and regenerate consistently.
3. **QL is "trained," others are not.** QL carries a Q-table across trials; the
   gradient-free methods start cold each trial. Different protocol — report it
   explicitly.

## Verification commands

All numbers above were produced from the pkls with the `soundmatch` conda env.
See `paper_experiments/analyze_results.py` (added alongside this report) to
regenerate them.
