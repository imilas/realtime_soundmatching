# Stats summary (auto-generated)

All P-loss = Euclidean distance in normalized param space (lower=better), budget 200, matched seeds. Each synth uses its canonical loss (see SYNTH_LOSS_CANONICAL).

## Final P-loss (median) — last evaluation of each trial

| synth | GD | RandomSearch | CMA-ES | LES |
|---|---|---|---|---|
| bandpass_noise_v1 | 0.013 | 0.507 | 0.000 | 0.011 |
| am_noise | 0.495 | 0.507 | 0.000 | nan |
| add_sinesaw | 0.716 | 0.507 | 0.350 | 0.377 |
| sine_mod_saw | 0.707 | 0.507 | 0.024 | 0.159 |
| chirplet | 0.472 | 0.507 | 0.535 | 0.499 |
| dx7_alg1 | 1.504 | 1.193 | 1.110 | 1.194 |
| dx7_alg2 | 1.618 | 1.193 | 1.177 | 1.201 |
| dx7_alg3 | 1.592 | 1.193 | 1.105 | 1.194 |

## Visited P-loss (median) — oracle best (optimistic)

| synth | GD | RandomSearch | CMA-ES | LES |
|---|---|---|---|---|
| bandpass_noise_v1 | 0.008 | 0.021 | 0.000 | 0.001 |
| am_noise | 0.371 | 0.021 | 0.000 | nan |
| add_sinesaw | 0.383 | 0.021 | 0.041 | 0.025 |
| sine_mod_saw | 0.364 | 0.021 | 0.001 | 0.011 |
| chirplet | 0.193 | 0.021 | 0.057 | 0.040 |
| dx7_alg1 | 1.111 | 0.492 | 0.568 | 0.640 |
| dx7_alg2 | 1.163 | 0.492 | 0.604 | 0.625 |
| dx7_alg3 | 1.127 | 0.492 | 0.583 | 0.618 |

## Final vs visited gap (final − visited; higher = final worse than best seen)

| synth | GD | RandomSearch | CMA-ES | LES |
|---|---|---|---|---|
| bandpass_noise_v1 | 0.005 | 0.486 | 0.000 | 0.010 |
| am_noise | 0.124 | 0.486 | 0.000 | nan |
| add_sinesaw | 0.333 | 0.486 | 0.309 | 0.352 |
| sine_mod_saw | 0.343 | 0.486 | 0.023 | 0.148 |
| chirplet | 0.279 | 0.486 | 0.479 | 0.459 |
| dx7_alg1 | 0.393 | 0.701 | 0.542 | 0.554 |
| dx7_alg2 | 0.455 | 0.701 | 0.574 | 0.576 |
| dx7_alg3 | 0.465 | 0.701 | 0.522 | 0.576 |

## Controlled wall-clock (E3): ms/eval | reach% (P≤0.05) | sec→thr

| synth | GD | RandomSearch | CMA-ES | LES |
|---|---|---|---|---|
| bandpass_noise_v1 | 2212 / 56% / 28.8s | 22 / 52% / 1.5s | 25 / 96% / 1.0s | — |
| am_noise | 55 / 14% / 0.8s | 283 / 52% / 19.8s | 286 / 80% / 9.4s | — |
| add_sinesaw | 24 / 6% / 0.3s | 19 / 52% / 1.3s | 19 / 56% / 0.7s | — |

