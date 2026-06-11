# Stats summary (auto-generated)

All P-loss = Euclidean distance in normalized param space (lower=better), budget 200, matched seeds. Each synth uses its canonical loss (see SYNTH_LOSS_CANONICAL).

## Final P-loss (median) — last evaluation of each trial

| synth | GD | RandomSearch | CMA-ES | LES |
|---|---|---|---|---|
| bandpass_noise_v1 | 0.014 | 0.480 | 0.001 | 0.129 |
| am_noise | 0.495 | 0.480 | 0.019 | nan |
| add_sinesaw | 0.706 | 0.480 | 0.341 | 0.420 |
| sine_mod_saw | 0.700 | 0.480 | 0.341 | 0.523 |
| chirplet | 0.450 | 0.480 | 0.529 | 0.440 |

## Visited P-loss (median) — oracle best (optimistic)

| synth | GD | RandomSearch | CMA-ES | LES |
|---|---|---|---|---|
| bandpass_noise_v1 | 0.009 | 0.033 | 0.000 | 0.025 |
| am_noise | 0.377 | 0.033 | 0.004 | nan |
| add_sinesaw | 0.387 | 0.033 | 0.041 | 0.045 |
| sine_mod_saw | 0.370 | 0.033 | 0.049 | 0.047 |
| chirplet | 0.245 | 0.033 | 0.058 | 0.044 |

## Final vs visited gap (final − visited; higher = final worse than best seen)

| synth | GD | RandomSearch | CMA-ES | LES |
|---|---|---|---|---|
| bandpass_noise_v1 | 0.005 | 0.447 | 0.001 | 0.104 |
| am_noise | 0.118 | 0.447 | 0.015 | nan |
| add_sinesaw | 0.320 | 0.447 | 0.300 | 0.376 |
| sine_mod_saw | 0.329 | 0.447 | 0.292 | 0.475 |
| chirplet | 0.205 | 0.447 | 0.471 | 0.396 |

## Controlled wall-clock (E3): ms/eval | reach% (P≤0.05) | sec→thr

| synth | GD | RandomSearch | CMA-ES | LES |
|---|---|---|---|---|
| bandpass_noise_v1 | 2212 / 56% / 28.8s | 22 / 52% / 1.5s | 25 / 96% / 1.0s | — |
| am_noise | 55 / 14% / 0.8s | 283 / 52% / 19.8s | 286 / 80% / 9.4s | — |
| add_sinesaw | 24 / 6% / 0.3s | 19 / 52% / 1.3s | 19 / 56% / 0.7s | — |

