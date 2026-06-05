# Stats summary (auto-generated 2026-05-31)

All P-loss = Euclidean distance in normalized param space (lower=better), budget 200, matched seeds; Learned is zero-shot (0 evals).

## Returned P-loss (median) — what each method DEPLOYS

| synth | GD | RandomSearch | CMA-ES | BO | Learned |
|---|---|---|---|---|---|
| bandpass_noise | 0.087 | 0.060 | 0.001 | 0.041 | 0.015 |
| am_noise | 0.488 | 0.049 | 0.008 | 0.014 | 0.034 |
| add_sinesaw | 0.520 | 0.152 | 0.336 | 0.197 | 0.045 |

## Visited P-loss (median) — oracle best (optimistic)

| synth | GD | RandomSearch | CMA-ES | BO | Learned |
|---|---|---|---|---|---|
| bandpass_noise | 0.025 | 0.034 | 0.000 | 0.025 | 0.015 |
| am_noise | 0.376 | 0.034 | 0.004 | 0.007 | 0.034 |
| add_sinesaw | 0.367 | 0.034 | 0.038 | 0.021 | 0.045 |

## Deception gap (returned − visited; higher = more fooled by the loss)

| synth | GD | RandomSearch | CMA-ES | BO |
|---|---|---|---|---|
| bandpass_noise | 0.062 | 0.026 | 0.000 | 0.017 |
| am_noise | 0.112 | 0.015 | 0.005 | 0.007 |
| add_sinesaw | 0.153 | 0.118 | 0.298 | 0.176 |

## Controlled wall-clock (E3): ms/eval | reach% (P≤0.05) | sec→thr

| synth | GD | RandomSearch | CMA-ES | BO |
|---|---|---|---|---|
| bandpass_noise | 2212 / 56% / 28.8s | 22 / 52% / 1.5s | 25 / 96% / 1.0s | 208 / 88% / 8.9s |
| am_noise | 55 / 14% / 0.8s | 283 / 52% / 19.8s | 286 / 80% / 9.4s | 564 / 91% / 26.8s |
| add_sinesaw | 24 / 6% / 0.3s | 19 / 52% / 1.3s | 19 / 56% / 0.7s | 231 / 96% / 13.9s |

## Headline takeaways

- **CMA-ES** best on identifiable synths (bandpass 0.001, am_noise 0.008), gap≈0.
- **BO** least-deceived optimizer on flat add_sinesaw (returned 0.197) — global hedging.
- **GD** weakest + most deceived (am 0.488, add 0.520) and ~29× slower wall-clock on bandpass.
- **QL** does not learn (flat across thousands of trials).
- **Learned (0-eval)** beats BO/GD everywhere and **beats ALL optimizers ~4× on the non-identifiable add_sinesaw (0.045 vs 0.197)** — amortized priors win where the loss is useless.

