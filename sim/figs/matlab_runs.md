# MATLAB runs (independent implementation, N_TRIAL = 15/30)

Cross-check of `matlab/nf_proposed.m` against the Python of `sim/`.  Recorded
here because the two implementations were written separately and agreeing is
evidence neither carries a silent modelling error.

## Demo, N=256, M=2048, W=3 GHz (30 trials)

| SNR | RMSE_th (deg) | RMSE_r (m) | RMSE_r / bound |
|---|---|---|---|
| -10 | 1.165e-03 | 2.511e-04 | 1.04 |
| -5 | 5.805e-04 | 1.120e-04 | 0.83 |
| 0 | 3.168e-04 | 6.814e-05 | 0.89 |
| +5 | 3.016e-03 † | 4.515e-05 | 1.05 |
| +10 | 1.277e-04 | 2.918e-05 | 1.21 |
| +15 | 6.627e-05 | 1.701e-05 | 1.25 |
| +20 | 3.416e-05 | 6.497e-06 | 0.85 |

† one trial in thirty landed 0.0165 deg out -- 0.04 beamwidths, well inside the
search window, so a lone convergence failure rather than a sidelobe.  The runner
now prints the median and worst trial so such points are identifiable.

**Agreement with Python.** MATLAB is 2.24x better on average, and Python thins
its refinement to a quarter of the tones, which costs exactly sqrt(4) = 2.  The
full-band MATLAB numbers sit at 0.83-1.25x the bound in both parameters, so the
estimator is statistically efficient and the ~2x in the Python sweeps is entirely
that thinning.

## Sweeps (15 trials)

| N | RMSE_r (m) | median | max | efficiency |
|---|---|---|---|---|
| 64 | 2.062e-04 | 1.818e-04 | 3.844e-04 | 1.35 |
| 128 | 1.452e-04 | 9.998e-05 | 2.311e-04 | 1.35 |
| 256 | 7.021e-05 | 5.344e-05 | 1.231e-04 | 0.92 |
| 512 | 6.120e-05 | 4.877e-05 | 1.677e-04 | 1.13 |
| 1024 | 4.052e-05 | 2.159e-05 | 8.837e-05 | 1.06 |

| W (GHz) | RMSE_r (m) | median | max | efficiency |
|---|---|---|---|---|
| 0.5 | 6.535e-04 | 4.542e-04 | 1.439e-03 | 1.43 |
| 1.0 | 2.474e-04 | 1.743e-04 | 6.108e-04 | 1.08 |
| 2.0 | 1.569e-04 | 1.295e-04 | 2.576e-04 | 1.37 |
| 3.0 | 1.114e-04 | 6.611e-05 | 2.549e-04 | 1.46 |
| 6.0 | 3.032e-05 | 1.578e-05 | 5.826e-05 | 0.79 |

## Fitted exponents, and why they are not yet conclusive

| | Python (15) | MATLAB (15) | predicted | bound |
|---|---|---|---|---|
| vs N | -0.37 | -0.59 | -0.50 | -0.50 exactly |
| vs W | -0.91 | -1.15 | -1.00 | -1.00 exactly |

The two independent implementations land on OPPOSITE sides of the prediction,
which is the signature of noise rather than bias -- a systematic error would push
both the same way.  A tested hypothesis, that efficiency improves as N or W grows
and so steepens the measured slope, is NOT supported: the efficiency column
scatters between 0.79 and 1.46 with no monotone trend in either sweep.

At 15 trials each RMSE carries about 18% standard error, giving a slope standard
error of roughly

    sigma_slope ~ 0.18 / (sqrt(5) * sigma_log_x) ~ 0.08,

so every one of the four fits is within 2 sigma of its prediction.

| trials | sigma_slope | separates -0.50 from -0.60? |
|---|---|---|
| 15 | 0.08 | no |
| 150 | 0.026 | marginally |
| 500 | 0.014 | yes |

Run these two sweeps at N_TRIAL = 500.  They are the cheapest experiments in the
set -- no baselines, no MUSIC -- so there is no reason to leave them underpowered.

The W = 6 GHz point sitting at 0.79x, i.e. below the CRLB, is impossible and is
simply the finite-sample undershoot expected at 15 trials.  If it is STILL below
the bound at 500 trials, that is a real problem with the bound or the model and
should be raised rather than reported.
