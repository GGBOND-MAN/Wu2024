#!/bin/bash
# Publication-scale rerun.  Everything in sim/ is scale-configurable; the
# defaults committed here are the quick checks used while developing, which
# carry roughly 11% standard error on each RMSE and a systematic ~2x from
# thinning the ML refinement.  Set the variables below for final figures.
#
# Expect hours, not minutes.  Delete figs/*_ckpt.json first for a clean run;
# every experiment checkpoints per point and resumes if interrupted.
set -e
cd "$(dirname "$0")"
export EXP_H_TRIALS=500 EXP_H_MLDECIM=1 EXP_H_BUDGET=1e9
export EXP_E_BUDGET=1e9 EXP_F_BUDGET=1e9 EXP_G_BUDGET=1e9 BASELINE_BUDGET=1e9

python3 - <<'PY'
import re, pathlib
# lift the development-scale constants to publication scale
for f, subs in {
    "exp_f_three_probe.py": [("N_TRIAL = 30", "N_TRIAL = 500"),
                             ("ML_DECIM = 4", "ML_DECIM = 1"),
                             ("SUB_DECIM = 2", "SUB_DECIM = 1")],
    "exp_e_proposed.py":    [("N_TRIAL = 40", "N_TRIAL = 500")],
    "exp_g_impairments.py": [("SNR, N_TRIAL = 10.0, 24", "SNR, N_TRIAL = 10.0, 500")],
    "baselines.py":         [("N_TRIAL = 40", "N_TRIAL = 500"),
                             ("n_grid=1500", "n_grid=8000")],
    "exp_a_inverse_crime.py": [("SNR_DB, N_TRIAL, N_SUB = 10, 30, 5",
                                "SNR_DB, N_TRIAL, N_SUB = 10, 500, 5")],
    "exp_c_threshold.py":   [("N_TRIAL, N_SUB = 120, 5", "N_TRIAL, N_SUB = 1000, 5")],
}.items():
    p = pathlib.Path(f); t = p.read_text()
    for a, b in subs:
        t = t.replace(a, b)
    p.write_text(t)
print("scaled up: exp_a, exp_c, exp_e, exp_f, exp_g, baselines, exp_h")
PY

for s in scaling_law acquisition multiuser baselines \
         exp_a_inverse_crime exp_b_trajectory exp_c_threshold \
         exp_e_proposed exp_f_three_probe exp_g_impairments exp_h_position; do
  echo "=== $s ==="
  python3 "$s.py" || echo "  ($s failed -- continuing)"
done
echo "done; figures and .npz are in sim/figs/"
