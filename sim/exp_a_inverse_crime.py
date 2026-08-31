"""
Experiment A -- the inverse crime behind Zhang2026 Fig. 13.

The paper derives (Eq. 17/18) that the Fresnel truncation leaves a residual
phase error growing like N^4 / r^3, then uses the Fresnel steering vector inside
MUSIC anyway.  Fig. 13 nonetheless shows the RMSE falling monotonically all the
way to N = 1024 with no error floor, which can only happen if the simulated data
were generated with the same approximate model the estimator assumes.

Here stage I is handed the TRUE position, so stage II is measured in the most
generous possible setting and every error we see is attributable to stage II
itself plus the model gap.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nf_model import Config, C, dist_exact, dist_fresnel
from estimators import music_refine, make_snapshot

N_LIST = [64, 128, 256, 512, 768, 1024]
SNR_DB, N_TRIAL, N_SUB = 10, 30, 5
R_TRUE, TH_TRUE = 30.0, np.deg2rad(15.0)


def residual_phase_deg(cfg, r, th):
    """Peak Fresnel-vs-exact phase error across the aperture, in degrees."""
    x = cfg.x
    dd = dist_exact(r, th, x) - dist_fresnel(r, th, x)
    return np.max(np.abs(2 * np.pi * cfg.fc * dd / C)) * 180 / np.pi


def run():
    rng = np.random.default_rng(2024)
    out = {"fresnel": ([], []), "exact": ([], [])}
    phase_err = []

    for N in N_LIST:
        cfg = Config(N=N, Ms=N // 2)   # paper uses Ms = N/2 (256 -> 128)
        S = cfg.freqs[:: cfg.M // N_SUB][:N_SUB]
        phase_err.append(residual_phase_deg(cfg, R_TRUE, TH_TRUE))
        for gen in ("fresnel", "exact"):
            eth, er = [], []
            for _ in range(N_TRIAL):
                ys = [make_snapshot(cfg, R_TRUE, TH_TRUE, f, SNR_DB, rng, gen) for f in S]
                # Stage I is given the exact truth -- best case for the paper.
                th, r = music_refine(ys, cfg, TH_TRUE, R_TRUE, S, model="fresnel")
                eth.append(th - TH_TRUE)
                er.append(r - R_TRUE)
            out[gen][0].append(np.rad2deg(np.sqrt(np.mean(np.square(eth)))))
            out[gen][1].append(np.sqrt(np.mean(np.square(er))))
            print(f"  N={N:5d} data={gen:7s} RMSE_th={out[gen][0][-1]:.3e} deg  "
                  f"RMSE_r={out[gen][1][-1]:.3e} m", flush=True)
        print(f"  N={N:5d} peak Fresnel phase error = {phase_err[-1]:7.1f} deg", flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for gen, lbl, st in (("fresnel", "data generated with Fresnel model (inverse crime)", "o--"),
                         ("exact", "data generated with exact spherical wave", "s-")):
        ax[0].loglog(N_LIST, out[gen][0], st, label=lbl)
        ax[1].loglog(N_LIST, out[gen][1], st, label=lbl)
    ax[0].axhline(1e-3, color="k", ls=":", label="Zhang2026 claim  0.001 deg")
    ax[1].axhline(1e-3, color="k", ls=":", label="Zhang2026 claim  0.001 m")
    ax[0].set(xlabel="Number of antennas N", ylabel="RMSE $\\theta$ (deg)", title="(a) Angle")
    ax[1].set(xlabel="Number of antennas N", ylabel="RMSE $r$ (m)", title="(b) Range")
    ax[2].loglog(N_LIST, phase_err, "d-", color="crimson")
    ax[2].axhline(180, color="k", ls=":", label="half a wavelength")
    ax[2].set(xlabel="Number of antennas N", ylabel="peak Fresnel phase error (deg)",
              title="(c) Model error, Zhang2026 Eq. (18)")
    for a in ax:
        a.grid(True, which="both", alpha=.3)
        a.legend(fontsize=7)
    fig.suptitle(f"Experiment A: Zhang2026 Fig. 13 reproduced with and without the inverse crime "
                 f"(r={R_TRUE} m, $\\theta$={np.rad2deg(TH_TRUE):.0f}$\\degree$, SNR={SNR_DB} dB, "
                 f"stage I given the exact truth)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "figs", "exp_a_inverse_crime.png"), dpi=140)
    np.savez(os.path.join(os.path.dirname(__file__), "figs", "exp_a.npz"),
             N=N_LIST, phase_err=phase_err,
             **{f"{g}_{k}": v for g in out for k, v in zip(("th", "r"), out[g])})


if __name__ == "__main__":
    run()
