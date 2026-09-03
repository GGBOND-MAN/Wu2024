"""
Experiment J -- accuracy per probe: the head-to-head the whole argument rests on.

Every scheme is run at the SAME user position and the SAME SNR grid, so the
comparison is not confounded by where the target sits:

    CBS-Low   (Luo2024)   T = 2
    CBS-High  (Luo2024)   T = 10   -- tuned in its own favour, see baselines.py
    Zhang2026             T = 256  -- one array snapshot needs N sequential probes
    proposed              T = 3

Zhang2026 is additionally reported at its own best case, a user sitting exactly
on the designed squint trajectory, since Experiment B showed its coarse stage
only works there; quoting it solely at a position it was never going to handle
would be the same strawmanning this work criticises.

Panel (c) gives the error CDF rather than RMSE alone.  RMSE is outlier-driven,
and part of the margin over CBS-High comes from its outliers, so the median and
the tail are shown side by side.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nf_model import Config, trajectory, ttd_ps_weights, coarse_observation
from estimators import coarse_estimate, music_refine, make_snapshot
import exp_f_three_probe as F
import baselines as B

FIG = os.path.join(os.path.dirname(__file__), "figs")
CKPT = os.path.join(FIG, "exp_j_ckpt.json")
N_TRIAL = int(os.environ.get("EXP_J_TRIALS", 30))
BUDGET = float(os.environ.get("EXP_J_BUDGET", 95))
SNRS = [-10, -5, 0, 5, 10, 15, 20]
R_TRUE, TH_TRUE = 30.0, np.deg2rad(15.0)
TS, RS, TE, RE = np.deg2rad(-60), 15.0, np.deg2rad(60), 50.0
PROBES = {"cbs_low": 2, "cbs_high": 10, "zhang": 256, "proposed": 3}


def zhang_pipeline(cfg, r_k, th_k, snr, rng, n_sub=64):
    """Zhang2026: squint power peak, then geometry-compensated local MUSIC.

    |S| is the one calibration target that cannot be fitted: its only anchor is
    the claim of sub-millimetre ranging, and no |S| reaches it -- the incoherent
    bound of its own class is 4.9e-2 m at |S| = M.  So it is set as large as is
    tractable and reported as the most generous assumption available.
    """
    th_t, r_t = trajectory(cfg, TS, RS, TE, RE)
    phi, tau = ttd_ps_weights(cfg, TS, RS, TE, RE)
    z = coarse_observation(cfg, r_k, th_k, phi, tau)
    th0, r0, _ = coarse_estimate(z, th_t, r_t, snr, rng)
    S = cfg.freqs[:: cfg.M // n_sub][:n_sub]
    ys = [make_snapshot(cfg, r_k, th_k, f, snr, rng, "exact") for f in S]
    return music_refine(ys, cfg, th0, r0, S, model="fresnel")


def run():
    cfg = Config()
    f = cfg.freqs[::F.SUB_DECIM]
    delta = 0.25 * 0.886 * cfg.lam / cfg.aperture
    ce_all = np.load(os.path.join(FIG, "exp_c.npz"))["coarse_th"]
    done = json.load(open(CKPT)) if os.path.exists(CKPT) else {}
    t0 = time.perf_counter()
    print(f"[J] all schemes at ({np.rad2deg(TH_TRUE):.0f} deg, {R_TRUE} m), "
          f"{N_TRIAL} trials, ML_DECIM={F.ML_DECIM}\n")

    for i, snr in enumerate(SNRS):
        if str(snr) in done:
            continue
        if time.perf_counter() - t0 > BUDGET:
            print(f"  [budget reached, {len(done)}/{len(SNRS)} done -- re-run]")
            return
        ce = max(float(ce_all[i]), 1e-3)
        errs = {k: [] for k in PROBES}
        rng = np.random.default_rng(9000 + i)
        for _ in range(N_TRIAL):
            errs["cbs_low"].append(B.cbs_low(cfg, R_TRUE, TH_TRUE, snr, rng)[1] - R_TRUE)
            errs["cbs_high"].append(B.cbs_high(cfg, R_TRUE, TH_TRUE, snr, rng)[1] - R_TRUE)
            errs["zhang"].append(zhang_pipeline(cfg, R_TRUE, TH_TRUE, snr, rng)[1] - R_TRUE)
            th0 = TH_TRUE + np.deg2rad(ce) * rng.standard_normal()
            V = F.beams(cfg, F.R_GUESS, th0, delta, f)
            z = F.simulate(cfg, R_TRUE, TH_TRUE, V, f, snr, rng)
            errs["proposed"].append(
                F.estimate(cfg, z, V, f, th0, F.R_GUESS,
                           th_halfwidth=np.deg2rad(3 * ce))[1] - R_TRUE)
        done[str(snr)] = {k: [float(np.sqrt(np.mean(np.square(v)))),
                              float(np.median(np.abs(v))),
                              float(np.percentile(np.abs(v), 90))]
                          for k, v in errs.items()}
        if snr == 0:
            done["cdf0"] = {k: sorted(np.abs(v).tolist()) for k, v in errs.items()}
        json.dump(done, open(CKPT, "w"))
        d = done[str(snr)]
        print(f"  SNR={snr:+3d} | " + " | ".join(
            f"{k} {d[k][0]:.2e}" for k in PROBES), flush=True)

    G = lambda k, i=0: [done[str(s)][k][i] for s in SNRS]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    style = {"cbs_low": ("^-", "CBS-Low (T=2)"), "cbs_high": ("v-", "CBS-High (T=10)"),
             "zhang": ("s-", "Zhang2026 (T=256)"), "proposed": ("o-", "proposed (T=3)")}
    for k, (st, lbl) in style.items():
        ax[0].semilogy(SNRS, G(k), st, label=lbl)
    ax[0].set(xlabel="per-element SNR (dB)", ylabel="RMSE $r$ (m)", title="(a) vs SNR")

    # Zhang2026 at its own best case: Experiment C placed the user exactly on the
    # designed trajectory, which is the only place its coarse stage works.  At the
    # position used here it is 71.6 m out, and quoting only that would strawman it.
    zc = np.load(os.path.join(FIG, "exp_c.npz"))
    ax[0].semilogy(zc["snr"], zc["rmse_all"], "s--", color="tab:red", alpha=.55,
                   label="Zhang2026, on-trajectory (its best case)")
    for k, (st, lbl) in style.items():
        ax[1].loglog(PROBES[k], done["0"][k][0], st[0], ms=11, label=lbl)
    ax[1].loglog(256, float(zc["rmse_all"][2]), "s", ms=11, color="tab:red", alpha=.55,
                 label="Zhang2026, on-trajectory")
    ax[1].set(xlabel="analog probes $T$", ylabel="RMSE $r$ (m) at 0 dB",
              title="(b) accuracy per probe")

    for k, (st, lbl) in style.items():
        v = np.array(done["cdf0"][k])
        ax[2].semilogx(np.maximum(v, 1e-9), np.arange(1, len(v) + 1) / len(v),
                       label=lbl, lw=1.8)
    ax[2].set(xlabel="|range error| (m) at 0 dB", ylabel="empirical CDF",
              title="(c) error distribution, not just RMSE")
    for a in ax:
        a.grid(True, which="both", alpha=.3); a.legend(fontsize=7.5)
    fig.suptitle("Experiment J: same user, same SNR, same noise -- accuracy against "
                 "acquisition cost", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "exp_j_overhead.png"), dpi=140)

    print("\n" + "=" * 74)
    print(f"{'scheme':<20}{'T':>5}{'RMSE_r @0dB':>15}{'median':>13}{'90th pct':>13}")
    for k in PROBES:
        d = done["0"][k]
        print(f"{style[k][1]:<20}{PROBES[k]:>5}{d[0]:>15.3e}{d[1]:>13.3e}{d[2]:>13.3e}")
    zbest = float(np.load(os.path.join(FIG, "exp_c.npz"))["rmse_all"][2])
    print(f"{'Zhang2026, on-trajectory':<20}{256:>5}{zbest:>15.3e}"
          f"{'(its best case)':>26}")
    pr = done["0"]["proposed"]
    print(f"\n  vs CBS-High  : {done['0']['cbs_high'][0]/pr[0]:6.0f}x RMSE, "
          f"{done['0']['cbs_high'][1]/pr[1]:6.0f}x median, at 3.3x fewer probes")
    print(f"  vs Zhang2026 : {zbest/pr[0]:6.0f}x RMSE even against its BEST case, "
          f"at {256/3:.0f}x fewer probes")
    print(f"  (at this position Zhang2026 is {done['0']['zhang'][0]:.1f} m out, the "
          f"off-trajectory failure of Experiment B --\n   reported but not used as "
          f"the headline number)")


if __name__ == "__main__":
    run()
