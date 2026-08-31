"""
Experiment G -- what actually threatens coherent wideband processing.

Coherent processing across the band is the one thing this proposal needs that
per-subcarrier methods do not, so it is where the approach could fail.  In OFDM
both oscillator phase noise and carrier frequency offset decompose into

    a common phase error, identical on every subcarrier,   plus  ICI,

and the common part is absorbed exactly by the unknown complex amplitude alpha,
which the estimator already marginalises.  So the CPE should cost nothing and
only the ICI residual should bite.

The genuine hazard is a TIMING offset: it puts -2 pi f_m tau on the phase, which
is the exact signature of a range, so it maps one-for-one into a range bias of
c*tau.  Monostatic sensing shares a clock between transmit and receive, so what
matters is short-term stability over the round trip, not absolute timing -- but
the sensitivity has to be quantified rather than assumed.

Three sweeps: common phase error, residual per-subcarrier phase, timing offset.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nf_model import Config, C, dist_fresnel
from acquisition import fim, bounds
import exp_f_three_probe as F

FIG = os.path.join(os.path.dirname(__file__), "figs")
CKPT = os.path.join(FIG, "exp_g_ckpt.json")
BUDGET = float(os.environ.get("EXP_G_BUDGET", 95))
SNR, N_TRIAL = 10.0, 24
R_TRUE, TH_TRUE, R_GUESS = 30.0, np.deg2rad(15.0), 30.0


def impair(z, f, rng, cpe_rms=0.0, pn_rms=0.0, tau=0.0):
    """Apply a common phase error, a residual per-subcarrier phase, and a timing offset."""
    out = z
    if cpe_rms:
        out = out * np.exp(1j * cpe_rms * rng.standard_normal())
    if pn_rms:
        out = out * np.exp(1j * pn_rms * rng.standard_normal(z.shape[0]))[:, None]
    if tau:
        out = out * np.exp(-2j * np.pi * f * tau)[:, None]
    return out


def trial_set(cfg, f, delta, rng, **imp):
    et, er = [], []
    for _ in range(N_TRIAL):
        th0 = TH_TRUE + np.deg2rad(0.016) * rng.standard_normal()
        V = F.beams(cfg, R_GUESS, th0, delta, f)
        z = F.simulate(cfg, R_TRUE, TH_TRUE, V, f, SNR, rng)
        z = impair(z, f, rng, **imp)
        th, r = F.estimate(cfg, z, V, f, th0, R_GUESS)
        et.append(th - TH_TRUE)
        er.append(r - R_TRUE)
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    return np.rad2deg(rms(et)), rms(er), float(np.mean(er))


def run():
    cfg = Config()
    f = cfg.freqs[::2]
    delta = 0.25 * 0.886 * cfg.lam / cfg.aperture
    done = json.load(open(CKPT)) if os.path.exists(CKPT) else {}
    t0 = time.perf_counter()

    V0 = F.beams(cfg, R_GUESS, TH_TRUE, delta, f)
    bt, br = bounds(fim(cfg, R_TRUE, TH_TRUE, SNR, f, np.linalg.qr(V0)[0]))
    print(f"[G] SNR={SNR:.0f} dB, bound: th {bt:.3e} deg, r {br:.3e} m")
    print(f"    (the sweep thins the ML by {F.ML_DECIM}x, so clean sits ~2x above)\n")

    jobs = []
    jobs += [(f"cpe_{v:g}", dict(cpe_rms=v)) for v in (0.0, 0.1, 0.5, 1.0, 3.0)]
    jobs += [(f"pn_{v:g}", dict(pn_rms=v)) for v in (0.01, 0.03, 0.1, 0.3)]
    jobs += [(f"tau_{v:g}", dict(tau=v)) for v in (1e-14, 1e-13, 1e-12, 1e-11)]

    for key, imp in jobs:
        if key in done:
            continue
        if time.perf_counter() - t0 > BUDGET:
            print(f"  [budget reached, {len(done)}/{len(jobs)} done -- re-run to continue]")
            return
        done[key] = trial_set(cfg, f, delta, np.random.default_rng(2026), **imp)
        json.dump(done, open(CKPT, "w"))
        print(f"  {key:<12} RMSE_th {done[key][0]:.3e} deg   RMSE_r {done[key][1]:.3e} m"
              f"   bias_r {done[key][2]:+.3e} m", flush=True)

    clean = done["cpe_0"]
    print("\n" + "=" * 72)
    print("1. Common phase error -- absorbed by alpha?")
    print("=" * 72)
    for v in (0.0, 0.1, 0.5, 1.0, 3.0):
        d = done[f"cpe_{v:g}"]
        print(f"  CPE rms {v:4.1f} rad -> RMSE_r {d[1]:.3e} m   "
              f"{d[1]/clean[1]:5.2f}x clean")

    print("\n" + "=" * 72)
    print("2. Residual per-subcarrier phase (the ICI that survives CPE removal)")
    print("=" * 72)
    for v in (0.01, 0.03, 0.1, 0.3):
        d = done[f"pn_{v:g}"]
        print(f"  sigma_phi {v:5.2f} rad ({np.rad2deg(v):5.2f} deg) -> RMSE_r {d[1]:.3e} m"
              f"   {d[1]/clean[1]:6.2f}x clean")

    print("\n" + "=" * 72)
    print("3. Timing offset -- expected to map one-for-one into c*tau of range bias")
    print("=" * 72)
    for v in (1e-14, 1e-13, 1e-12, 1e-11):
        d = done[f"tau_{v:g}"]
        print(f"  tau {v:.0e} s -> bias {d[2]:+.3e} m   predicted c*tau = {C*v:.3e} m"
              f"   ratio {d[2]/(C*v):5.2f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.1))
    cp = [0.0, 0.1, 0.5, 1.0, 3.0]
    ax[0].semilogy(cp, [done[f"cpe_{v:g}"][1] for v in cp], "o-")
    ax[0].axhline(clean[1], color="k", ls=":", label="clean")
    ax[0].set(xlabel="common phase error rms (rad)", ylabel="RMSE $r$ (m)",
              title="(a) CPE is absorbed by $\\alpha$")
    pn = [0.01, 0.03, 0.1, 0.3]
    ax[1].loglog(pn, [done[f"pn_{v:g}"][1] for v in pn], "s-")
    ax[1].axhline(clean[1], color="k", ls=":", label="clean")
    ax[1].set(xlabel="residual per-subcarrier phase rms (rad)", ylabel="RMSE $r$ (m)",
              title="(b) ICI residual")
    tt = [1e-14, 1e-13, 1e-12, 1e-11]
    ax[2].loglog(tt, [abs(done[f"tau_{v:g}"][2]) for v in tt], "d-", label="measured bias")
    ax[2].loglog(tt, [C * v for v in tt], "k--", label="$c\\tau$")
    ax[2].set(xlabel="timing offset $\\tau$ (s)", ylabel="range bias (m)",
              title="(c) Timing offset is the real hazard")
    for a in ax:
        a.grid(True, which="both", alpha=.3); a.legend(fontsize=8)
    fig.suptitle("Experiment G: impairment sensitivity of coherent wideband processing", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "exp_g_impairments.png"), dpi=140)


if __name__ == "__main__":
    run()
