"""
Experiment F -- end-to-end, with only three analog probes.

Everything so far has been CRLB analysis: it says the information is there.
This runs an actual estimator on actual noisy data through the three-probe
acquisition and checks it reaches the bound.

  Stage A (1 probe)  a beam-squint sweep gives the coarse ANGLE (Experiment B:
                     0.016 deg median).  Its range output is discarded.
  Stage B (2 probes) a constant-modulus monopulse pair straddling that angle by
                     delta = 0.25 beamwidths, with the range guess pinned at a
                     fixed 30 m -- never estimated.

The per-antenna delay trick is gone here, since T=2 leaves no array snapshot.
It is not needed: after combining,

    h[t,m] = (1/sqrt(N)) sum_n exp(-j 2 pi f_m [r_n(r,th) - r_n(r0,th_t)] / c)

and with the beam roughly matched the bracket is nearly constant in n, so the
phase of z[t,:] against frequency still carries the range as a delay slope.
That initialises a two-parameter local ML.

Noise is referred to the antennas, so the combined noise covariance is
sigma^2 V^H V; the likelihood is whitened accordingly rather than pretending
the two beams are orthogonal.

NOTE on the swept curves: the ML refinement is thinned by ML_DECIM to keep the
sweep tractable, which costs sqrt(ML_DECIM) and puts the curves about 2x above
the bound.  That factor is the thinning, not the estimator -- a spot check at
0 dB with the full band gives 0.98x in range and 1.39x in angle.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nf_model import Config, C, dist_fresnel
from acquisition import fim, bounds

FIG = os.path.join(os.path.dirname(__file__), "figs")
SNRS = [-10, -5, 0, 5, 10, 15, 20]
N_TRIAL = 30
SUB_DECIM = 2           # use every 2nd subcarrier throughout: 1024 tones keep the
                        # unambiguous range at 51.2 m, still covering the whole
                        # 15-50 m sensing region
ML_DECIM = 4            # ML refinement thins further; the coarse FFT does not
CKPT = os.path.join(os.path.dirname(__file__), "figs", "exp_f_ckpt.json")
TIME_BUDGET = float(os.environ.get("EXP_F_BUDGET", 95))
R_GUESS = 30.0                      # fixed; never estimated

# The coarse angle handed to Stage B is not a constant.  Experiment B measured
# 0.016 deg NOISELESSLY; under noise Experiment C measured it per SNR, and that
# is what Stage B actually receives.  Using the noiseless figure at every SNR
# would flatter the scheme at the low end, so the measured values are used.
_c = np.load(os.path.join(FIG, "exp_c.npz"))
COARSE_ANGLE_ERR_DEG = dict(zip([int(v) for v in _c["snr"]], _c["coarse_th"]))


def beams(cfg, r0, th0, delta, f):
    """Raw constant-modulus monopulse pair (N x T per frequency)."""
    cols = []
    for tt in (th0 - delta, th0 + delta):
        d = dist_fresnel(r0, tt, cfg.x[None, :])
        cols.append(np.exp(-2j * np.pi * f[:, None] * d / C) / np.sqrt(cfg.N))
    return np.stack(cols, axis=2)                       # (F, N, 2)


def response(cfg, r, th, V, f):
    """h[t,m] = v_{t,m}^H a_m(theta, r)."""
    a = np.exp(-2j * np.pi * f[:, None] * dist_fresnel(r, th, cfg.x[None, :]) / C)
    return np.einsum('fnt,fn->ft', V.conj(), a)         # (F, T)


def simulate(cfg, r, th, V, f, snr_db, rng):
    """Antenna-referred noise, then combining: z = V^H (a alpha + n)."""
    a = np.exp(-2j * np.pi * f[:, None] * dist_fresnel(r, th, cfg.x[None, :]) / C)
    a = a * np.exp(2j * np.pi * rng.random())
    sig = np.sqrt(10 ** (-snr_db / 10) / 2)
    n = sig * (rng.standard_normal(a.shape) + 1j * rng.standard_normal(a.shape))
    return np.einsum('fnt,fn->ft', V.conj(), a + n)


def whitener(V):
    """L^-H such that L^-H z has identity noise covariance (noise cov = V^H V)."""
    G = np.einsum('fnt,fns->ts', V.conj(), V) / V.shape[0]
    L = np.linalg.cholesky(G + 1e-12 * np.eye(G.shape[0]))
    return np.linalg.inv(L).conj().T


def coarse_range(z, f, r0):
    """Range from the phase slope of the sum channel, relative to the beam focus."""
    s = z.sum(axis=1)
    df = f[1] - f[0]
    nfft = 1 << int(np.ceil(np.log2(len(f) * 8)))
    prof = np.fft.fft(np.conj(s), n=nfft)
    k = int(np.argmax(np.abs(prof[: nfft // 2])))
    dr = k * C / (nfft * df)
    # The FFT resolves |r - r0|; the sign follows from which side improves the fit.
    return r0 + dr, r0 - dr


def estimate(cfg, z, V, f, th_init, r0, th_halfwidth=None):
    """Two-parameter local ML on the whitened likelihood.

    The angle search is bounded to th_init +- th_halfwidth.  Unbounded
    Nelder-Mead occasionally walks into a sidelobe of the combined pattern about
    1.5 beamwidths out and returns a confidently wrong angle; the bound keeps it
    in the mainlobe.  The half-width is set to three times the MEASURED stage-A
    angle RMSE, not tuned to the answer, and the fraction of trials that leave
    the window is reported rather than hidden."""
    cands = [x for x in coarse_range(z, f, r0) if 0.5 < x < 200]
    if len(cands) == 2 and abs(cands[0] - cands[1]) < 1e-6:
        cands = cands[:1]                      # the FFT found no offset; one start suffices
    V, z, f = V[::ML_DECIM], z[::ML_DECIM], f[::ML_DECIM]
    W = whitener(V)
    zw = z @ W.T

    def nll(p):
        h = response(cfg, p[0], p[1], V, f) @ W.T
        num = abs(np.vdot(h, zw)) ** 2
        den = np.vdot(h, h).real
        return -num / max(den, 1e-300)

    def resid(p):
        """Complex residual with the amplitude concentrated out, split for LM."""
        h = response(cfg, p[0], p[1], V, f) @ W.T
        alpha = np.vdot(h, zw) / max(np.vdot(h, h).real, 1e-300)
        e = (zw - alpha * h).ravel()
        return np.concatenate([e.real, e.imag])

    best, best_cost = None, np.inf
    for r_try in cands:
        lo = [max(0.5, r_try - 5.0), th_init - (th_halfwidth or np.deg2rad(1.0))]
        hi = [r_try + 5.0, th_init + (th_halfwidth or np.deg2rad(1.0))]
        sol = least_squares(resid, [r_try, th_init], bounds=(lo, hi),
                            method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if sol.cost < best_cost:
            best, best_cost = sol, sol.cost

    return (best.x[1], best.x[0]) if best is not None else (th_init, r0)


def run():
    cfg = Config()
    rng = np.random.default_rng(4242)
    f = cfg.freqs[::SUB_DECIM]
    r_true, th_true = 30.0, np.deg2rad(15.0)
    bw = 0.886 * cfg.lam / cfg.aperture
    delta = 0.25 * bw

    print(f"[F] user ({np.rad2deg(th_true):.3f} deg, {r_true} m), N={cfg.N}, M={cfg.M}")
    print(f"    monopulse delta = {np.rad2deg(delta):.4f} deg (0.25 beamwidths), "
          f"range guess pinned at {R_GUESS} m, never estimated")
    print(f"    T = 3 probes total (1 squint sweep for angle + 2 monopulse beams)")
    print(f"    stage-A angle error per SNR (measured in Experiment C): "
          f"{ {k: round(float(v), 4) for k, v in COARSE_ANGLE_ERR_DEG.items()} }\n")

    import json, time
    done = json.load(open(CKPT)) if os.path.exists(CKPT) else {}
    t0 = time.perf_counter()
    for snr in SNRS:
        if str(snr) in done:
            print(f"  SNR={snr:+3d} dB | cached", flush=True)
            continue
        if time.perf_counter() - t0 > TIME_BUDGET:
            print(f"  [budget reached, {len(done)}/{len(SNRS)} done -- re-run to continue]")
            return
        ce = max(COARSE_ANGLE_ERR_DEG[snr], 1e-3)   # measured stage-A angle RMSE
        et, er, n_edge = [], [], 0
        for _ in range(N_TRIAL):
            th_hat0 = th_true + np.deg2rad(ce) * rng.standard_normal()
            V = beams(cfg, R_GUESS, th_hat0, delta, f)
            z = simulate(cfg, r_true, th_true, V, f, snr, rng)
            th, r = estimate(cfg, z, V, f, th_hat0, R_GUESS,
                             th_halfwidth=np.deg2rad(3 * ce))
            et.append(th - th_true)
            er.append(r - r_true)
            n_edge += abs(th - th_hat0) > 0.98 * np.deg2rad(3 * ce)
        rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
        V0 = beams(cfg, R_GUESS, th_true, delta, f)
        bt, br = bounds(fim(cfg, r_true, th_true, snr, f, np.linalg.qr(V0)[0]))
        done[str(snr)] = [np.rad2deg(rms(et)), rms(er), bt, br, 100.0 * n_edge / N_TRIAL]
        json.dump(done, open(CKPT, "w"))
        d = done[str(snr)]
        print(f"  SNR={snr:+3d} dB | RMSE_th {d[0]:.3e} deg ({d[0]/d[2]:5.2f}x) | "
              f"RMSE_r {d[1]:.3e} m ({d[1]/d[3]:5.2f}x) | window +-{3*ce:.3f} deg, "
              f"{d[4]:.0f}% at edge", flush=True)

    res_th = [done[str(s)][0] for s in SNRS]; res_r = [done[str(s)][1] for s in SNRS]
    b_th = [done[str(s)][2] for s in SNRS];   b_r = [done[str(s)][3] for s in SNRS]

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for a, est, bd, lbl, unit in ((ax[0], res_th, b_th, "$\\theta$", "deg"),
                                  (ax[1], res_r, b_r, "$r$", "m")):
        a.semilogy(SNRS, est, "o-", label="three-probe estimator")
        a.semilogy(SNRS, bd, "k:", label="CRLB through the same combiners")
        a.set(xlabel="per-element SNR (dB)", ylabel=f"RMSE {lbl} ({unit})")
        a.grid(True, which="both", alpha=.3); a.legend(fontsize=8)
    ax[1].axhline(0.904, color="tab:red", ls="--", lw=1,
                  label="reproduced Zhang2026 @0 dB (T=256)")
    ax[1].legend(fontsize=7)
    ax[0].set_title("(a) Angle"); ax[1].set_title("(b) Range")
    fig.suptitle("Experiment F: end-to-end with three analog probes, "
                 "range guess never estimated", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "exp_f_three_probe.png"), dpi=140)
    np.savez(os.path.join(FIG, "exp_f.npz"), snr=SNRS, rmse_th=res_th, rmse_r=res_r,
             crlb_th=b_th, crlb_r=b_r)


if __name__ == "__main__":
    run()
