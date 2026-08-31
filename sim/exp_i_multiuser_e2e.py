"""
Experiment I -- multi-user, end to end.

multiuser.py settled the information question: matched pairs cost 2K probes and
lose nothing against the full array.  This runs a real estimator through that
acquisition, since a bound says only that the information is present.

Acquisition, T = 2K + 1:
    Stage A (1 probe)   one squint sweep.  Users appear at distinct peak
                        subcarriers -- the separation mechanism Luo2024 and
                        Zhang2026 already rely on -- giving K coarse ANGLES.
    Stage B (2K probes) one constant-modulus monopulse pair per user.

All K users contribute to all 2K channels, so the estimator is a joint ML over
the 2K position parameters with the K complex amplitudes concentrated out.  It
is initialised from the Stage A angles and from a per-user delay read off that
user's own pair.

Metrics: RMSE is reported alongside the MEDIAN, because RMSE is outlier-driven
and part of the advantage over CBS-High comes from its outliers; quoting only
RMSE would invite exactly that objection.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.optimize import least_squares
from nf_model import Config, C, dist_fresnel
import exp_f_three_probe as F

FIG = os.path.join(os.path.dirname(__file__), "figs")
CKPT = os.path.join(FIG, "exp_i_ckpt.json")
N_TRIAL = int(os.environ.get("EXP_I_TRIALS", 12))
BUDGET = float(os.environ.get("EXP_I_BUDGET", 95))
SNR, R_GUESS = 0.0, 30.0
KS = [1, 2, 3, 5]


def pairs_for(cfg, th_hats, f, delta):
    cols = []
    for th in th_hats:
        for tt in (th - delta, th + delta):
            d = dist_fresnel(R_GUESS, tt, cfg.x[None, :])
            cols.append(np.exp(-2j * np.pi * f[:, None] * d / C) / np.sqrt(cfg.N))
    # Deliberately NOT orthonormalised.  QR mixes the two beams of a pair, so
    # z[:,2k]+z[:,2k+1] would no longer be the physical sum channel and the
    # delay slope the coarse range reads would be destroyed.  The whitener
    # handles the non-orthogonality instead, exactly as in the single-user case.
    return np.stack(cols, axis=2)


def simulate_multi(cfg, users, V, f, snr_db, rng):
    tot = 0
    for (r, th) in users:
        a = np.exp(-2j * np.pi * f[:, None] * dist_fresnel(r, th, cfg.x[None, :]) / C)
        tot = tot + a * np.exp(2j * np.pi * rng.random())
    sig = np.sqrt(10 ** (-snr_db / 10) / 2)
    n = sig * (rng.standard_normal(tot.shape) + 1j * rng.standard_normal(tot.shape))
    return np.einsum('fnt,fn->ft', V.conj(), tot + n)


def estimate_multi(cfg, z, V, f, th0s, r0s, halfwidth):
    """Joint ML over 2K parameters, K amplitudes concentrated out."""
    K = len(th0s)
    V, z, f = V[::F.ML_DECIM], z[::F.ML_DECIM], f[::F.ML_DECIM]
    W = F.whitener(V)
    zw = (z @ W.T).ravel()

    def basis(p):
        H = []
        for k in range(K):
            a = np.exp(-2j * np.pi * f[:, None]
                       * dist_fresnel(p[2 * k], p[2 * k + 1], cfg.x[None, :]) / C)
            H.append((np.einsum('fnt,fn->ft', V.conj(), a) @ W.T).ravel())
        return np.stack(H, axis=1)

    def resid(p):
        H = basis(p)
        alpha = np.linalg.lstsq(H, zw, rcond=None)[0]
        e = zw - H @ alpha
        return np.concatenate([e.real, e.imag])

    p0, lo, hi = [], [], []
    for k in range(K):
        p0 += [r0s[k], th0s[k]]
        lo += [max(0.5, r0s[k] - 5.0), th0s[k] - halfwidth]
        hi += [r0s[k] + 5.0, th0s[k] + halfwidth]
    sol = least_squares(resid, p0, bounds=(lo, hi), method="trf",
                        xtol=1e-13, ftol=1e-13, gtol=1e-13)
    return [(sol.x[2 * k + 1], sol.x[2 * k]) for k in range(K)]


def run():
    cfg = Config()
    f = cfg.freqs[::F.SUB_DECIM]
    delta = 0.25 * 0.886 * cfg.lam / cfg.aperture
    ce = float(np.load(os.path.join(FIG, "exp_c.npz"))["coarse_th"][2])
    done = json.load(open(CKPT)) if os.path.exists(CKPT) else {}
    t0 = time.perf_counter()
    print(f"[I] SNR={SNR:.0f} dB, {N_TRIAL} trials, stage-A angle error {ce:.4f} deg, "
          f"ML_DECIM={F.ML_DECIM}\n")

    for K in KS:
        if str(K) in done:
            continue
        if time.perf_counter() - t0 > BUDGET:
            print(f"  [budget reached, {len(done)}/{len(KS)} done -- re-run to continue]")
            return
        rng = np.random.default_rng(1000 + K)
        er, et = [], []
        for _ in range(N_TRIAL):
            # users spread in angle so Stage A can separate them
            ths = np.deg2rad(np.linspace(-45, 45, K) + rng.normal(0, 2, K))
            rs = 20.0 + 25.0 * rng.random(K)
            users = list(zip(rs, ths))
            th0s = ths + np.deg2rad(ce) * rng.standard_normal(K)
            V = pairs_for(cfg, th0s, f, delta)
            z = simulate_multi(cfg, users, V, f, SNR, rng)
            r0s = []
            for k in range(len(users)):
                r0s.append(F.coarse_range(z[:, 2 * k:2 * k + 2], f, R_GUESS)[0])
            out = estimate_multi(cfg, z, V, f, th0s, np.clip(r0s, 5, 80),
                                 np.deg2rad(3 * ce))
            for k, (r, th) in enumerate(users):
                et.append(out[k][0] - th); er.append(out[k][1] - r)
        rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
        med = lambda v: float(np.median(np.abs(v)))
        done[str(K)] = [np.rad2deg(rms(et)), rms(er),
                        np.rad2deg(med(et)), med(er), 2 * K + 1]
        json.dump(done, open(CKPT, "w"))
        d = done[str(K)]
        print(f"  K={K} (T={d[4]:2d}) | RMSE_th {d[0]:.3e} deg  RMSE_r {d[1]:.3e} m"
              f" | median |dth| {d[2]:.3e} deg  |dr| {d[3]:.3e} m", flush=True)

    print("\n" + "=" * 72)
    print(f"{'K':>3}{'T':>5}{'RMSE_th':>13}{'RMSE_r':>13}{'med|dth|':>13}{'med|dr|':>13}")
    for K in KS:
        d = done[str(K)]
        print(f"{K:>3}{d[4]:>5}{d[0]:>13.3e}{d[1]:>13.3e}{d[2]:>13.3e}{d[3]:>13.3e}")
    np.savez(os.path.join(FIG, "exp_i.npz"), K=KS,
             **{k: [done[str(x)][i] for x in KS] for i, k in
                enumerate(["rmse_th", "rmse_r", "med_th", "med_r", "T"])})


if __name__ == "__main__":
    run()
