"""
Multi-user: how does the probe count scale with the number of targets?

Zhang2026 localises up to K = 20 users (its Fig. 7 and Fig. 12), so a
single-target comparison only covers part of what it claims.  Two questions:

  1. Can beam-squint RECEIVE combiners serve every user at once, giving T = O(1)?
  2. If not, what does T = 2K cost against Zhang2026's fixed T = N = 256?

On (1) there is a tension worth stating plainly.  Squint separates users in the
frequency domain -- that is exactly how Luo2024 and Zhang2026 tell users apart.
But frequency is precisely the resource delay ranging lives on.  A squint beam
illuminates a given user only over roughly

    M * (beamwidth / angular span) = 2048 * (0.398 / 120) ~ 7 subcarriers,

so a scheme that separates users by squint hands each user about 7 tones instead
of 2048, and RMSE_r ~ c / (W_eff sqrt(N M_eff)) collapses.  Separation and
ranging want the same axis, and cannot both have it.

On (2), K users with unknown amplitudes span a 2K-dimensional informative
subspace, so T = 2K is not a shortcoming of the design but close to what the
problem requires.  Capturing the whole N-dimensional space, as Zhang2026 does,
only becomes the cheaper option once 2K exceeds N.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from nf_model import Config, C, dist_fresnel


def steer_and_derivs(cfg, r, th, f):
    x = cfg.x[None, :]
    fc = np.asarray(f)[:, None]
    k = 2 * np.pi * fc / C
    a = np.exp(-2j * np.pi * fc * dist_fresnel(r, th, x) / C)
    dth = a * 1j * k * (x * np.cos(th) - x ** 2 * np.sin(2 * th) / (2 * r))
    dr = a * 1j * (-k) * (1 - x ** 2 * np.cos(th) ** 2 / (2 * r ** 2))
    return a, dth, dr


def crlb_multi(cfg, users, snr_db, f, Vs=None):
    """CRLB for K users with K unknown complex amplitudes marginalised.

    users : list of (r, theta).  Vs : (F, N, T) orthonormal, or None for V = I.
    Returns per-user (RMSE_theta_deg, RMSE_r_m).
    """
    A, D = [], []
    for (r, th) in users:
        a, dth, dr = steer_and_derivs(cfg, r, th, f)
        if Vs is not None:
            p = lambda X: np.einsum('fnt,fn->ft', Vs.conj(), X)
            a, dth, dr = p(a), p(dth), p(dr)
        A.append(a.ravel()); D += [dth.ravel(), dr.ravel()]
    A = np.stack(A, axis=1)                       # (FT, K)
    D = np.stack(D, axis=1)                       # (FT, 2K)
    D = D - A @ np.linalg.lstsq(A, D, rcond=None)[0]      # project off span(A)
    J = 2 * 10 ** (snr_db / 10) * np.real(D.conj().T @ D)
    Ci = np.linalg.inv(J + 1e-12 * np.eye(J.shape[0]) * np.trace(J) / J.shape[0])
    d = np.sqrt(np.abs(np.diag(Ci)))
    return [(np.rad2deg(d[2 * i]), d[2 * i + 1]) for i in range(len(users))]


def V_pairs(cfg, users, f, delta):
    """Two constant-modulus beams per user, straddling that user's angle."""
    cols = []
    for (r, th) in users:
        for tt in (th - delta, th + delta):
            d = dist_fresnel(30.0, tt, cfg.x[None, :])
            cols.append(np.exp(-2j * np.pi * np.asarray(f)[:, None] * d / C) / np.sqrt(cfg.N))
    return np.linalg.qr(np.stack(cols, axis=2))[0]


def V_squint_rx(cfg, f, T, region):
    """Beam-squint RECEIVE combiners -- the T = O(1) idea under test."""
    th_lo, th_hi, r_lo, r_hi = region
    V = np.zeros((len(f), cfg.N, T), dtype=complex)
    for t in range(T):
        s = (t + 0.5) / T
        phi = cfg.f0 * dist_fresnel(r_lo, th_lo + s * (th_hi - th_lo), cfg.x) / C
        tau = cfg.fM * dist_fresnel(r_hi, th_hi - s * (th_hi - th_lo), cfg.x) / (cfg.W * C) - phi / cfg.W
        ft = (np.asarray(f) - cfg.f0)[:, None]
        V[:, :, t] = np.exp(-2j * np.pi * (phi[None, :] + ft * tau[None, :])) / np.sqrt(cfg.N)
    return np.linalg.qr(V)[0]


def main():
    cfg = Config()
    f = cfg.freqs[:: cfg.M // 64][:64]
    snr = 0.0
    delta = 0.25 * 0.886 * cfg.lam / cfg.aperture
    region = (cfg.theta_min, cfg.theta_max, cfg.r_min, cfg.r_max)
    rng = np.random.default_rng(3)

    print("=" * 78)
    print("Probe count vs number of users (worst-user range bound, SNR = 0 dB)")
    print("=" * 78)
    print(f"{'K':>3} | {'full array T=256':>18} | {'matched pairs T=2K':>22} | "
          f"{'squint Rx T=2':>16}")
    for K in (1, 2, 3, 5, 8):
        users = [(20.0 + 25.0 * rng.random(), np.deg2rad(-50 + 100 * rng.random()))
                 for _ in range(K)]
        wr = lambda res: max(v[1] for v in res)
        full = wr(crlb_multi(cfg, users, snr, f))
        pair = wr(crlb_multi(cfg, users, snr, f, V_pairs(cfg, users, f, delta)))
        try:
            sq = wr(crlb_multi(cfg, users, snr, f, V_squint_rx(cfg, f, 2, region)))
            sqs = f"{sq:.3e} ({sq/full:6.0f}x)"
        except np.linalg.LinAlgError:
            sqs = "singular"
        print(f"{K:>3} | {full:>18.3e} | {pair:>10.3e} ({pair/full:5.2f}x) | {sqs:>16}")

    print("\n" + "=" * 78)
    print("Probe budget: ours 2K+1 vs Zhang2026's fixed N = 256")
    print("=" * 78)
    for K in (1, 2, 5, 10, 20, 50, 128, 200):
        t = 2 * K + 1
        tag = "cheaper" if t < cfg.N else ("break-even" if t == cfg.N else "dearer")
        print(f"  K = {K:3d}  ->  T = {t:3d}   vs 256   {cfg.N/t:6.1f}x   {tag}")
    print(f"\n  Crossover at K = {(cfg.N - 1) // 2}: beyond it, capturing the whole")
    print(f"  N-dimensional space costs less than 2K user-specific dimensions.")


if __name__ == "__main__":
    main()
