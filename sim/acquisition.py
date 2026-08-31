"""
How many analog probes does delay-domain localization actually need?

This is the paper's exposed flank.  Luo2024's CBS-Low reads a scalar per
subcarrier and needs 2 beam sweeps.  Zhang2026 needs the full N-element array
snapshot, which under a single RF chain costs T = N = 256 sequential probing
intervals (its Eq. 9-11), and the proposed estimator inherits that cost.  A
reviewer will ask whether the comparison is fair.  So: what is the minimum T?

With T analog combiners the observation per subcarrier is z_m = V_m^H a_m alpha,
the parameter derivatives become V_m^H d_{i,m}, and

    J_ij  ~  sum_m Re{ d_{i,m}^H V_m V_m^H d_{j,m} }.

Everything the estimator can learn about (theta, r) therefore lives in the
THREE-dimensional subspace span{a_m, da_m/dtheta, da_m/dr}.  If the columns of
V_m span it, no information is lost -- so T = 3 suffices in the oracle case,
and the practical T is set by how well a fixed V covers the sensing region.

Note the subspace is frequency-dependent: a_m rotates with f_m.  A phase-shifter
-only combiner is frequency-flat and cannot track it, but a PS+TTD combiner can
-- which is an information-theoretic argument for the TTD hardware that the
beam-squint literature only ever justifies as a way to suppress squint.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from nf_model import Config, C, dist_fresnel


def derivs(cfg, r, th, f):
    """a_m and its derivatives w.r.t. theta and r, for each frequency in f."""
    x = cfg.x[None, :]
    f = np.asarray(f)[:, None]
    k = 2 * np.pi * f / C
    a = np.exp(-2j * np.pi * f * dist_fresnel(r, th, x) / C)
    dth = a * 1j * k * (x * np.cos(th) - x ** 2 * np.sin(2 * th) / (2 * r))
    dr = a * 1j * (-k) * (1 - x ** 2 * np.cos(th) ** 2 / (2 * r ** 2))
    return a, dth, dr                                   # each (F, N)


def fim(cfg, r, th, snr_db, f, Vs=None):
    """Coherent wideband FIM, optionally through combiners.

    Vs : None for the full array (V = I_N), else (F, N, T) with orthonormal
         columns per frequency, or (N, T) for a frequency-flat combiner.
    """
    a, dth, dr = derivs(cfg, r, th, f)
    if Vs is None:
        A, Dth, Dr = a, dth, dr
    else:
        if Vs.ndim == 2:
            Vs = np.broadcast_to(Vs, (len(f),) + Vs.shape)
        proj = lambda X: np.einsum('fnt,fn->ft', Vs.conj(), X)
        A, Dth, Dr = proj(a), proj(dth), proj(dr)
    av = A.ravel()
    D = np.stack([Dth.ravel(), Dr.ravel()], axis=1)
    D = D - np.outer(av, av.conj() @ D) / (av.conj() @ av)   # marginalise alpha
    return 2 * 10 ** (snr_db / 10) * np.real(D.conj().T @ D)


def bounds(J):
    Ci = np.linalg.inv(J)
    return np.rad2deg(np.sqrt(Ci[0, 0])), np.sqrt(Ci[1, 1])


# --------------------------------------------------------------- combiner designs

def V_random_ps(cfg, T, rng, F=None):
    """Frequency-flat unit-modulus phase-shifter combiners, then orthonormalised."""
    V = np.exp(2j * np.pi * rng.random((cfg.N, T))) / np.sqrt(cfg.N)
    return np.linalg.qr(V)[0]


def V_oracle(cfg, r, th, f, T=3):
    """Oracle: span{a_m, d_theta a_m, d_r a_m} at each frequency."""
    a, dth, dr = derivs(cfg, r, th, f)
    S = np.stack([a, dth, dr], axis=2)[:, :, :T]          # (F, N, T)
    return np.linalg.qr(S)[0]


def V_squint(cfg, T, f, region, rng=None):
    """T PS+TTD beam-squint combiners, each sweeping the sensing region.

    Combiner t focuses its lowest subcarrier at one end of the region and its
    highest at the other, with the endpoints staggered across t -- the same
    hardware Luo2024 and Zhang2026 already use, repurposed as a measurement
    basis instead of a power-peak readout.
    """
    th_lo, th_hi, r_lo, r_hi = region
    V = np.zeros((len(f), cfg.N, T), dtype=complex)
    for t in range(T):
        s = (t + 0.5) / T
        th_s, th_e = th_lo + s * (th_hi - th_lo), th_hi - s * (th_hi - th_lo)
        r_s, r_e = r_lo + s * (r_hi - r_lo), r_hi - s * (r_hi - r_lo)
        phi = cfg.f0 * dist_fresnel(r_s, th_s, cfg.x) / C
        tau = cfg.fM * dist_fresnel(r_e, th_e, cfg.x) / (cfg.W * C) - phi / cfg.W
        ft = (np.asarray(f) - cfg.f0)[:, None]
        V[:, :, t] = np.exp(-2j * np.pi * (phi[None, :] + ft * tau[None, :])) / np.sqrt(cfg.N)
    return np.linalg.qr(V)[0]


def main():
    cfg = Config()
    rng = np.random.default_rng(0)
    r, th, snr = 30.0, np.deg2rad(15.0), 0.0
    f = cfg.freqs[:: cfg.M // 64][:64]
    region = (cfg.theta_min, cfg.theta_max, cfg.r_min, cfg.r_max)

    b_full = bounds(fim(cfg, r, th, snr, f))
    print(f"full array (T = N = {cfg.N}):  RMSE_th >= {b_full[0]:.3e} deg   "
          f"RMSE_r >= {b_full[1]:.3e} m\n")

    print("=" * 74)
    print("Information retained vs number of analog probes T")
    print("=" * 74)
    print(f"{'T':>4} | {'oracle 3-D subspace':>28} | {'random PS':>18} | {'squint':>18}")
    print(f"{'':>4} | {'RMSE_r':>13}{'loss':>15} | {'RMSE_r':>10}{'loss':>8} | {'RMSE_r':>10}{'loss':>8}")
    print("-" * 74)
    for T in (1, 2, 3, 4, 8, 16, 32, 64):
        row = f"{T:>4} |"
        if T <= 3:
            bo = bounds(fim(cfg, r, th, snr, f, V_oracle(cfg, r, th, f, T)))
            row += f" {bo[1]:>13.3e}{bo[1]/b_full[1]:>13.2f}x |"
        else:
            row += f" {'(saturated at T=3)':>28} |"
        try:
            br = bounds(fim(cfg, r, th, snr, f, V_random_ps(cfg, T, rng)))
            row += f" {br[1]:>10.3e}{br[1]/b_full[1]:>7.1f}x |"
        except np.linalg.LinAlgError:
            row += f" {'singular':>10}{'':>8} |"
        try:
            bs = bounds(fim(cfg, r, th, snr, f, V_squint(cfg, T, f, region)))
            row += f" {bs[1]:>10.3e}{bs[1]/b_full[1]:>7.1f}x"
        except np.linalg.LinAlgError:
            row += f" {'singular':>10}"
        print(row)

    print("\n" + "=" * 74)
    print("Does the oracle T=3 result hold across the sensing region?")
    print("=" * 74)
    for rr in (15., 30., 50.):
        for td in (0., 30., 55.):
            t = np.deg2rad(td)
            bf = bounds(fim(cfg, rr, t, snr, f))
            bo = bounds(fim(cfg, rr, t, snr, f, V_oracle(cfg, rr, t, f, 3)))
            print(f"  (r={rr:4.0f} m, th={td:4.0f} deg)  full {bf[1]:.3e} m   "
                  f"oracle-3 {bo[1]:.3e} m   loss {bo[1]/bf[1]:.3f}x")


if __name__ == "__main__":
    main()


def V_matched(cfg, r0, th0, f, T=2):
    """Combiners matched to a COARSE estimate (r0, th0) rather than the truth."""
    return V_oracle(cfg, r0, th0, f, T)


def sensitivity():
    """How accurate must the coarse estimate be for matched combiners to work?

    Experiment B showed the beam-squint power peak nails the ANGLE (0.016 deg
    median) while the range it reports is meaningless (47 m median error).  So
    the question that decides the whole design is: are matched combiners
    sensitive to range error, or only to angle error?
    """
    cfg = Config()
    r, th, snr = 30.0, np.deg2rad(15.0), 0.0
    f = cfg.freqs[:: cfg.M // 64][:64]
    b_full = bounds(fim(cfg, r, th, snr, f))[1]

    print("\n" + "=" * 74)
    print("Sensitivity of T=2 matched combiners to coarse-estimate error")
    print("=" * 74)
    print(f"  reference: full-array bound RMSE_r = {b_full:.3e} m\n")

    print("  angle error in the combiner (range exact):")
    for dth in (0., 0.01, 0.1, 0.5, 1.0, 2.0, 5.0):
        V = V_matched(cfg, r, th + np.deg2rad(dth), f, 2)
        try:
            v = bounds(fim(cfg, r, th, snr, f, V))[1]
            print(f"    d_theta = {dth:5.2f} deg  ->  RMSE_r = {v:.3e} m   loss {v/b_full:6.2f}x")
        except np.linalg.LinAlgError:
            print(f"    d_theta = {dth:5.2f} deg  ->  singular")

    print("\n  range error in the combiner (angle exact):")
    for dr in (0., 1., 5., 15., 40., 70.):
        V = V_matched(cfg, r + dr, th, f, 2)
        try:
            v = bounds(fim(cfg, r, th, snr, f, V))[1]
            print(f"    d_r = {dr:5.0f} m      ->  RMSE_r = {v:.3e} m   loss {v/b_full:6.2f}x")
        except np.linalg.LinAlgError:
            print(f"    d_r = {dr:5.0f} m      ->  singular")

    print("\n  both wrong, at the accuracy Experiment B actually delivers")
    print("  (angle to 0.016 deg, range essentially unknown):")
    for dr in (0., 20., 47., 88.):
        V = V_matched(cfg, r + dr, th + np.deg2rad(0.016), f, 2)
        try:
            v = bounds(fim(cfg, r, th, snr, f, V))[1]
            print(f"    d_theta = 0.016 deg, d_r = {dr:3.0f} m  ->  RMSE_r = {v:.3e} m"
                  f"   loss {v/b_full:6.2f}x")
        except np.linalg.LinAlgError:
            print(f"    d_theta = 0.016 deg, d_r = {dr:3.0f} m  ->  singular")


if __name__ == "__main__":
    sensitivity()


def V_monopulse(cfg, r0, th0, f, delta, extra=None):
    """A constant-modulus monopulse pair, realisable with PS + TTD.

    Focusing every subcarrier on one point (r, th) needs
        phi_n = f0 * r_n / c        (phase shifters)
        t_n   = r_n / c             (true-time delays)
    which gives v_m[n] = exp(-j 2 pi f_m r_n / c) -- unit modulus, no amplitude
    taper.  Two such beams steered to th0 +- delta span, to first order,
        span{ a(th0), da/dtheta(th0) },
    the same subspace the oracle T=2 result uses, but without ever asking the
    hardware for the taper that da/dtheta implies.

    `extra` optionally appends further focus points as additional columns.
    """
    pts = [(r0, th0 - delta), (r0, th0 + delta)] + list(extra or [])
    cols = []
    for (rr, tt) in pts:
        d = dist_fresnel(rr, tt, cfg.x[None, :])
        cols.append(np.exp(-2j * np.pi * np.asarray(f)[:, None] * d / C) / np.sqrt(cfg.N))
    return np.linalg.qr(np.stack(cols, axis=2))[0]


def monopulse():
    cfg = Config()
    r, th, snr = 30.0, np.deg2rad(15.0), 0.0
    f = cfg.freqs[:: cfg.M // 64][:64]
    b_full = bounds(fim(cfg, r, th, snr, f))[1]
    bw = np.rad2deg(0.886 * cfg.lam / cfg.aperture)      # 3-dB beamwidth

    print("=" * 76)
    print("Constant-modulus monopulse pair (T=2) vs the oracle T=2 bound")
    print("=" * 76)
    print(f"  full-array bound RMSE_r = {b_full:.3e} m")
    print(f"  3-dB beamwidth          = {bw:.4f} deg\n")
    print(f"{'delta (deg)':>12}{'delta/BW':>10}{'RMSE_r (m)':>14}{'loss':>9}")
    best = (np.inf, None)
    for d_deg in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2):
        V = V_monopulse(cfg, r, th, f, np.deg2rad(d_deg))
        try:
            v = bounds(fim(cfg, r, th, snr, f, V))[1]
            print(f"{d_deg:>12.3f}{d_deg/bw:>10.2f}{v:>14.3e}{v/b_full:>8.2f}x")
            if v < best[0]:
                best = (v, d_deg)
        except np.linalg.LinAlgError:
            print(f"{d_deg:>12.3f}{d_deg/bw:>10.2f}{'singular':>14}{'':>9}")
    print(f"\n  best: delta = {best[1]:.3f} deg ({best[1]/bw:.2f} beamwidths)"
          f" -> {best[0]/b_full:.2f}x the full-array bound")

    d_opt = np.deg2rad(best[1])
    print("\n" + "=" * 76)
    print("Robustness of the monopulse pair to coarse-angle error")
    print("=" * 76)
    for dth in (0., 0.016, 0.05, 0.1, 0.2, 0.4):
        V = V_monopulse(cfg, r, th + np.deg2rad(dth), f, d_opt)
        v = bounds(fim(cfg, r, th, snr, f, V))[1]
        print(f"  coarse angle error {dth:5.3f} deg -> RMSE_r {v:.3e} m   loss {v/b_full:6.2f}x")

    print("\n" + "=" * 76)
    print("Robustness to coarse-range error (Experiment B leaves range unknown)")
    print("=" * 76)
    for dr in (0., 20., 47., 88.):
        V = V_monopulse(cfg, r + dr, th + np.deg2rad(0.016), f, d_opt)
        v = bounds(fim(cfg, r, th, snr, f, V))[1]
        print(f"  angle 0.016 deg + range {dr:3.0f} m -> RMSE_r {v:.3e} m   loss {v/b_full:6.2f}x")

    print("\n" + "=" * 76)
    print("Does a third constant-modulus beam help?")
    print("=" * 76)
    for name, extra in (("pair only (T=2)", None),
                        ("+ centre beam (T=3)", [(r, th)]),
                        ("+ range-offset beam (T=3)", [(r * 1.5, th)]),
                        ("+ both (T=4)", [(r, th), (r * 1.5, th)])):
        V = V_monopulse(cfg, r, th + np.deg2rad(0.016), f, d_opt, extra)
        v = bounds(fim(cfg, r, th, snr, f, V))[1]
        print(f"  {name:<28} RMSE_r {v:.3e} m   loss {v/b_full:6.2f}x")

    print("\n" + "=" * 76)
    print("Across the sensing region, at delta_opt and 0.016 deg coarse error")
    print("=" * 76)
    for rr in (15., 30., 50.):
        for td in (0., 30., 55.):
            t = np.deg2rad(td)
            bf = bounds(fim(cfg, rr, t, snr, f))[1]
            V = V_monopulse(cfg, 30.0, t + np.deg2rad(0.016), f, d_opt)   # range guess fixed at 30 m
            v = bounds(fim(cfg, rr, t, snr, f, V))[1]
            print(f"  (r={rr:4.0f} m, th={td:4.0f} deg)  full {bf:.3e}  monopulse {v:.3e}"
                  f"   loss {v/bf:6.2f}x")


if __name__ == "__main__":
    monopulse()
