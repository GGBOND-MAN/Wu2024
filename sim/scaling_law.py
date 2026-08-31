"""
The scaling laws behind the attack: where does range information come from?

Two mutually exclusive sources of range information exist in this system.

  CURVATURE  (what Luo2024, Zhang2026 and arXiv:2603.16390 all use).
    Range enters only through the quadratic Fresnel term x^2 cos^2(theta)/(2r).
    With the complex amplitude alpha unknown, the constant part of dPhi/dr is
    absorbed by arg(alpha), leaving only the deviation of x^2 about its mean:

        dPhi_n/dr - mean  =  (2 pi f_c / c) (cos^2 th / 2r^2) (x_n^2 - <x^2>)

    For x uniform on [-D/2, D/2],  sum_n (x^2 - <x^2>)^2 = N D^4 / 180, so

        J_r^curv  =  2 SNR (2 pi f_c/c)^2 cos^4(th) N D^4 / (720 r^4)

  DELAY  (what nobody in this lineage uses).
    Phi(n,m) = -(2 pi f_m/c) r_n is exactly linear in m.  A single shared alpha
    absorbs only the mean over f, leaving the deviation f_m - f_c:

        J_r^delay =  2 SNR N M (2 pi/c)^2 W^2 / 12

Consequences, and the reason the attack is structural rather than incidental:

    RMSE_r^curv   ~  lambda r^2 / (N^2.5 d^2 cos^2 th)      grows as r^2
    RMSE_r^delay  ~  c / (W sqrt(N M))                      independent of r, th

    RMSE_curv / RMSE_delay  =  sqrt(60 M) (W/f_c) r^2 / (D^2 cos^2 th)

Zhang2026 fuses M subcarriers incoherently (Eq. 44), which buys sqrt(M) of
averaging but no delay term, so it sits a factor sqrt(M) below the curvature
bound and still sqrt(60)(W/f_c) r^2/(D^2 cos^2 th) above the delay bound.

This script checks every exponent above against the numerically evaluated FIM.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from nf_model import Config, C
from estimators import crlb, crlb_wideband

SNR = 0.0


def rmse_curv(cfg, r, th):
    """Narrowband, amplitude marginalised: range from wavefront curvature only."""
    return crlb(cfg, r, th, SNR, cfg.fc, marginalize=True)[1]


def rmse_delay(cfg, r, th, n_sub=256):
    """Coherent wideband: the delay term dominates."""
    return crlb_wideband(cfg, r, th, SNR, n_sub=n_sub, coherent=True)[1]


def rmse_incoh(cfg, r, th, n_sub=256):
    """Per-subcarrier amplitudes: the class Zhang2026's Eq. (44) belongs to."""
    return crlb_wideband(cfg, r, th, SNR, n_sub=n_sub, coherent=False)[1]


def slope(xs, ys):
    return np.polyfit(np.log(xs), np.log(ys), 1)[0]


def predicted_ratio(cfg, r, th):
    D = cfg.aperture
    return np.sqrt(60 * cfg.M) * (cfg.W / cfg.fc) * r ** 2 / (D ** 2 * np.cos(th) ** 2)


def main():
    base = Config()
    r0, th0 = 30.0, np.deg2rad(15.0)
    print(f"reference: N={base.N}, M={base.M}, W={base.W/1e9:.0f} GHz, "
          f"fc={base.fc/1e9:.0f} GHz, D={base.aperture:.4f} m, SNR={SNR:.0f} dB/element\n")

    print("=" * 78)
    print("1. EXPONENTS  (numerically fitted vs analytically predicted)")
    print("=" * 78)
    rows = []

    Ns = np.array([64, 128, 256, 512, 1024])
    rows.append(("RMSE_curv  vs N", slope(Ns, [rmse_curv(Config(N=n), r0, th0) for n in Ns]), -2.5))
    rows.append(("RMSE_delay vs N", slope(Ns, [rmse_delay(Config(N=n), r0, th0) for n in Ns]), -0.5))

    rs = np.array([10., 20., 30., 50., 80.])
    rows.append(("RMSE_curv  vs r", slope(rs, [rmse_curv(base, r, th0) for r in rs]), +2.0))
    rows.append(("RMSE_delay vs r", slope(rs, [rmse_delay(base, r, th0) for r in rs]), 0.0))

    Ws = np.array([0.5e9, 1e9, 2e9, 3e9, 6e9])
    rows.append(("RMSE_delay vs W", slope(Ws, [rmse_delay(Config(W=w), r0, th0) for w in Ws]), -1.0))

    Ms = np.array([256, 512, 1024, 2048])
    rows.append(("RMSE_delay vs M", slope(Ms, [rmse_delay(Config(M=m), r0, th0, n_sub=128) for m in Ms]), -0.5))
    rows.append(("RMSE_incoh vs M", slope(Ms, [rmse_incoh(Config(M=m), r0, th0, n_sub=128) for m in Ms]), -0.5))

    print(f"{'quantity':<20}{'fitted':>10}{'predicted':>12}   {'':<6}")
    for name, fit, pred in rows:
        ok = "OK" if abs(fit - pred) < 0.06 else "MISMATCH"
        print(f"{name:<20}{fit:>10.3f}{pred:>12.1f}   {ok}")

    ths = np.deg2rad(np.array([5., 15., 30., 45., 55.]))
    fit = slope(1 / np.cos(ths) ** 2, [rmse_curv(base, r0, t) for t in ths])
    print(f"{'RMSE_curv vs 1/cos^2':<20}{fit:>10.3f}{1.0:>12.1f}   "
          f"{'OK' if abs(fit - 1) < 0.06 else 'MISMATCH'}")

    print("\n" + "=" * 78)
    print("2. THE CLOSED-FORM RATIO  vs numerically evaluated bounds")
    print("=" * 78)
    print(f"{'(r, theta)':<16}{'RMSE_curv':>12}{'RMSE_delay':>12}{'measured':>11}"
          f"{'predicted':>11}{'err':>7}")
    for r in (10., 30., 50., 80.):
        for th_deg in (15., 45.):
            th = np.deg2rad(th_deg)
            a, b = rmse_curv(base, r, th), rmse_delay(base, r, th)
            meas, pred = a / b, predicted_ratio(base, r, th)
            print(f"({r:.0f} m, {th_deg:.0f} deg){'':<3}{a:>12.3e}{b:>12.3e}"
                  f"{meas:>11.0f}{pred:>11.0f}{100*abs(meas-pred)/pred:>6.1f}%")

    print("\n" + "=" * 78)
    print("3. CROSSOVER: beyond what range is curvature ranging already hopeless?")
    print("=" * 78)
    for th_deg in (0., 15., 30., 45., 60.):
        th = np.deg2rad(th_deg)
        r_x = base.aperture * np.cos(th) * (base.fc / base.W / np.sqrt(60 * base.M)) ** 0.5
        print(f"  theta = {th_deg:4.0f} deg  ->  curvature competitive only within "
              f"r < {r_x*100:.1f} cm")
    print(f"\n  The declared sensing region of Zhang2026 starts at {base.r_min} m, "
          f"which is\n  {base.r_min/ (base.aperture*(base.fc/base.W/np.sqrt(60*base.M))**0.5):.0f}x "
          f"beyond that crossover.")


if __name__ == "__main__":
    main()
