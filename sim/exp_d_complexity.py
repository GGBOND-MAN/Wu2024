"""
Experiment D -- the claimed accuracy and the claimed latency cannot both hold.

Zhang2026 reports RMSE below 0.001 deg and 0.001 m from a 2-D grid search over
a 1 deg / 1 m window (Sec. IV-C-1), and a total latency of 3.90 ms (Fig. 9).
We measure what that grid actually costs, using the CHEAPEST correct
implementation -- for a one-dimensional signal subspace the MUSIC denominator
collapses to ||a||^2 - |u1^H a|^2, which is O(Ms) per grid point instead of the
O(Ms^2) quadratic form the paper's own Table I bills for.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from nf_model import Config
from estimators import _music_den

CLAIMED_LATENCY_MS = 3.90
WIN_TH, WIN_R = 2.0, 2.0        # full width of the +-1 deg / +-1 m window
TARGET_TH, TARGET_R = 1e-3, 1e-3


def run():
    cfg = Config()
    n_th, n_r = int(WIN_TH / TARGET_TH), int(WIN_R / TARGET_R)
    n_grid = n_th * n_r
    print(f"[D] to resolve {TARGET_TH} deg and {TARGET_R} m over a "
          f"{WIN_TH} deg x {WIN_R} m window:")
    print(f"    grid = {n_th} x {n_r} = {n_grid:,} points\n")

    # Measured throughput of the efficient spectrum evaluation.
    rng = np.random.default_rng(0)
    u1 = rng.standard_normal(cfg.Ms) + 1j * rng.standard_normal(cfg.Ms)
    u1 /= np.linalg.norm(u1)
    n_probe = 200
    g_th = np.linspace(0, np.deg2rad(1), n_probe)
    g_r = np.linspace(29, 31, n_probe)
    _music_den(g_th[:4], g_r[:4], cfg, cfg.fc, u1)          # warm up
    t0 = time.perf_counter()
    _music_den(g_th, g_r, cfg, cfg.fc, u1)
    dt = time.perf_counter() - t0
    pts_per_s = n_probe ** 2 / dt
    print(f"    measured throughput (vectorised numpy, this machine): "
          f"{pts_per_s:,.0f} grid points/s")

    for n_sub in (1, 5, 11):
        secs = n_grid * n_sub / pts_per_s
        print(f"    |S| = {n_sub:2d} subcarriers -> {secs:9.2f} s "
              f"= {secs * 1e3 / CLAIMED_LATENCY_MS:12,.0f}x the claimed {CLAIMED_LATENCY_MS} ms")

    print("\n    FLOP accounting, independent of this machine:")
    for name, per_pt in (("efficient  ||a||^2 - |u1^H a|^2   (O(Ms))", 8 * cfg.Ms),
                         ("paper Table I  a^H Un Un^H a      (O(Ms^2))", 8 * cfg.Ms ** 2)):
        for n_sub in (5,):
            fl = n_grid * n_sub * per_pt
            print(f"      {name}: {fl:.2e} flops for |S|={n_sub}")
            for rate, hw in ((1e11, "100 GFLOPS"), (1e12, "1 TFLOPS")):
                print(f"          at {hw:>10s}: {fl / rate * 1e3:12,.1f} ms "
                      f"({fl / rate * 1e3 / CLAIMED_LATENCY_MS:,.0f}x claimed)")

    print("\n    Inverting the question -- what accuracy fits in 3.90 ms?")
    budget = CLAIMED_LATENCY_MS * 1e-3 * pts_per_s / 5
    step_th = WIN_TH / np.sqrt(budget)
    print(f"      {budget:,.0f} grid points affordable  ->  "
          f"{np.sqrt(budget):.0f} x {np.sqrt(budget):.0f} grid  ->  "
          f"step {step_th:.4f} deg / {step_th:.4f} m")
    print(f"      i.e. {step_th / TARGET_TH:.0f}x coarser than the claimed accuracy.")

    print("\n    Sensing overhead hidden in Eq. (9)-(11):")
    print(f"      a single RF chain with V = I_N needs T = N = {cfg.N} sequential probing")
    print(f"      intervals to synthesise one array snapshot, versus 2 beam sweeps for")
    print(f"      CBS-Low [20] -- a {cfg.N // 2}x sensing overhead reported as "
          f"'almost the same' latency.")
    print(f"      Turning on one element at a time also costs 10*log10(N) = "
          f"{10 * np.log10(cfg.N):.1f} dB of array gain per measurement.")


if __name__ == "__main__":
    run()
