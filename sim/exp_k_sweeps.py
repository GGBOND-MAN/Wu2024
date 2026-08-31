"""
Experiments K1 and K2 -- do the scaling laws survive end to end?

scaling_law.py verified the exponents on the CRLB.  These run the actual
estimator while sweeping the two system parameters the laws depend on:

    K1  vs N :  delay ranging ~ N^-0.5      (curvature would be N^-2.5)
    K2  vs W :  delay ranging ~ W^-1        (curvature is blind to bandwidth)

Two things have to scale with the sweep or the comparison is not like for like:
  * the monopulse separation delta is 0.25 BEAMWIDTHS, and the beamwidth is
    lambda/D, so it shrinks as N grows;
  * the stage-A angle error is likewise held at a fixed fraction of a beamwidth
    (0.153, which is the 0.061 deg measured at N=256 in Experiment C).

Bandwidth is capped at 6 GHz: the unambiguous range is c*M/(2W), which at
M = 2048 is 51.2 m there and would fall below the 50 m sensing region beyond it.

SCALE: EXP_K_TRIALS (paper: 500+), EXP_K_MLDECIM (1 = full band).
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nf_model import Config
from acquisition import fim, bounds
import exp_f_three_probe as F

FIG = os.path.join(os.path.dirname(__file__), "figs")
CKPT = os.path.join(FIG, "exp_k_ckpt.json")
N_TRIAL = int(os.environ.get("EXP_K_TRIALS", 15))
BUDGET = float(os.environ.get("EXP_K_BUDGET", 95))
F.ML_DECIM = int(os.environ.get("EXP_K_MLDECIM", 4))
SNR, R_TRUE, TH_TRUE = 0.0, 30.0, np.deg2rad(15.0)
COARSE_FRAC = 0.153                      # stage-A angle error, in beamwidths
N_LIST = [64, 128, 256, 512, 1024]
W_LIST = [0.5e9, 1e9, 2e9, 3e9, 6e9]


def one_point(cfg, seed):
    """RMSE over N_TRIAL runs of the full three-probe scheme at this config."""
    f = cfg.freqs[::F.SUB_DECIM]
    bw = 0.886 * cfg.lam / cfg.aperture
    delta, ce = 0.25 * bw, COARSE_FRAC * bw
    rng = np.random.default_rng(seed)
    et, er = [], []
    for _ in range(N_TRIAL):
        th0 = TH_TRUE + ce * rng.standard_normal()
        V = F.beams(cfg, F.R_GUESS, th0, delta, f)
        z = F.simulate(cfg, R_TRUE, TH_TRUE, V, f, SNR, rng)
        th, r = F.estimate(cfg, z, V, f, th0, F.R_GUESS, th_halfwidth=3 * ce)
        et.append(th - TH_TRUE); er.append(r - R_TRUE)
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    V0 = np.linalg.qr(F.beams(cfg, F.R_GUESS, TH_TRUE, delta, f))[0]
    bt, br = bounds(fim(cfg, R_TRUE, TH_TRUE, SNR, f, V0))
    return [np.rad2deg(rms(et)), rms(er), bt, br]


def run():
    done = json.load(open(CKPT)) if os.path.exists(CKPT) else {}
    t0 = time.perf_counter()
    print(f"[K] SNR={SNR:.0f} dB, {N_TRIAL} trials/point, ML_DECIM={F.ML_DECIM}\n")

    jobs = ([(f"N{n}", Config(N=n)) for n in N_LIST] +
            [(f"W{int(w/1e6)}", Config(W=w)) for w in W_LIST])
    for key, cfg in jobs:
        if key in done:
            continue
        if time.perf_counter() - t0 > BUDGET:
            print(f"  [budget reached, {len(done)}/{len(jobs)} done -- re-run]")
            return
        done[key] = one_point(cfg, abs(hash(key)) % 2**31)
        json.dump(done, open(CKPT, "w"))
        d = done[key]
        print(f"  {key:<8} RMSE_th {d[0]:.3e} deg  RMSE_r {d[1]:.3e} m   "
              f"(bounds {d[2]:.2e}, {d[3]:.2e})", flush=True)

    gN = lambda i: [done[f"N{n}"][i] for n in N_LIST]
    gW = lambda i: [done[f"W{int(w/1e6)}"][i] for w in W_LIST]
    sl = lambda x, y: np.polyfit(np.log(x), np.log(y), 1)[0]

    print("\n" + "=" * 72)
    print(f"  RMSE_r vs N : fitted {sl(N_LIST, gN(1)):+.2f}   predicted -0.50 "
          f"(curvature would be -2.50)")
    print(f"  RMSE_r vs W : fitted {sl(W_LIST, gW(1)):+.2f}   predicted -1.00 "
          f"(curvature is blind to W)")
    print(f"  bound   vs N : fitted {sl(N_LIST, gN(3)):+.2f}")
    print(f"  bound   vs W : fitted {sl(W_LIST, gW(3)):+.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax[0].loglog(N_LIST, gN(1), "o-", label="proposed, measured")
    ax[0].loglog(N_LIST, gN(3), "k:", label="CRLB through the combiners")
    ref = gN(1)[0] * (np.array(N_LIST) / N_LIST[0]) ** -0.5
    ax[0].loglog(N_LIST, ref, "--", color="gray", label="$N^{-1/2}$ reference")
    ax[0].set(xlabel="number of antennas $N$", ylabel="RMSE $r$ (m)",
              title="(a) K1: vs array size")
    ax[1].loglog(np.array(W_LIST) / 1e9, gW(1), "s-", label="proposed, measured")
    ax[1].loglog(np.array(W_LIST) / 1e9, gW(3), "k:", label="CRLB through the combiners")
    ref = gW(1)[0] * (np.array(W_LIST) / W_LIST[0]) ** -1.0
    ax[1].loglog(np.array(W_LIST) / 1e9, ref, "--", color="gray", label="$W^{-1}$ reference")
    ax[1].set(xlabel="bandwidth $W$ (GHz)", ylabel="RMSE $r$ (m)",
              title="(b) K2: vs bandwidth")
    for a in ax:
        a.grid(True, which="both", alpha=.3); a.legend(fontsize=8)
    fig.suptitle("Experiments K1/K2: the scaling laws, end to end", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "exp_k_sweeps.png"), dpi=140)
    np.savez(os.path.join(FIG, "exp_k.npz"), N=N_LIST, W=W_LIST,
             N_th=gN(0), N_r=gN(1), N_bth=gN(2), N_br=gN(3),
             W_th=gW(0), W_r=gW(1), W_bth=gW(2), W_br=gW(3))


if __name__ == "__main__":
    run()
