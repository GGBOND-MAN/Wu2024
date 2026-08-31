"""
Experiment E -- the proposed coherent space-frequency estimator, head to head
with the reproduced Zhang2026 pipeline under identical conditions.

The user position and SNR grid are exactly those of Experiment C, and
exp_c.npz is loaded for the Zhang2026 curve, so the comparison is like for like.

Measurement fairness: both methods consume the SAME observation.  Synthesising
the array snapshot costs T = N sequential probing intervals either way, and one
OFDM symbol delivers all M subcarriers at once -- so the frequency axis is free.
Zhang2026 uses |S| = 5 of those subcarriers and fuses them by a geometric mean
(incoherent); the proposed method uses all M coherently.  The gap below is
therefore about what is done with the data, not about how much data is taken.

Step 1-2 = delay-domain ranging + exact-geometry multilateration.
Step 1-3 = the above plus a local ML polish on the exact manifold.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nf_model import Config
from estimators import crlb_wideband
from proposed_sfj import estimate, ml_refine, make_wideband_data

FIG = os.path.join(os.path.dirname(__file__), "figs")
SNRS = [-10, -5, 0, 5, 10, 15, 20]
N_TRIAL = 40
CKPT = os.path.join(os.path.dirname(__file__), "figs", "exp_e_ckpt.json")
# Background processes do not survive a container suspend, only the filesystem
# does.  So run in the foreground in slices: each invocation works through as
# many SNRs as fit in TIME_BUDGET, checkpoints, and exits.  Re-invoke until done.
TIME_BUDGET = float(os.environ.get("EXP_E_BUDGET", 100))
TH_TRUE, R_TRUE = np.deg2rad(-46.634), 29.974      # identical to Experiment C


def run():
    cfg = Config()
    rng = np.random.default_rng(2026)
    f = cfg.freqs
    print(f"[E] user at ({np.rad2deg(TH_TRUE):.3f} deg, {R_TRUE:.3f} m), "
          f"N={cfg.N}, M={cfg.M}, W={cfg.W/1e9:.0f} GHz")
    print(f"    unambiguous range c/(2 df) = {3e8 / (2 * (f[1] - f[0])):.1f} m\n")

    # Checkpoint per SNR so the run survives a container restart.
    import json
    done = json.load(open(CKPT)) if os.path.exists(CKPT) else {}
    t_start = time.perf_counter()
    for snr in SNRS:
        if time.perf_counter() - t_start > TIME_BUDGET:
            print(f"  [budget reached, {len(done)}/{len(SNRS)} SNRs done -- re-run to continue]")
            return
        if str(snr) in done:
            print(f"  SNR={snr:+3d} dB | cached", flush=True)
            continue
        acc = {k: ([], []) for k in ("s12", "s13", "mix")}
        t0 = time.perf_counter()
        for _ in range(N_TRIAL):
            Y = make_wideband_data(cfg, R_TRUE, TH_TRUE, f, snr, rng, "exact")
            th2, r2 = estimate(Y, f, cfg.x)
            th3, r3 = ml_refine(Y, f, cfg.x, th2, r2)   # reuse step 1-2, don't redo it
            for key, (th, r) in (("s12", (th2, r2)), ("s13", (th3, r3)),
                                 ("mix", (th3, r2))):   # angle from step 3, range from step 1-2
                acc[key][0].append(th - TH_TRUE)
                acc[key][1].append(r - R_TRUE)
        rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
        done[str(snr)] = {k: [np.rad2deg(rms(acc[k][0])), rms(acc[k][1])] for k in acc}
        json.dump(done, open(CKPT, "w"))
        d = done[str(snr)]
        print(f"  SNR={snr:+3d} dB | s12 {d['s12'][0]:.3e} deg / {d['s12'][1]:.3e} m"
              f" | s13 {d['s13'][0]:.3e} deg / {d['s13'][1]:.3e} m"
              f" | mix {d['mix'][0]:.3e} deg / {d['mix'][1]:.3e} m"
              f" | {(time.perf_counter()-t0)/N_TRIAL*1e3:5.0f} ms/trial", flush=True)

    res = {k: ([done[str(s)][k][0] for s in SNRS], [done[str(s)][k][1] for s in SNRS])
           for k in ("s12", "s13", "mix")}

    bt, br = zip(*[crlb_wideband(cfg, R_TRUE, TH_TRUE, s, n_sub=256, coherent=True)
                   for s in SNRS])

    zc = None
    if os.path.exists(os.path.join(FIG, "exp_c.npz")):
        zc = np.load(os.path.join(FIG, "exp_c.npz"))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for a, idx, lbl, unit in ((ax[0], 0, "$\\theta$", "deg"), (ax[1], 1, "$r$", "m")):
        a.semilogy(SNRS, res["s12"][idx], "o--", label="proposed, step 1-2 (delay + geometry)")
        a.semilogy(SNRS, res["s13"][idx], "s-", label="proposed, step 1-3 (+ ML polish)")
        a.semilogy(SNRS, res["mix"][idx], "d-", label="proposed, combined (angle from 3, range from 1-2)")
        a.semilogy(SNRS, (bt if idx == 0 else br), "k:", label="coherent wideband CRLB")
        a.set(xlabel="per-element SNR (dB)", ylabel=f"RMSE {lbl} ({unit})")
        a.grid(True, which="both", alpha=.3)
    if zc is not None:
        ax[1].semilogy(zc["snr"], zc["rmse_all"], "^-", color="tab:red",
                       label="reproduced Zhang2026 (stage I + MUSIC)")
        ax[1].semilogy(zc["snr"], zc["coarse_r"], "v-.", color="tab:orange",
                       label="reproduced Zhang2026, stage I alone")
    ax[1].axhline(1e-3, color="crimson", ls="-.", lw=1, label="Zhang2026 claim  0.001 m")
    ax[0].legend(fontsize=7.5); ax[1].legend(fontsize=7)
    ax[0].set_title("(a) Angle"); ax[1].set_title("(b) Range")
    fig.suptitle("Experiment E: coherent space-frequency estimation vs the reproduced "
                 "Zhang2026 pipeline, same user, same SNR, same measurements", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "exp_e_proposed.png"), dpi=140)
    np.savez(os.path.join(FIG, "exp_e.npz"), snr=SNRS, crlb_th=bt, crlb_r=br,
             **{f"{k}_{n}": v for k in res for n, v in zip(("th", "r"), res[k])})

    print("\n  efficiency (RMSE / CRLB):")
    for i, s in enumerate(SNRS):
        print(f"    SNR={s:+3d} dB | angle: step1-2 {res['s12'][0][i]/np.rad2deg(bt[i]):7.1f}x"
              f" -> step1-3 {res['s13'][0][i]/np.rad2deg(bt[i]):6.2f}x"
              f" | range: step1-3 {res['s13'][1][i]/br[i]:.2f}x")


if __name__ == "__main__":
    run()
