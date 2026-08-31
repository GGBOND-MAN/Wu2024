"""
Experiment C -- the missing threshold effect.

Zhang2026 sizes its stage-II search window (1 deg, 1 m) from the stage-I RMSE,
which is itself about 1 deg / 1 m at low SNR.  An RMSE of 1 m means a large
fraction of realisations land further away than 1 m, so the truth falls outside
the window and stage II returns a value pinned to the window edge.  Every real
estimator built this way shows a threshold SNR below which outliers dominate;
Fig. 10 shows smooth near-CRLB curves down to -10 dB and no threshold at all.

To give the paper the best possible case the user is placed exactly ON the
designed trajectory, so the structural failure of Experiment B is excluded and
only noise is at play.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nf_model import Config, trajectory, ttd_ps_weights, coarse_observation
from estimators import coarse_estimate, music_refine, make_snapshot, crlb_wideband

FIG = os.path.join(os.path.dirname(__file__), "figs")
TS, RS, TE, RE = np.deg2rad(-60), 15.0, np.deg2rad(60), 50.0
SNRS = [-10, -5, 0, 5, 10, 15, 20]
N_TRIAL, N_SUB = 120, 5
DTH_WIN, DR_WIN = 1.0, 1.0


def run():
    cfg = Config()
    rng = np.random.default_rng(7)
    th_traj, r_traj = trajectory(cfg, TS, RS, TE, RE)
    phi, t = ttd_ps_weights(cfg, TS, RS, TE, RE)

    # Put the user exactly on the trajectory -- the best case for the paper --
    # and pick a subcarrier whose focal range actually lies in the sensing region
    # (the trajectory bulges out to 103 m, see Experiment B).
    ok_m = np.where((r_traj >= cfg.r_min) & (r_traj <= cfg.r_max))[0]
    m_star = int(ok_m[np.argmin(np.abs(r_traj[ok_m] - 30.0))])
    TH_TRUE, R_TRUE = th_traj[m_star], r_traj[m_star]
    print(f"[C] user placed on the trajectory at "
          f"({np.rad2deg(TH_TRUE):.3f} deg, {R_TRUE:.3f} m)")

    z_clean = coarse_observation(cfg, R_TRUE, TH_TRUE, phi, t)
    S = cfg.freqs[:: cfg.M // N_SUB][:N_SUB]

    cap, rmse_all, rmse_in, c_th, c_r = [], [], [], [], []
    for snr in SNRS:
        ok, er_all, er_in, eth_c, er_c = 0, [], [], [], []
        for _ in range(N_TRIAL):
            th0, r0, _ = coarse_estimate(z_clean, th_traj, r_traj, snr, rng)
            eth_c.append(np.rad2deg(th0 - TH_TRUE)); er_c.append(r0 - R_TRUE)
            inside = (abs(np.rad2deg(th0 - TH_TRUE)) <= DTH_WIN
                      and abs(r0 - R_TRUE) <= DR_WIN)
            ok += inside
            ys = [make_snapshot(cfg, R_TRUE, TH_TRUE, f, snr, rng, "exact") for f in S]
            _, r_hat = music_refine(ys, cfg, th0, r0, S, model="fresnel")
            er_all.append(r_hat - R_TRUE)
            if inside:
                er_in.append(r_hat - R_TRUE)
        cap.append(100 * ok / N_TRIAL)
        rmse_all.append(np.sqrt(np.mean(np.square(er_all))))
        rmse_in.append(np.sqrt(np.mean(np.square(er_in))) if er_in else np.nan)
        c_th.append(np.sqrt(np.mean(np.square(eth_c))))
        c_r.append(np.sqrt(np.mean(np.square(er_c))))
        print(f"  SNR={snr:+3d} dB | stage-I RMSE {c_th[-1]:6.3f} deg / {c_r[-1]:6.3f} m"
              f" | window capture {cap[-1]:5.1f}%"
              f" | final RMSE_r  all {rmse_all[-1]:.3e} m  captured-only {rmse_in[-1]:.3e} m",
              flush=True)

    bound = [crlb_wideband(cfg, R_TRUE, TH_TRUE, s, n_sub=N_SUB, coherent=False)[1]
             for s in SNRS]

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax[0].plot(SNRS, cap, "o-", color="tab:red")
    ax[0].set(xlabel="SNR (dB)", ylabel="% of trials with the truth inside the window",
              title="(a) Stage-II window capture probability", ylim=(-3, 103))
    ax[0].axhline(100, color="k", ls=":")
    ax[1].semilogy(SNRS, rmse_all, "o-", label="all trials (outliers included)")
    ax[1].semilogy(SNRS, rmse_in, "s--", label="captured trials only (what a smooth curve hides)")
    ax[1].semilogy(SNRS, bound, "k:", label="correct incoherent wideband CRLB")
    ax[1].axhline(1e-3, color="crimson", ls="-.", label="Zhang2026 claim  0.001 m")
    ax[1].set(xlabel="SNR (dB)", ylabel="RMSE $r$ (m)", title="(b) Range RMSE")
    for a in ax:
        a.grid(True, which="both", alpha=.3)
    ax[1].legend(fontsize=7.5)
    fig.suptitle("Experiment C: the threshold effect absent from Zhang2026 Fig. 10 "
                 "(user placed exactly on the trajectory)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "exp_c_threshold.png"), dpi=140)
    np.savez(os.path.join(FIG, "exp_c.npz"), snr=SNRS, cap=cap, rmse_all=rmse_all,
             rmse_in=rmse_in, bound=bound, coarse_th=c_th, coarse_r=c_r)


if __name__ == "__main__":
    run()
