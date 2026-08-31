"""
Experiment B -- the coarse stage is under-determined, and the range trajectory
is not even monotone.

Zhang2026 Sec. III-C claims:

    "For this mapping to be unambiguous, the relationship between the subcarrier
     index m and the spatial coordinates must be monotonic.  This is guaranteed
     by the structure of (19) and (20), ensuring that each subcarrier corresponds
     to a unique spatial point."

B1 tests that claim directly on the paper's own Fig. 5 configuration.
B2 maps the noiseless stage-I error over the whole declared sensing region and
   compares it against the stage-II search window (1 deg, 1 m).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nf_model import Config, trajectory, ttd_ps_weights, coarse_observation
from estimators import coarse_estimate

FIG = os.path.join(os.path.dirname(__file__), "figs")
TS, RS, TE, RE = np.deg2rad(-60), 15.0, np.deg2rad(60), 50.0   # Zhang2026 Fig. 5
DTH_WIN, DR_WIN = 1.0, 1.0                                     # Zhang2026 Sec. IV-C-1


def b1_monotonicity(cfg):
    """Test the paper's monotonicity claim on its own Fig. 5 configuration."""
    th, r = trajectory(cfg, TS, RS, TE, RE)
    turn = int(np.argmax(r))
    print(f"[B1] trajectory ({np.rad2deg(TS):+.0f} deg, {RS} m) -> "
          f"({np.rad2deg(TE):+.0f} deg, {RE} m),  M = {cfg.M} subcarriers")
    print(f"     theta_m monotone : {bool(np.all(np.diff(th) > 0))}")
    print(f"     r_m     monotone : "
          f"{bool(np.all(np.diff(r) > 0) or np.all(np.diff(r) < 0))}")
    print(f"     r_m is a single-peaked arc: rises 15.00 -> {r.max():.2f} m at "
          f"m={turn} (theta={np.rad2deg(th[turn]):.1f} deg), then falls to {r[-1]:.2f} m")
    print(f"     it overshoots the declared sensing region [{cfg.r_min}, {cfg.r_max}] m "
          f"by {r.max() - cfg.r_max:.1f} m ({r.max() / cfg.r_max:.1f}x)")

    # Two-to-one: any range on the descending branch is reached twice.
    for probe in (40.0, 60.0, 80.0, 95.0):
        x = int(np.sum(np.sign(r[:-1] - probe) != np.sign(r[1:] - probe)))
        print(f"     r_m = {probe:5.1f} m is attained by {x} distinct subcarriers")

    # Mechanism: Eq. (20) carries a 1/cos^2(theta_m) factor.
    print(f"     mechanism: Eq. (20) gives r_m proportional to cos^2(theta_m); a span that"
          f" crosses broadside\n                inflates r_m by up to "
          f"1/cos^2({np.rad2deg(abs(TS)):.0f} deg) = {1/np.cos(TS)**2:.1f}x at theta=0.")
    for lo, hi in ((-60, 60), (-30, 30), (5, 60), (20, 60)):
        _, rr = trajectory(cfg, np.deg2rad(lo), RS, np.deg2rad(hi), RE)
        mono = bool(np.all(np.diff(rr) > 0) or np.all(np.diff(rr) < 0))
        print(f"       span [{lo:+3d}, {hi:+3d}] deg -> r_m monotone: {str(mono):5s}"
              f"  peak r = {rr.max():6.2f} m  {'(crosses broadside)' if lo*hi < 0 else ''}")
    return th, r


def b2_coverage(cfg, th_traj, r_traj):
    """Noiseless stage-I error across the sensing region."""
    phi, t = ttd_ps_weights(cfg, TS, RS, TE, RE)
    th_g = np.deg2rad(np.linspace(-55, 55, 45))
    r_g = np.linspace(cfg.r_min, cfg.r_max, 40)
    err_th = np.zeros((len(th_g), len(r_g)))
    err_r = np.zeros_like(err_th)
    for i, th in enumerate(th_g):
        for j, r in enumerate(r_g):
            z = coarse_observation(cfg, r, th, phi, t)
            th0, r0, _ = coarse_estimate(z, th_traj, r_traj)
            err_th[i, j] = abs(np.rad2deg(th0 - th))
            err_r[i, j] = abs(r0 - r)
    inside = (err_th <= DTH_WIN) & (err_r <= DR_WIN)
    print(f"\n[B2] noiseless stage-I error over the declared sensing region")
    print(f"     median |dtheta| = {np.median(err_th):6.3f} deg   "
          f"median |dr| = {np.median(err_r):7.3f} m")
    print(f"     90th pct |dtheta| = {np.percentile(err_th, 90):6.3f} deg   "
          f"90th pct |dr| = {np.percentile(err_r, 90):7.3f} m")
    print(f"     worst |dr| = {err_r.max():.1f} m")
    print(f"     fraction of the region where the truth falls inside the "
          f"stage-II window: {100 * inside.mean():.1f}%")

    ext = [cfg.r_min, cfg.r_max, -55, 55]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for a, dat, ttl, unit in ((ax[0], err_th, "(a) stage-I angle error", "deg"),
                              (ax[1], err_r, "(b) stage-I range error", "m")):
        im = a.imshow(dat, origin="lower", aspect="auto", extent=ext,
                      cmap="magma", norm=matplotlib.colors.LogNorm())
        a.contour(*np.meshgrid(np.linspace(*ext[:2], dat.shape[1]),
                               np.linspace(*ext[2:], dat.shape[0])),
                  dat, levels=[1.0], colors="cyan", linewidths=2)
        plt.colorbar(im, ax=a, label=unit)
        a.plot(r_traj[(r_traj >= cfg.r_min) & (r_traj <= cfg.r_max)],
               np.rad2deg(th_traj[(r_traj >= cfg.r_min) & (r_traj <= cfg.r_max)]),
               "w.", ms=1, alpha=.5)
        a.set(xlabel="true range r (m)", ylabel="true angle $\\theta$ (deg)", title=ttl)
    ax[2].plot(np.rad2deg(th_traj), r_traj, lw=1.2)
    ax[2].axhspan(cfg.r_min, cfg.r_max, color="tab:green", alpha=.15,
                  label="declared sensing region")
    ax[2].set(xlabel="trajectory angle $\\theta_m$ (deg)", ylabel="trajectory range $r_m$ (m)",
              title="(c) designed trajectory: range is not monotone")
    ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)
    fig.suptitle("Experiment B: a one-dimensional squint trajectory cannot cover a "
                 "two-dimensional region.  Cyan contour = boundary of the 1 deg / 1 m "
                 "stage-II search window; white dots = the trajectory itself.", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "exp_b_trajectory.png"), dpi=140)
    np.savez(os.path.join(FIG, "exp_b.npz"), err_th=err_th, err_r=err_r,
             th_g=th_g, r_g=r_g, th_traj=th_traj, r_traj=r_traj)


if __name__ == "__main__":
    cfg = Config()
    th, r = b1_monotonicity(cfg)
    b2_coverage(cfg, th, r)
