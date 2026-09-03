"""
Baselines, including the one Zhang2026 never ran.

Zhang2026 benchmarks only against CBS-Low, the weaker of Luo2024's two schemes,
and never against CBS-High -- which is precisely the variant that already
mitigates the sequential error propagation Zhang2026 sets out to fix.  Criticising
that and then repeating it would be indefensible, so CBS-High is implemented here
from Luo2024 Sec. IV-C.

CBS-Low  (Luo2024 Sec. IV-B): one angular sweep for theta, then one radial sweep
         at that angle for r.  2 sweeps.
CBS-High (Luo2024 Sec. IV-C): P sweeps with slightly different angular spans.
         Angle is the average of the P per-sweep estimates (its Eq. 26).  Range
         comes from a one-dimensional search that matches the measured phases of
         the P peak subcarriers against their predicted phases (its Eq. 27).

Both are stated in Luo2024 for downlink with user feedback.  They are
implemented here in the monostatic model used throughout this repository, so
that every scheme sees the same observation; the adaptation is noted rather
than glossed over.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from nf_model import (Config, C, trajectory, ttd_ps_weights,
                      coarse_observation, dist_fresnel)


def sweep(cfg, r_k, th_k, th_s, r_s, th_e, r_e, snr_db, rng, model="fresnel"):
    """One controllable-squint sweep: returns the complex per-subcarrier echo.

    Noise is referred to the antenna elements at the SAME per-element SNR used
    everywhere else in this repository, so every scheme sees one physical noise
    level.  coarse_observation returns sum_n exp(j.) with unit-modulus terms, so
    the combined noise has variance N * sigma^2.

    Normalising instead to the sweep's MEAN received power -- the obvious first
    guess -- would be unfair to these baselines: a squint sweep puts most of its
    subcarriers off-target, so the mean sits far below the peak and the baseline
    would be handed far more noise than the proposed scheme.
    """
    phi, tau = ttd_ps_weights(cfg, th_s, r_s, th_e, r_e, model)
    z = coarse_observation(cfg, r_k, th_k, phi, tau, model)
    sigma2 = 10 ** (-snr_db / 10)
    s = np.sqrt(cfg.N * sigma2 / 2)
    return z + s * (rng.standard_normal(z.shape) + 1j * rng.standard_normal(z.shape))


def cbs_low(cfg, r_k, th_k, snr_db, rng):
    """Luo2024 CBS-Low: angular sweep then radial sweep.  2 probes."""
    r_mid = 0.5 * (cfg.r_min + cfg.r_max)
    th_t, r_t = trajectory(cfg, cfg.theta_max, r_mid, cfg.theta_min, r_mid)
    z = sweep(cfg, r_k, th_k, cfg.theta_max, r_mid, cfg.theta_min, r_mid, snr_db, rng)
    th_hat = th_t[int(np.argmax(np.abs(z) ** 2))]

    _, r_t2 = trajectory(cfg, th_hat, cfg.r_min, th_hat, cfg.r_max)
    z2 = sweep(cfg, r_k, th_k, th_hat, cfg.r_min, th_hat, cfg.r_max, snr_db, rng)
    return th_hat, r_t2[int(np.argmax(np.abs(z2) ** 2))]


def _phase_at(cfg, rs, th, phi, tau, f_m):
    """Exact phase of the echo at ONE subcarrier, for a whole grid of candidate
    ranges -- Luo2024 Eq. (28) evaluated directly rather than through a
    hand-rolled approximation.  Only the common delay term is obvious in closed
    form; the sum over antennas also moves with r, and dropping it puts the
    range search on the wrong surface entirely."""
    rn = dist_fresnel(np.atleast_1d(rs)[:, None], th, cfg.x[None, :])   # (G, N)
    ph = f_m * rn / C - phi[None, :] - (f_m - cfg.f0) * tau[None, :]
    return np.angle(np.exp(2j * np.pi * ph).sum(axis=1))


def cbs_high(cfg, r_k, th_k, snr_db, rng, P=10, stagger_deg=1.0, n_grid=16000,
             stagger_mode="step"):
    """Luo2024 CBS-High: P staggered sweeps, angle averaged, range from phases.

    P probes.  The range search (its Eq. 27) coherently combines the measured
    phases of the P peak subcarriers, which is a sparse sampling of the delay
    axis -- P points out of M, at irregular spacings.
    """
    # r_mid1 = r_mid2 at the region midpoint.  The author confirms the two were
    # equal, and calibration shows the angle anchors do not depend on the value:
    # across 0.15-0.85 of the region the spread was 10.3%, against the 9.1%
    # standard error of the trial count, so it is noise rather than dependence.
    r_mid = 0.5 * (cfg.r_min + cfg.r_max)
    ths, fs, phases, wts = [], [], [], []
    for p in range(P):
        # Luo2024 Sec. IV-C requires slightly different spans per sweep but never
        # says how they are laid out, and the two natural readings scale with P
        # differently: dividing a fixed span among P sweeps tops out at a 2.1x
        # range improvement from P=4 to P=12, while a fixed STEP grows the span
        # as (P-1) and reaches the paper's own 4.46x.  Calibrating against its
        # Fig. 13 anchors selects the step reading, and the author confirms it:
        # the spans run "60, 61, 62 ..." symmetrically about broadside.  The
        # 1 degree step is the author's recollection, and matches the P=12
        # anchor to 1.02x.
        if stagger_mode == "step":
            pad = np.deg2rad(stagger_deg) * (p - (P - 1) / 2)
        else:
            pad = np.deg2rad(stagger_deg) * (p - (P - 1) / 2) / max(P - 1, 1)
        th_s, th_e = cfg.theta_max + pad, cfg.theta_min - pad
        th_t, _ = trajectory(cfg, th_s, r_mid, th_e, r_mid)
        z = sweep(cfg, r_k, th_k, th_s, r_mid, th_e, r_mid, snr_db, rng)
        m = int(np.argmax(np.abs(z) ** 2))
        ths.append(th_t[m]); fs.append(cfg.freqs[m]); phases.append(np.angle(z[m]))
        wts.append(ttd_ps_weights(cfg, th_s, r_mid, th_e, r_mid))

    th_hat = float(np.mean(ths))                                  # Eq. (26)

    # Eq. (27): pick r maximising |sum_p exp(j[phi_measured - phi_predicted(r)])|
    # Search beyond the sensing region: with the grid ending exactly at r_min, a
    # search that pins to the lower edge scores ZERO error for a user at r_min,
    # turning a failure into an apparently perfect result.  The margin makes
    # edge-pinning visible as the error it is.
    # Lower edge clamped strictly positive: dist_fresnel divides by r, so a grid
    # reaching 0 fills the score with NaN and argmax silently returns the first
    # bin.  With the default r_min = 15 the margin never reached 0 and this never
    # fired; it does the moment r_min <= 5, as in Luo2024's own Fig. 13 setup.
    grid = np.linspace(max(1.0, cfg.r_min - 5.0), cfg.r_max + 10.0, n_grid)
    score = np.zeros(n_grid)
    acc = np.zeros(n_grid, dtype=complex)
    for p in range(P):
        pred = _phase_at(cfg, grid, th_hat, wts[p][0], wts[p][1], fs[p])
        acc += np.exp(1j * (phases[p] - pred))
    score = np.abs(acc)
    return th_hat, float(grid[int(np.argmax(score))])


CKPT = os.path.join(os.path.dirname(__file__), "figs", "baselines_ckpt.json")
BUDGET = float(os.environ.get("BASELINE_BUDGET", 95))


def run():
    import json, time
    cfg = Config()
    rng = np.random.default_rng(11)
    SNRS = [-10, -5, 0, 5, 10, 15, 20]
    N_TRIAL = 40
    done = json.load(open(CKPT)) if os.path.exists(CKPT) else {}
    t0 = time.perf_counter()
    r_k, th_k = 30.0, np.deg2rad(15.0)

    print(f"user ({np.rad2deg(th_k):.0f} deg, {r_k} m), {N_TRIAL} trials\n")
    print(f"{'SNR':>5} | {'CBS-Low (2 probes)':>28} | {'CBS-High (10 probes)':>28}")
    print(f"{'':>5} | {'RMSE_th':>13}{'RMSE_r':>15} | {'RMSE_th':>13}{'RMSE_r':>15}")
    for snr in SNRS:
        if str(snr) in done:
            lo, hi = done[str(snr)]["low"], done[str(snr)]["high"]
            print(f"{snr:>+5} | {lo[0]:>13.3e}{lo[1]:>15.3e} | {hi[0]:>13.3e}{hi[1]:>15.3e}")
            continue
        if time.perf_counter() - t0 > BUDGET:
            print(f"  [budget reached, {len(done)}/{len(SNRS)} done -- re-run to continue]")
            return
        acc = {k: ([], []) for k in ("low", "high")}
        for _ in range(N_TRIAL):
            for key, fn in (("low", cbs_low), ("high", cbs_high)):
                t, r = fn(cfg, r_k, th_k, snr, rng)
                acc[key][0].append(t - th_k); acc[key][1].append(r - r_k)
        rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
        done[str(snr)] = {k: (np.rad2deg(rms(acc[k][0])), rms(acc[k][1])) for k in acc}
        json.dump(done, open(CKPT, "w"))
        lo, hi = done[str(snr)]["low"], done[str(snr)]["high"]
        print(f"{snr:>+5} | {lo[0]:>13.3e}{lo[1]:>15.3e} | {hi[0]:>13.3e}{hi[1]:>15.3e}", flush=True)
    np.savez(os.path.join(os.path.dirname(__file__), "figs", "baselines.npz"), snr=SNRS,
             low_th=[done[str(s)]["low"][0] for s in SNRS],
             low_r=[done[str(s)]["low"][1] for s in SNRS],
             high_th=[done[str(s)]["high"][0] for s in SNRS],
             high_r=[done[str(s)]["high"][1] for s in SNRS])


if __name__ == "__main__":
    run()
