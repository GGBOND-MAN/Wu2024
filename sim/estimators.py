"""
The Zhang2026 two-stage estimator, plus both CRLBs.

`music_refine` follows Algorithm 1 line by line: geometry-compensated spatial
smoothing over overlapping subarrays, EVD, a local 2-D search, and geometric
averaging of the per-subcarrier spectra (Eq. 44).

The local search is a multi-level zoom rather than one flat grid.  That is
strictly more generous than the paper -- it reaches a far finer resolution for
the same number of spectrum evaluations -- so any error floor we observe is a
property of the method, not of a coarse grid.
"""
import numpy as np
from nf_model import C, DIST, steering


# ----------------------------------------------------------------------------
# Stage I -- coarse localization from the received power spectrum (Eq. 23-25)
# ----------------------------------------------------------------------------

def coarse_estimate(z, traj_th, traj_r, snr_db=None, rng=None):
    """Peak-subcarrier index -> designed focal point of that subcarrier."""
    if snr_db is not None:
        p_sig = np.mean(np.abs(z) ** 2)
        sigma = np.sqrt(p_sig / (2 * 10 ** (snr_db / 10)))
        z = z + sigma * (rng.standard_normal(z.shape) + 1j * rng.standard_normal(z.shape))
    m = int(np.argmax(np.abs(z) ** 2))
    return traj_th[m], traj_r[m], m


# ----------------------------------------------------------------------------
# Stage II -- geometry-compensated smoothing + local near-field MUSIC
# ----------------------------------------------------------------------------

def smoothed_covariance(y, cfg, th0, r0, f, model="fresnel"):
    """Eq. (30)-(35).  Returns the Ms x Ms aligned, smoothed covariance."""
    x, Ms, P = cfg.x, cfg.Ms, cfg.P
    p0 = (P - 1) // 2
    d_ref = DIST[model](r0, th0, x[p0:p0 + Ms])              # reference subarray

    R = np.zeros((Ms, Ms), dtype=np.complex128)
    for p in range(P):
        d_p = DIST[model](r0, th0, x[p:p + Ms])
        D = np.exp(-2j * np.pi * f * (d_ref - d_p) / C)      # Eq. (33)
        yt = D * y[p:p + Ms]                                 # Eq. (34)
        R += np.outer(yt, yt.conj())
    return R / P


def _music_den(grid_th, grid_r, cfg, f, u1, model="fresnel"):
    """a^H Un Un^H a = ||a||^2 - |u1^H a|^2 for a one-dimensional signal subspace."""
    p0 = (cfg.P - 1) // 2
    xs = cfg.x[p0:p0 + cfg.Ms]
    TH, R = np.meshgrid(grid_th, grid_r, indexing="ij")
    d = DIST[model](R[..., None], TH[..., None], xs)          # (nt, nr, Ms)
    A = np.exp(-2j * np.pi * f * d / C)
    proj = A @ u1.conj()
    return cfg.Ms - np.abs(proj) ** 2


def music_refine(ys, cfg, th0, r0, sub_freqs, model="fresnel",
                 dth=np.deg2rad(1.0), dr=1.0, n_grid=41, n_levels=3):
    """Stage II of Algorithm 1.  `ys[i]` is the array snapshot at sub_freqs[i]."""
    # Aligned covariance and signal eigenvector, once per subcarrier.
    u1s = []
    for y, f in zip(ys, sub_freqs):
        R = smoothed_covariance(y, cfg, th0, r0, f, model)
        w, V = np.linalg.eigh(R)
        u1s.append(V[:, -1])

    th_c, r_c, hw_th, hw_r = th0, r0, dth, dr
    for _ in range(n_levels):
        g_th = np.linspace(th_c - hw_th, th_c + hw_th, n_grid)
        g_r = np.linspace(r_c - hw_r, r_c + hw_r, n_grid)
        # Geometric mean of spectra == arithmetic mean of log-spectra (Eq. 44).
        acc = np.zeros((n_grid, n_grid))
        for u1, f in zip(u1s, sub_freqs):
            acc -= np.log(np.maximum(_music_den(g_th, g_r, cfg, f, u1, model), 1e-30))
        i, j = np.unravel_index(np.argmax(acc), acc.shape)
        th_c, r_c = g_th[i], g_r[j]
        hw_th, hw_r = 2 * hw_th / (n_grid - 1), 2 * hw_r / (n_grid - 1)
    return th_c, r_c


def make_snapshot(cfg, r, th, f, snr_db, rng, model="exact"):
    """y_m = alpha_m a(theta, r, f_m) + n_m   (Eq. 28), alpha with random phase."""
    a = steering(r, th, f, cfg.x, model)
    a = a * np.exp(2j * np.pi * rng.random())
    sigma = np.sqrt(10 ** (-snr_db / 10) / 2)     # per-element SNR = |alpha|^2/sigma^2
    return a + sigma * (rng.standard_normal(cfg.N) + 1j * rng.standard_normal(cfg.N))


# ----------------------------------------------------------------------------
# CRLB -- the paper's version, and the version that marginalises alpha
# ----------------------------------------------------------------------------

def crlb(cfg, r, th, snr_db, f=None, marginalize=False, L=1):
    """Zhang2026 Appendix B (marginalize=False) vs the correct bound.

    The paper declares the complex amplitude unknown but never marginalises it
    out of the FIM, so its Eq. (59)/(64) credits the estimator with knowing the
    absolute carrier phase.  With alpha unknown the FIM is

        J = (2 |alpha|^2 / sigma^2) Re{ D^H Pi_a^perp D },  D = [d_theta, d_r]

    i.e. the components of dtheta/dr that lie along `a` carry no information.
    """
    f = cfg.fc if f is None else f
    x = cfg.x
    k = 2 * np.pi * f / C
    dPhi_dth = k * (x * np.cos(th) - x ** 2 * np.sin(2 * th) / (2 * r))   # Eq. (62)
    dPhi_dr = -k * (1 - x ** 2 * np.cos(th) ** 2 / (2 * r ** 2))          # Eq. (63)

    a = steering(r, th, f, x, "fresnel")
    D = np.stack([a * 1j * dPhi_dth, a * 1j * dPhi_dr], axis=1)           # Eq. (60)/(61)

    if marginalize:
        D = D - np.outer(a, a.conj() @ D) / (a.conj() @ a)                # Pi_a^perp D

    snr = 10 ** (snr_db / 10)                     # per-element |alpha|^2/sigma^2
    J = 2 * snr * L * np.real(D.conj().T @ D)
    Ci = np.linalg.inv(J)
    return np.sqrt(Ci[0, 0]), np.sqrt(Ci[1, 1])   # RMSE bounds on theta (rad), r (m)


def crlb_wideband(cfg, r, th, snr_db, n_sub=128, coherent=True, model="fresnel"):
    """Joint space-frequency CRLB over the OFDM band.

    coherent=True : one unknown complex alpha shared by all subcarriers, so the
        phase-vs-frequency slope -2 pi f_m r / c is observable.  This is the
        regime a delay-aware estimator can reach.
    coherent=False: an independent unknown alpha_m per subcarrier, which is what
        per-subcarrier processing (Zhang2026 Eq. 41/44) is actually limited to.

    SNR is per element per subcarrier, matching `crlb`.
    """
    x = cfg.x
    f = cfg.freqs[:: max(1, cfg.M // n_sub)][:n_sub]
    scale = cfg.M / len(f)                       # extrapolate to all M subcarriers
    k = 2 * np.pi * f[:, None] / C

    dPhi_dth = k * (x * np.cos(th) - x ** 2 * np.sin(2 * th) / (2 * r))
    dPhi_dr = -k * (1 - x ** 2 * np.cos(th) ** 2 / (2 * r ** 2))
    A = np.exp(-2j * np.pi * f[:, None] * DIST[model](r, th, x) / C)      # (F, N)
    Dth, Dr = A * 1j * dPhi_dth, A * 1j * dPhi_dr

    snr = 10 ** (snr_db / 10)
    if coherent:
        a = A.ravel()
        D = np.stack([Dth.ravel(), Dr.ravel()], axis=1)
        D = D - np.outer(a, a.conj() @ D) / (a.conj() @ a)
        J = 2 * snr * np.real(D.conj().T @ D) * scale
    else:
        J = np.zeros((2, 2))
        for i in range(len(f)):
            a = A[i]
            D = np.stack([Dth[i], Dr[i]], axis=1)
            D = D - np.outer(a, a.conj() @ D) / (a.conj() @ a)
            J += 2 * snr * np.real(D.conj().T @ D)
        J *= scale
    Ci = np.linalg.inv(J)
    return np.sqrt(Ci[0, 0]), np.sqrt(Ci[1, 1])
