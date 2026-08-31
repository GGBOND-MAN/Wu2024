"""
Proposed method -- coherent space-frequency (delay-domain) near-field localization.

THE OBSERVATION BOTH PAPERS MISS
--------------------------------
The received phase at antenna n on subcarrier m is, with NO approximation of
any kind,

    Phi(n, m) = -(2 pi f_m / c) * r_n(theta, r),        f_m = f_0 + m * df

so along the FREQUENCY axis, at fixed n, the phase is EXACTLY linear in m with
slope -(2 pi df / c) r_n.  No Fresnel expansion, no far-field assumption, no
paraxial approximation -- this is just the definition of a propagation delay.

That gives a two-step, grid-free estimator:

  Step 1.  For every antenna n, read r_n off the phase-vs-frequency slope.
           This turns the array into N range-measuring virtual anchors, each
           with raw resolution c / (2W) = 5 cm at W = 3 GHz.
  Step 2.  Fit (theta, r) to the EXACT spherical geometry
               r_n = sqrt(r^2 + x_n^2 - 2 r x_n sin theta)
           by nonlinear least squares -- i.e. plain multilateration.

Near-field localization stops being an ill-conditioned curvature-fitting problem
and becomes a well-conditioned multilateration problem.  Neither paper uses the
frequency axis this way: Luo2024 uses phase differences only at a handful of
peak subcarriers, and Zhang2026 fuses per-subcarrier MUSIC spectra by a
GEOMETRIC MEAN (Eq. 44), which is incoherent and discards the slope entirely.

Unambiguous range: the slope stays inside (-pi, pi] while r_n < c M / (2W),
which is 102.4 m for the paper's own M = 2048 and W = 3 GHz.
"""
import numpy as np
from scipy.optimize import least_squares
from nf_model import C, dist_exact


def per_antenna_ranges(Y, freqs):
    """Step 1: r_n from the phase-vs-frequency slope of each row of Y.

    Y : (N, M) space-frequency observation, pilots already removed.
    Coarse delay from an IFFT peak, then a linear phase fit for sub-bin accuracy.
    """
    N, M = Y.shape
    df = freqs[1] - freqs[0]
    nfft = 1 << int(np.ceil(np.log2(M * 8)))

    # Coarse: Y*[n,m] ~ exp(+j2pi m df d_n / c), so an FFT along m peaks at
    # bin k = nfft * df * d_n / c.  (An ifft here would pick the wrong alias.)
    prof = np.fft.fft(Y.conj(), n=nfft, axis=1)
    kpk = np.argmax(np.abs(prof[:, : nfft // 2]), axis=1)
    r_coarse = kpk * C / (nfft * df)

    # Fine: de-rotate by the coarse delay.  The residual is at most half a bin,
    # so the residual phase stays well inside +-pi and needs no unwrapping.
    resid = Y * np.exp(2j * np.pi * np.outer(r_coarse, freqs) / C)
    ph = np.angle(resid * np.exp(-1j * np.angle(resid.mean(axis=1, keepdims=True))))
    fc_ = freqs - freqs.mean()
    slope = (fc_ * ph).sum(axis=1) / (fc_ ** 2).sum()
    return r_coarse - slope * C / (2 * np.pi)


def fit_geometry(r_hat, x, r0=None, th0=None, unknown_offset=False):
    """Step 2: least-squares multilateration on the exact spherical geometry."""
    if r0 is None:
        r0 = float(np.median(r_hat))
    if th0 is None:
        # Slope of r_n across the aperture is -sin(theta) to first order.
        g = np.polyfit(x, r_hat, 1)[0]
        th0 = np.arcsin(np.clip(-g, -1, 1))

    def resid(p):
        pred = dist_exact(p[0], p[1], x)
        return (pred + p[2] if unknown_offset else pred) - r_hat

    p0 = [r0, th0, 0.0] if unknown_offset else [r0, th0, 0.0]
    sol = least_squares(resid, p0[:3] if unknown_offset else p0[:2], method="lm",
                        xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return sol.x[1], sol.x[0]          # theta, r


def estimate(Y, freqs, x, unknown_offset=False):
    """Full proposed estimator.  Grid-free, exact-model, O(N M log M)."""
    return fit_geometry(per_antenna_ranges(Y, freqs), x, unknown_offset=unknown_offset)


def make_wideband_data(cfg, r, th, freqs, snr_db, rng, model="exact"):
    """Y[n,m] = alpha * exp(-j 2 pi f_m r_n / c) + noise, per-element SNR as elsewhere."""
    from nf_model import DIST
    d = DIST[model](r, th, cfg.x)
    Y = np.exp(-2j * np.pi * np.outer(d, freqs) / C) * np.exp(2j * np.pi * rng.random())
    sig = np.sqrt(10 ** (-snr_db / 10) / 2)
    return Y + sig * (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape))


def ml_refine(Y, freqs, x, th0, r0, n_sub=64):
    """Optional step 3: one local ML polish on the exact manifold.

    Step 1-2 are range-efficient but leave the angle short of the bound, because
    the geometry fit sees only the estimated ranges and throws the residual phase
    away.  Maximising the coherent matched filter

        |sum_{n,m} a*(theta, r, f_m)_n Y[n,m]|^2

    recovers it.  This is a smooth 2-parameter local search started from an
    estimate already accurate to ~0.02 deg and ~0.1 mm -- no grid, no ambiguity,
    a handful of function evaluations.  Contrast the 4,000,000-point grid that
    Zhang2026 Sec. IV-C-1 needs for the same nominal resolution.
    """
    from scipy.optimize import minimize
    f = freqs[:: max(1, len(freqs) // n_sub)][:n_sub]
    Ys = Y[:, :: max(1, len(freqs) // n_sub)][:, :n_sub]

    def negmf(p):
        d = dist_exact(p[0], p[1], x)
        A = np.exp(-2j * np.pi * np.outer(d, f) / C)
        return -abs(np.vdot(A, Ys))

    sol = minimize(negmf, [r0, th0], method="Nelder-Mead",
                   options=dict(xatol=1e-10, fatol=1e-10, maxiter=250))
    return sol.x[1], sol.x[0]


def estimate_full(Y, freqs, x):
    """Steps 1-3: delay-domain ranging, geometry fit, then local ML polish."""
    th, r = estimate(Y, freqs, x)
    return ml_refine(Y, freqs, x, th, r)
