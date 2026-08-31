"""
Near-field wideband XL-MIMO model, reproducing the setup of

  Zhang2026 : "Beam Squint Assisted Joint Angle-Distance Localization for
               Near-Field Communications", IEEE TVT, DOI 10.1109/TVT.2026.3706538
  Luo2024   : "Beam Squint Assisted User Localization in Near-Field ISAC
               Systems", IEEE TWC 23(5):4504-4517, 2024

Everything here follows the papers' own equations.  The one deliberate addition
is `dist_exact`, which lets us generate data with the true spherical wavefront
while the estimator keeps using the Fresnel model the papers assume.  That gap
is what Experiment A measures.
"""
import numpy as np

C = 299_792_458.0


class Config:
    """Default configuration of Zhang2026, Table II."""

    def __init__(self, N=256, fc=60e9, W=3e9, M=2048, Ms=128,
                 theta_min=-60.0, theta_max=60.0, r_min=15.0, r_max=50.0):
        self.N, self.fc, self.W, self.M, self.Ms = N, fc, W, M, Ms
        self.lam = C / fc
        self.d = self.lam / 2
        self.f0 = fc - W / 2                      # lowest subcarrier, Luo's convention
        self.fM = fc + W / 2
        self.P = N - Ms + 1                       # overlapping subarrays, unit shift
        self.theta_min, self.theta_max = np.deg2rad(theta_min), np.deg2rad(theta_max)
        self.r_min, self.r_max = r_min, r_max

    @property
    def x(self):
        """Antenna positions x_n = n_d * d, centred on the array (Zhang2026 Eq. 58)."""
        return (np.arange(self.N) - (self.N - 1) / 2) * self.d

    @property
    def freqs(self):
        """f_m = fc - W/2 + m*df  (Zhang2026 Eq. 1)."""
        return self.f0 + np.arange(self.M) * (self.W / self.M)

    @property
    def aperture(self):
        return (self.N - 1) * self.d

    @property
    def rayleigh(self):
        return 2 * self.aperture ** 2 / self.lam


# ----------------------------------------------------------------------------
# Geometry: exact spherical wave vs the second-order Fresnel approximation
# ----------------------------------------------------------------------------

def dist_exact(r, theta, x):
    """r_n = sqrt(r^2 + x^2 - 2 r x sin(theta))   -- Zhang2026 Eq. (15)."""
    return np.sqrt(r ** 2 + x ** 2 - 2 * r * x * np.sin(theta))


def dist_fresnel(r, theta, x):
    """r_n ~ r - x sin(theta) + x^2 cos^2(theta) / (2r)  -- Zhang2026 Eq. (2)/(16)."""
    return r - x * np.sin(theta) + x ** 2 * np.cos(theta) ** 2 / (2 * r)


DIST = {"exact": dist_exact, "fresnel": dist_fresnel}


def steering(r, theta, f, x, model="fresnel"):
    """[a]_n = exp(-j 2 pi f r_n / c)  -- Zhang2026 Eq. (5)/(27)."""
    return np.exp(-2j * np.pi * f * DIST[model](r, theta, x) / C)


# ----------------------------------------------------------------------------
# Controllable beam-squint trajectory (Luo2024 Eq. 19/20 == Zhang2026 Eq. 19/20)
# ----------------------------------------------------------------------------

def trajectory(cfg, theta_s, r_s, theta_e, r_e, f=None):
    """Designed focal point (theta_m, r_m) of each subcarrier.

    Uses Luo's convention f_tilde = f_m - f0 in [0, W], which is the only one
    under which Eq. (19)/(20) reduce to the endpoints at m = 0 and m = M-1.
    """
    f = cfg.freqs if f is None else np.atleast_1d(f)
    ft = f - cfg.f0
    W, f0 = cfg.W, cfg.f0

    w_s = (W - ft) * f0 / (W * f)          # weight of the start point
    w_e = (W + f0) * ft / (W * f)          # weight of the end point

    sin_th = w_s * np.sin(theta_s) + w_e * np.sin(theta_e)
    sin_th = np.clip(sin_th, -1.0, 1.0)
    th = np.arcsin(sin_th)

    inv_r = (w_s * np.cos(theta_s) ** 2 / r_s
             + w_e * np.cos(theta_e) ** 2 / r_e) / np.cos(th) ** 2
    return th, 1.0 / inv_r


def ttd_ps_weights(cfg, theta_s, r_s, theta_e, r_e, model="fresnel"):
    """PS phases and TTD delays that realise the trajectory (Luo2024 Sec. III-C).

        phi_n = f0 * r_{s,n} / c ,   t_n = fM * r_{e,n} / (W c) - phi_n / W
    """
    x = cfg.x
    phi = cfg.f0 * DIST[model](r_s, theta_s, x) / C
    t = cfg.fM * DIST[model](r_e, theta_e, x) / (cfg.W * C) - phi / cfg.W
    return phi, t


def coarse_observation(cfg, r_k, theta_k, phi, t, model="fresnel"):
    """Scalar per-subcarrier observation z_m used by the coarse stage.

        z_m = sum_n exp( j2pi [ f_m r_{k,n}/c - phi_n - f_tilde_m t_n ] )

    which is Luo2024 Eq. (21) up to the constant alpha/sqrt(N).
    """
    x = cfg.x
    rk_n = DIST[model](r_k, theta_k, x)                     # (N,)
    f = cfg.freqs[:, None]                                  # (M,1)
    ft = (cfg.freqs - cfg.f0)[:, None]
    ph = f * rk_n[None, :] / C - phi[None, :] - ft * t[None, :]
    return np.exp(2j * np.pi * ph).sum(axis=1)              # (M,)
