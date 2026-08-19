"""
Model-free (Lipari-Szabo) order-parameter fitting under anisotropic overall
tumbling, restricted to Models 1-3 (S2; S2+tau_e; S2+Rex).

Physics summary (see the accompanying methods writeup for the full
derivation): the total spectral density for a bond vector under anisotropic
overall tumbling plus fast internal motion is

    J(w) = sum_k A_k * (2/5) * [ S2*tau_k/(1+w^2 tau_k^2)
                                  + (1-S2)*tau_k'/(1+w^2 tau_k'^2) ]
    1/tau_k' = 1/tau_k + 1/tau_e

where (A_k, tau_k) are the five Woessner (1962) mode amplitudes/correlation
times for one N-H bond, already computed by
`makeshift.hydronmr.physics.nmr.mode_amplitudes` from the rigid-body
diffusion tensor. This module only adds the internal-motion (S2, tau_e)
layer on top of that existing per-residue anisotropic tumbling machinery.

Models 4 and 5 (three free parameters: S2+tau_e+Rex, or S2f+S2s+tau_e) are
not implemented here: against three single-field observables (R1, R2, NOE)
they are exactly- or over-parameterized and, in practice, ill-conditioned
by near-degeneracy between their extra parameters. Model selection between
1/2/3 is done from which residual each extra parameter physically explains
(a NOE deficit relative to the rigid prediction implicates tau_e; an R2
excess beyond what the R1-derived S2 already explains implicates Rex) —
not by the nested F-test/SSE-percentile cascade conventional model-free
software uses to arbitrate among five candidate models.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

# tau_e search range (seconds): a few ps up to a few ns, comfortably below
# typical overall correlation times (tau_m ~ 5-15 ns for globular proteins)
# so tau_e stays a genuinely fast, separated timescale.
TAU_E_MIN_S = 1.0e-12
TAU_E_MAX_S = 3.0e-9


def spectral_density_rigid(amplitudes, taus, omega):
    """J(w) for S2=1 (no internal motion) -- the same rigid-body anisotropic
    spectral density used for HYDRONMR-style rigid predictions."""
    return sum(a * (2.0 / 5.0) * t / (1.0 + (omega * t) ** 2)
               for a, t in zip(amplitudes, taus))


def spectral_density(amplitudes, taus, omega, S2, tau_e=0.0):
    """General model-free J(w): S2 alone when tau_e<=0 (each tau_k' -> 0,
    so the (1-S2) term vanishes and this reduces exactly to
    S2 * spectral_density_rigid(...))."""
    total = 0.0
    for a, tk in zip(amplitudes, taus):
        if tau_e > 0.0:
            tk_prime = (tk * tau_e) / (tk + tau_e)   # 1 / (1/tk + 1/tau_e)
            fast = (1.0 - S2) * tk_prime / (1.0 + (omega * tk_prime) ** 2)
        else:
            fast = 0.0
        total += a * (2.0 / 5.0) * (S2 * tk / (1.0 + (omega * tk) ** 2) + fast)
    return total


def relaxation_rates(d2, c2, gamma_h, gamma_x, omega_h, omega_x,
                      amplitudes, taus, S2, tau_e=0.0):
    """R1, R2, NOE (no Rex) from J(w) built with the given S2/tau_e, using
    the same dipolar+CSA combination as makeshift.hydronmr.physics.nmr."""
    def J(w):
        return spectral_density(amplitudes, taus, w, S2, tau_e)

    j0 = J(0.0)
    jx = J(omega_x)
    jh = J(omega_h)
    jm = J(omega_h - omega_x)
    jp = J(omega_h + omega_x)

    r1 = d2 / 4.0 * (jm + 3 * jx + 6 * jp) + c2 * jx
    r2 = (d2 / 8.0 * (4 * j0 + jm + 3 * jx + 6 * jh + 6 * jp)
          + c2 / 6.0 * (4 * j0 + 3 * jx))
    noe = 1.0 + (d2 / (4.0 * r1)) * (gamma_h / gamma_x) * (6 * jp - jm)
    return r1, r2, noe


def calibrate_tau_scale(residues, d2, c2, gamma_h, gamma_x, omega_h, omega_x):
    """
    Fit a single global scale factor k applied to every mode's tau_k
    (tau_k -> k*tau_k, for all five modes and all residues alike) so the
    rigid (S2=1) R1/R2 prediction best matches observed R1/R2 for a set of
    presumed-rigid calibration residues.

    The hydrodynamic bead model's rotational-diffusion *anisotropy shape*
    (the amplitudes A_k, i.e. which residues are relatively more/less
    exposed to which tumbling mode) is validated elsewhere
    (demos/hydronmr_validation) to track the true tensor well; its
    *absolute* timescale is a known, documented approximation (uniform
    3.0-Angstrom-per-heavy-atom beads, not the real AtoB table -- see
    makeshift/hydronmr/physics/structure.py). `add_rigid_prediction`
    already corrects for this same absolute-magnitude offset with a
    scale factor on the predicted R2/R1 ratio; this is the analogous
    correction for the model-free fit, applied to the underlying
    correlation times (so it also corrects R1 and NOE, not just the
    R2/R1 ratio) before any residue's S2 is fit.

    `residues` is a list of (amplitudes, taus, R1_obs, R1_err, R2_obs,
    R2_err) tuples for the calibration set. Returns k (dimensionless).
    """
    def chi2(log_k):
        k = 10.0 ** log_k
        total = 0.0
        for amplitudes, taus, R1_obs, R1_err, R2_obs, R2_err in residues:
            taus_k = [t * k for t in taus]
            r1p, r2p, _ = relaxation_rates(
                d2, c2, gamma_h, gamma_x, omega_h, omega_x,
                amplitudes, taus_k, S2=1.0, tau_e=0.0)
            w1 = 1.0 / R1_err ** 2 if (R1_err and np.isfinite(R1_err) and R1_err > 0) else 1.0
            w2 = 1.0 / R2_err ** 2 if (R2_err and np.isfinite(R2_err) and R2_err > 0) else 1.0
            total += w1 * (R1_obs - r1p) ** 2 + w2 * (R2_obs - r2p) ** 2
        return total

    grid = np.linspace(-1.0, 1.0, 41)   # k in [0.1, 10]
    costs = [chi2(lk) for lk in grid]
    best = grid[int(np.argmin(costs))]
    step = grid[1] - grid[0]
    res = minimize_scalar(chi2, bounds=(best - step, best + step),
                           method="bounded", options={"xatol": 1e-4})
    return 10.0 ** res.x


def _weighted_slope(pairs):
    """Weighted least squares slope S2 of y_i = S2 * x_i (through the
    origin), for pairs of (y_obs, y_err, x_pred). Skips entries with a
    non-finite or non-positive x_pred. Returns (S2, S2_err); (nan, nan) if
    nothing usable was given."""
    num = den = 0.0
    for y_obs, y_err, x in pairs:
        if x is None or not np.isfinite(x) or x <= 0:
            continue
        if y_obs is None or not np.isfinite(y_obs):
            continue
        w = 1.0 / y_err ** 2 if (y_err and np.isfinite(y_err) and y_err > 0) else 1.0
        num += w * y_obs * x
        den += w * x ** 2
    if den <= 0:
        return np.nan, np.nan
    return num / den, 1.0 / np.sqrt(den)


@dataclass
class ResidueFit:
    model: Optional[str]       # "1", "2", "3", "ambiguous", or None (no fit attempted)
    S2: float = np.nan
    S2_err: float = np.nan
    tau_e_ps: float = np.nan
    Rex: float = np.nan
    NOE_pred_rigid: float = np.nan
    noe_flag: bool = False     # NOE deficit vs. rigid prediction (tau_e signature)
    r2_flag: bool = False      # R2 excess vs. S2(R1)-scaled rigid prediction (Rex signature)


def fit_residue(R1_obs, R1_err, R2_obs, R2_err, NOE_obs, NOE_err,
                 amplitudes, taus, d2, c2, gamma_h, gamma_x, omega_h, omega_x,
                 sigma_flag=2.0, noe_fallback_err=0.05, r2_fallback_frac=0.05):
    """Fit one residue's order parameter, choosing among Models 1-3 by the
    physical-residual criterion described in the module docstring.

    `sigma_flag` is the number of (error-propagated) standard deviations a
    residual must exceed to be considered a real deficit/excess rather than
    noise. When an observable's experimental error is missing/zero,
    `noe_fallback_err` (absolute) or `r2_fallback_frac` (fractional) is
    used as a stand-in so the diagnostic still has a scale to compare
    against.

    Returns a ResidueFit. If R1_obs is missing/non-finite, no fit is
    attempted (Model 1/3's S2 depends on R1; Model 2 needs it too) and
    `model` is None.
    """
    if R1_obs is None or not np.isfinite(R1_obs) or R1_obs <= 0:
        return ResidueFit(model=None)

    R1_rigid, R2_rigid, NOE_rigid = relaxation_rates(
        d2, c2, gamma_h, gamma_x, omega_h, omega_x, amplitudes, taus,
        S2=1.0, tau_e=0.0)

    # Model 1 point estimate (closed form, R1+R2), used as the baseline S2
    # for both diagnostics and as the fallback value for ambiguous residues.
    S2_m1, S2_m1_err = _weighted_slope([
        (R1_obs, R1_err, R1_rigid),
        (R2_obs, R2_err, R2_rigid),
    ])

    # NOE diagnostic: NOE is independent of S2 when tau_e=0, so any
    # deficit relative to the parameter-free rigid prediction implicates
    # tau_e != 0, regardless of what S2_m1 came out to.
    noe_flag = False
    if NOE_obs is not None and np.isfinite(NOE_obs):
        noe_err = NOE_err if (NOE_err and np.isfinite(NOE_err) and NOE_err > 0) \
            else noe_fallback_err
        noe_flag = (NOE_rigid - NOE_obs) > sigma_flag * noe_err

    # Rex diagnostic: S2 from R1 alone is exchange-free (Rex doesn't touch
    # R1); compare the R2 that S2 implies against what was actually observed.
    r2_flag = False
    S2_from_R1 = R1_obs / R1_rigid if R1_rigid > 0 else np.nan
    if R2_obs is not None and np.isfinite(R2_obs) and np.isfinite(S2_from_R1):
        r2_pred_no_rex = S2_from_R1 * R2_rigid
        r2_err = R2_err if (R2_err and np.isfinite(R2_err) and R2_err > 0) \
            else r2_fallback_frac * max(R2_obs, 1e-12)
        r2_flag = (R2_obs - r2_pred_no_rex) > sigma_flag * r2_err

    if noe_flag and r2_flag:
        return ResidueFit(model="ambiguous", S2=S2_m1, S2_err=S2_m1_err,
                           NOE_pred_rigid=NOE_rigid, noe_flag=True, r2_flag=True)

    if noe_flag:
        S2, tau_e_s, S2_err = _fit_model2(
            R1_obs, R1_err, R2_obs, R2_err,
            d2, c2, gamma_h, gamma_x, omega_h, omega_x, amplitudes, taus)
        return ResidueFit(model="2", S2=S2, S2_err=S2_err,
                           tau_e_ps=tau_e_s * 1e12, NOE_pred_rigid=NOE_rigid,
                           noe_flag=True)

    if r2_flag:
        Rex = R2_obs - S2_from_R1 * R2_rigid
        return ResidueFit(model="3", S2=S2_from_R1, S2_err=np.nan, Rex=Rex,
                           NOE_pred_rigid=NOE_rigid, r2_flag=True)

    return ResidueFit(model="1", S2=S2_m1, S2_err=S2_m1_err,
                       NOE_pred_rigid=NOE_rigid)


def _fit_model2(R1_obs, R1_err, R2_obs, R2_err,
                 d2, c2, gamma_h, gamma_x, omega_h, omega_x, amplitudes, taus):
    """Model 2 (S2, tau_e): 1D bounded search over tau_e (the only place it
    enters nonlinearly), with S2 solved by exact weighted linear regression
    against R1 and R2 at each trial tau_e (both are affine in S2 for fixed
    tau_e: R(S2) = R_B(tau_e) + S2*(R_rigid - R_B(tau_e)), where R_B is the
    S2=0 endpoint at that tau_e). Returns (S2, tau_e_seconds, S2_err)."""
    R1_rigid, R2_rigid, _ = relaxation_rates(
        d2, c2, gamma_h, gamma_x, omega_h, omega_x, amplitudes, taus,
        S2=1.0, tau_e=0.0)

    def inner_S2(tau_e):
        R1_b, R2_b, _ = relaxation_rates(
            d2, c2, gamma_h, gamma_x, omega_h, omega_x, amplitudes, taus,
            S2=0.0, tau_e=tau_e)
        S2, S2_err = _weighted_slope([
            (R1_obs - R1_b, R1_err, R1_rigid - R1_b),
            (R2_obs - R2_b, R2_err, R2_rigid - R2_b),
        ])
        return S2, S2_err

    def chi2(log_tau_e):
        tau_e = 10.0 ** log_tau_e
        S2, _ = inner_S2(tau_e)
        if not np.isfinite(S2):
            return np.inf
        S2 = min(max(S2, 0.0), 1.0)
        r1p, r2p, _ = relaxation_rates(
            d2, c2, gamma_h, gamma_x, omega_h, omega_x, amplitudes, taus,
            S2=S2, tau_e=tau_e)
        w1 = 1.0 / R1_err ** 2 if (R1_err and np.isfinite(R1_err) and R1_err > 0) else 1.0
        w2 = 1.0 / R2_err ** 2 if (R2_err and np.isfinite(R2_err) and R2_err > 0) else 1.0
        return w1 * (R1_obs - r1p) ** 2 + w2 * (R2_obs - r2p) ** 2

    # coarse log-spaced grid, then a bounded local polish -- mirrors the
    # perl script's own "gridsearch then Powell" pattern.
    log_lo, log_hi = np.log10(TAU_E_MIN_S), np.log10(TAU_E_MAX_S)
    grid = np.linspace(log_lo, log_hi, 40)
    costs = [chi2(g) for g in grid]
    best_log = grid[int(np.argmin(costs))]

    lo = max(log_lo, best_log - (grid[1] - grid[0]))
    hi = min(log_hi, best_log + (grid[1] - grid[0]))
    res = minimize_scalar(chi2, bounds=(lo, hi), method="bounded",
                           options={"xatol": 1e-3})
    tau_e = 10.0 ** res.x
    S2, S2_err = inner_S2(tau_e)
    S2 = min(max(S2, 0.0), 1.0) if np.isfinite(S2) else np.nan
    return S2, tau_e, S2_err
