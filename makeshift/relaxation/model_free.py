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
from scipy.optimize import minimize_scalar, minimize

from ..hydronmr.physics.nmr import diffusion_tensor_taus, mode_amplitudes_raw


class FittingError(RuntimeError):
    """Raised by `RelaxationProfile.fit_order_parameters` when the
    tensor calibration can't be trusted (see `assess_calibration`) and no
    S2 fit was attempted, rather than returning numbers that silently
    shouldn't be trusted."""

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


def calibrated_tensor(dx0, dy0, dz0, rate, shape):
    """Apply a (rate, shape) calibration to the hydrodynamic diffusion
    tensor's eigenvalues (dx0,dy0,dz0), in the spirit of modelfree4's own
    `diffusion` tensor optimization (it refits the tensor's overall rate
    and axial ratio against the data, not just a single isotropic
    correlation time).

    `shape` redistributes anisotropy around the tensor's mean eigenvalue
    without changing that mean: shape=1 reproduces (dx0,dy0,dz0) exactly;
    shape=0 collapses to the isotropic average; shape>1 exaggerates the
    hydrodynamic model's anisotropy. `rate` then uniformly scales the
    (possibly reshaped) tensor's mean eigenvalue -- the same "absolute
    timescale is a known approximation, shape is not" correction
    `calibrate_tau_scale` used to apply as a single tau_scale, generalized
    to also let the anisotropy itself flex.

    Returns (dx, dy, dz, dav, aniso_delta, taus) -- the calibrated
    eigenvalues plus everything `mode_amplitudes_raw` needs to build
    per-residue amplitudes against them.
    """
    dav0 = (dx0 + dy0 + dz0) / 3.0
    dx = rate * (dav0 + shape * (dx0 - dav0))
    dy = rate * (dav0 + shape * (dy0 - dav0))
    dz = rate * (dav0 + shape * (dz0 - dav0))
    taus, dav, aniso_delta = diffusion_tensor_taus(dx, dy, dz)
    return dx, dy, dz, dav, aniso_delta, taus


def calibrate_tensor(residues, dx0, dy0, dz0, d2, c2, gamma_h, gamma_x, omega_h, omega_x,
                     shape_bounds=(1.0, 1.0)):
    """
    Jointly refit the diffusion tensor's overall RATE and ANISOTROPY SHAPE
    (not orientation, which stays fixed at rotani()'s principal axes)
    against a presumed-rigid calibration set's R1/R2/NOE -- the same idea
    as modelfree4's own tensor optimization step, simplified to the two
    parameters (rate, shape) a single-field, single-structure calibration
    set can actually constrain (see `calibrated_tensor`).

    `residues` is a list of (l1, l2, l3, R1_obs, R1_err, R2_obs, R2_err,
    NOE_obs, NOE_err) direction-cosine tuples (NOE_obs/NOE_err may be
    None to fit R1/R2 only) -- direction cosines rather than precomputed
    amplitudes/taus because those depend on the trial tensor being
    calibrated here (see `makeshift.hydronmr.physics.nmr.mode_amplitudes_raw`).
    Including NOE matters for the same reason described in
    `fit_residue`'s docstring: with tau_e=0 here, NOE is independent of
    S2, so fitting it removes any systematic NOE offset before it's
    misread as widespread spurious tau_e need.

    `shape_bounds` defaults to (1.0, 1.0) -- i.e. rate-only calibration,
    reshaping disabled -- not out of caution alone: checked directly
    against real data (BMRB 4390, 20 calibration residues), letting shape
    range over e.g. [0, 3] finds a chi2 surface that varies by <1% across
    the whole span, so the grid/local-polish search just chases numerical
    noise and can land anywhere in the allowed range, including its
    bounds. A single-field, ~20-residue "presumed rigid" calibration set
    doesn't carry enough independent information to pin down anisotropy
    *shape* separately from rate (modelfree4 itself only attempts full
    tensor optimization with substantially more residues/experiments than
    that). Widen `shape_bounds` only if the calibration set backing a
    particular call is large and well-conditioned enough that its chi2
    surface actually has a resolvable minimum in shape -- check before
    trusting the result, the same way this default was chosen.

    Returns (rate, shape).
    """
    def chi2(params):
        log_rate, shape = params
        rate = 10.0 ** log_rate
        dx, dy, dz, dav, aniso_delta, taus = calibrated_tensor(dx0, dy0, dz0, rate, shape)
        total = 0.0
        for res in residues:
            l1, l2, l3, R1_obs, R1_err, R2_obs, R2_err = res[:7]
            NOE_obs = res[7] if len(res) > 7 else None
            NOE_err = res[8] if len(res) > 8 else None
            amplitudes = mode_amplitudes_raw(l1, l2, l3, dx, dy, dz, dav, aniso_delta)
            r1p, r2p, noep = relaxation_rates(
                d2, c2, gamma_h, gamma_x, omega_h, omega_x,
                amplitudes, taus, S2=1.0, tau_e=0.0)
            w1 = 1.0 / R1_err ** 2 if (R1_err and np.isfinite(R1_err) and R1_err > 0) else 1.0
            w2 = 1.0 / R2_err ** 2 if (R2_err and np.isfinite(R2_err) and R2_err > 0) else 1.0
            total += w1 * (R1_obs - r1p) ** 2 + w2 * (R2_obs - r2p) ** 2
            if NOE_obs is not None and np.isfinite(NOE_obs):
                w3 = 1.0 / NOE_err ** 2 if (NOE_err and np.isfinite(NOE_err) and NOE_err > 0) else 1.0
                total += w3 * (NOE_obs - noep) ** 2
        return total

    log_rate_grid = np.linspace(-1.0, 1.0, 21)   # rate in [0.1, 10]
    shape_grid = np.linspace(shape_bounds[0], shape_bounds[1], 13)
    best_cost = np.inf
    best = (0.0, 1.0)
    for lr in log_rate_grid:
        for sh in shape_grid:
            cost = chi2((lr, sh))
            if cost < best_cost:
                best_cost = cost
                best = (lr, sh)

    res = minimize(chi2, x0=best, method="Nelder-Mead",
                    bounds=[(-1.0, 1.0), shape_bounds],
                    options={"xatol": 1e-4, "fatol": 1e-8})
    log_rate, shape = res.x
    return 10.0 ** log_rate, shape


def assess_calibration(residues, rate, shape, dx0, dy0, dz0,
                        d2, c2, gamma_h, gamma_x, omega_h, omega_x,
                        gap_threshold=0.3):
    """
    Check whether the calibrated (rate, shape) tensor actually reconciles
    R1 and R2 for the calibration set, or is a forced compromise between
    two genuinely inconsistent constraints.

    R1(tau) has a peak and turns over; R2(tau) rises roughly monotonically
    in the same regime. A single calibrated tensor can only satisfy both
    if the calibration set is genuinely close to rigid for both
    observables at once. When it isn't -- most often because residues
    with real dynamics (conformational exchange elevating R2, or fast
    motion depressing it) got included in the "presumed rigid"
    NOE-selected calibration set -- `calibrate_tensor` still returns its
    best least-squares compromise, but that compromise silently favors
    whichever observable's loss is steeper near the optimum (usually R1,
    since it's past its peak there) while leaving the other
    systematically unexplained. That isn't a bug in the calibration; it's
    the calibration correctly reporting that no single overall tumbling
    tensor fits this protein, which downstream Model 1/3 S2 (linear in R1
    and R2 respectively) will silently inherit as a systematic bias
    rather than an obviously-wrong number.

    This computes, per calibration residue, an independent order
    parameter estimate from R1 alone (S2 = R1_obs / R1_rigid) and from R2
    alone (S2 = R2_obs / R2_rigid), both using the already-calibrated
    tensor. For a genuinely rigid residue these agree closely (both near
    1 -- the reason it was selected for calibration in the first place);
    a systematic gap between their medians across the whole calibration
    set is the diagnostic.

    `gap_threshold` default (0.3) is set empirically, not arbitrarily:
    checked against several already-validated BMRB entries (see
    demos/s2_validation), well-fit profiles cluster under |gap|~0.1 (one
    good fit reached 0.20), the weakest entry in that validation set sits
    at 0.47, and a profile with genuinely widespread exchange (BMRB 7036)
    hit 1.98 -- an order of magnitude past everything else. 0.3 sits
    cleanly between the good cluster and the flagged cases.

    Returns a dict: median_s2_from_r1, median_s2_from_r2, gap (r2 minus
    r1, signed), n, and unreliable (True if abs(gap) > gap_threshold).
    """
    dx, dy, dz, dav, aniso_delta, taus = calibrated_tensor(dx0, dy0, dz0, rate, shape)
    s2_r1, s2_r2 = [], []
    for res in residues:
        l1, l2, l3, R1_obs, R1_err, R2_obs, R2_err = res[:7]
        amplitudes = mode_amplitudes_raw(l1, l2, l3, dx, dy, dz, dav, aniso_delta)
        r1p, r2p, _ = relaxation_rates(
            d2, c2, gamma_h, gamma_x, omega_h, omega_x, amplitudes, taus,
            S2=1.0, tau_e=0.0)
        if R1_obs is not None and np.isfinite(R1_obs) and r1p > 0:
            s2_r1.append(R1_obs / r1p)
        if R2_obs is not None and np.isfinite(R2_obs) and r2p > 0:
            s2_r2.append(R2_obs / r2p)

    if not s2_r1 or not s2_r2:
        return dict(median_s2_from_r1=np.nan, median_s2_from_r2=np.nan,
                    gap=np.nan, n=0, unreliable=False)

    med_r1 = float(np.median(s2_r1))
    med_r2 = float(np.median(s2_r2))
    gap = med_r2 - med_r1
    return dict(median_s2_from_r1=med_r1, median_s2_from_r2=med_r2, gap=gap,
                n=min(len(s2_r1), len(s2_r2)), unreliable=abs(gap) > gap_threshold)


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
    # S2 is physically bounded to [0,1] -- fastmodelfree/modelfree4 enforce
    # this as a hard constraint on the fit itself (S2 gridsearch/optimization
    # bounds of 0/1.0 in the original perl script); the closed-form solution
    # here is the unconstrained optimum of the same convex objective, so
    # clamping it post hoc gives the identical constrained-optimum result,
    # not an approximation.
    S2_m1, S2_m1_err = _weighted_slope([
        (R1_obs, R1_err, R1_rigid),
        (R2_obs, R2_err, R2_rigid),
    ])
    if np.isfinite(S2_m1):
        S2_m1 = min(max(S2_m1, 0.0), 1.0)

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
    # Clamped to [0,1] for the same reason as S2_m1 above.
    r2_flag = False
    S2_from_R1 = R1_obs / R1_rigid if R1_rigid > 0 else np.nan
    if np.isfinite(S2_from_R1):
        S2_from_R1 = min(max(S2_from_R1, 0.0), 1.0)
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
            R1_obs, R1_err, R2_obs, R2_err, NOE_obs, NOE_err,
            d2, c2, gamma_h, gamma_x, omega_h, omega_x, amplitudes, taus)
        return ResidueFit(model="2", S2=S2, S2_err=S2_err,
                           tau_e_ps=tau_e_s * 1e12, NOE_pred_rigid=NOE_rigid,
                           noe_flag=True)

    if r2_flag:
        # Rex is a chemical-exchange broadening term -- physically >= 0
        # (same as fastmodelfree's own Rex lower bound of 0).
        Rex = max(R2_obs - S2_from_R1 * R2_rigid, 0.0)
        return ResidueFit(model="3", S2=S2_from_R1, S2_err=np.nan, Rex=Rex,
                           NOE_pred_rigid=NOE_rigid, r2_flag=True)

    return ResidueFit(model="1", S2=S2_m1, S2_err=S2_m1_err,
                       NOE_pred_rigid=NOE_rigid)


def _fit_model2(R1_obs, R1_err, R2_obs, R2_err, NOE_obs, NOE_err,
                 d2, c2, gamma_h, gamma_x, omega_h, omega_x, amplitudes, taus):
    """Model 2 (S2, tau_e): 1D bounded search over tau_e (the only place it
    enters nonlinearly), with S2 solved by exact weighted linear regression
    against R1 and R2 at each trial tau_e (both are affine in S2 for fixed
    tau_e: R(S2) = R_B(tau_e) + S2*(R_rigid - R_B(tau_e)), where R_B is the
    S2=0 endpoint at that tau_e).

    NOE is not part of that inner linear solve (it's a nonlinear ratio in
    S2 even at fixed tau_e -- see model_free module docstring), but it is
    part of the outer chi-square that picks tau_e: without it, tau_e/S2
    are only constrained by the same two numbers (R1, R2) Model 1 already
    uses, which is degenerate -- multiple (S2, tau_e) pairs fit R1/R2
    equally well, and nothing favors the tau_e->0 (Model 1) solution even
    when it's correct. Returns (S2, tau_e_seconds, S2_err)."""
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
        r1p, r2p, noep = relaxation_rates(
            d2, c2, gamma_h, gamma_x, omega_h, omega_x, amplitudes, taus,
            S2=S2, tau_e=tau_e)
        w1 = 1.0 / R1_err ** 2 if (R1_err and np.isfinite(R1_err) and R1_err > 0) else 1.0
        w2 = 1.0 / R2_err ** 2 if (R2_err and np.isfinite(R2_err) and R2_err > 0) else 1.0
        cost = w1 * (R1_obs - r1p) ** 2 + w2 * (R2_obs - r2p) ** 2
        if NOE_obs is not None and np.isfinite(NOE_obs):
            w3 = 1.0 / NOE_err ** 2 if (NOE_err and np.isfinite(NOE_err) and NOE_err > 0) else 1.0
            cost += w3 * (NOE_obs - noep) ** 2
        return cost

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
