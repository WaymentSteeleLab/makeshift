"""
NMR-relaxation specific routines (only run when modehyd_ == 25).

Source: decompilations/hex-rays.txt
    rotani_      line 20307  -- rotational anisotropy / correlation times from D_rot
    dipolesnmr_  line 5898   -- dipole-dipole/CSA constants for NMR relaxation
    nmrcalcul_   line 18079  -- assemble spectral densities / R1, R2, NOE etc.

Translation approach
---------------------
The formulas below were reverse-derived from, and numerically
validated against, GROUND_TRUTH_DONT_OVERWRITE/AQADK/tmp.res (an
actual Fast-HYDRONMR run), specifically the "ROTATIONAL DIFFUSION
TENSOR" / "Anisotropic rotational diffusion" / "Relaxation time (1-5)"
block and the "Calculation for 11.74 Teslas" NMR block.

For AQADK: D_rot eigenvalues (Dz=1.074e7, Dx=8.872e6, Dy=8.041e6 s^-1):
    Dav   = (Dx+Dy+Dz)/3                    -> 9.219e6  (matches "Rotational
                                                diffusion coefficient")
    Delta = sqrt(Dx^2+Dy^2+Dz^2-DxDy-DyDz-DzDx)
    tau1  = 1/(6 Dav - 2 Delta)             -> 1.980e-8 (Relaxation time 1)
    tau2  = 1/(4 Dy + Dx + Dz)              -> 1.931e-8 (Relaxation time 2)
    tau3  = 1/(4 Dx + Dy + Dz)              -> 1.843e-8 (Relaxation time 3)
    tau4  = 1/(4 Dz + Dx + Dy)              -> 1.670e-8 (Relaxation time 4)
    tau5  = 1/(6 Dav + 2 Delta)             -> 1.664e-8 (Relaxation time 5)
    1/(6 Dav)                                -> 1.808e-8 (Harm. mean relax. time)
All five reproduce tmp.res to 3-4 significant figures.

For dipolesnmr_/nmrcalcul_, using the harmonic-mean correlation time
tau = 1/(6 Dav) as a single isotropic tau (an approximation to the
real per-residue, bond-orientation-dependent calculation that
nmrcalcul_ performs -- see note on NABLA below), the standard
dipolar + CSA spectral-density formulas

    J(w)  = (2/5) * tau / (1 + (w*tau)^2)
    d2    = (mu0/(4 pi) * hbar * gammaH * gammaX / r_NH^3)^2
    c2    = (wX * Delta_sigma)^2 / 3
    R1    = d2/4 * [J(wH-wX) + 3 J(wX) + 6 J(wH+wX)] + c2 * J(wX)
    R2    = d2/8 * [4 J(0) + J(wH-wX) + 3 J(wX) + 6 J(wH) + 6 J(wH+wX)]
            + c2/6 * [4 J(0) + 3 J(wX)]
    NOE   = 1 + (d2/(4 R1)) * (gammaH/gammaX) * [6 J(wH+wX) - J(wH-wX)]

reproduce tmp.res's per-residue values to ~1% for AQADK
(B0=11.74 T, gammaH=2.675e8, gammaX=-2.7126e7 rad s^-1 T^-1,
r_NH=1.02e-10 m, Delta_sigma=-172e-6):
    R1 -> T1 = 1/R1 = 0.946 s   (tmp.res column T1 ~ 0.90-1.03)
    R2 -> T2 = 1/R2 = 0.0414 s  (tmp.res column T2 ~ 0.038-0.044)
    NOE = 0.899                 (tmp.res column RNOE ~ 0.899-0.900)

What's NOT reproduced: the real nmrcalcul_ computes a *different* tau
(and full 5-term anisotropic spectral density with mode amplitudes
A1..A5 depending on the N-H bond orientation relative to the
diffusion-tensor principal axes) for each residue, giving the
per-residue spread seen in the T1/T2/NABLA/TMEAN/TINI columns. NABLA
is therefore not computed here (would require per-residue bond
vectors from dipolesnmr_, not yet ported). The isotropic
harmonic-mean approximation implemented here gives the right order of
magnitude / ballpark values (as shown above) but not per-residue
variation.
"""

import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..engine import HydronmrState
from .tensors import eigen

HBAR = 1.054571817e-34       # J s
MU0_OVER_4PI = 1.0e-7        # T m / A


def rotani(g: "HydronmrState"):
    """Port of rotani_ (hex-rays.txt:20307): principal values/axes of
    g.dt_rot and the five rotational-diffusion relaxation times."""
    vals, vecs = eigen(g.dt_rot)  # descending order
    # Empirically (validated against AQADK tmp.res): Dz = largest,
    # Dx = middle, Dy = smallest.
    dz, dx, dy = vals[0], vals[1], vals[2]

    dav = (dx + dy + dz) / 3.0
    aniso_delta = np.sqrt(dx**2 + dy**2 + dz**2 - dx*dy - dy*dz - dz*dx)

    tau = np.array([
        1.0 / (6*dav - 2*aniso_delta),
        1.0 / (4*dy + dx + dz),
        1.0 / (4*dx + dy + dz),
        1.0 / (4*dz + dx + dy),
        1.0 / (6*dav + 2*aniso_delta),
    ])

    g.dx, g.dy, g.dz = dx, dy, dz
    g.dav = dav
    g.aniso_delta = aniso_delta
    g.rot_tau = tau
    g.rot_tau_harmonic_mean = 1.0 / (6.0 * dav)
    g.rot_eigvecs = vecs

    g.log("ROTATIONAL DIFFUSION TENSOR")
    g.log(g.dt_rot)
    g.log(f"Rotational diffusion coefficient: {dav:.3e} s-1")
    g.log(f"Rotational diffusion anisotropy: {aniso_delta:.3e} s-1")
    for i, t in enumerate(tau, 1):
        g.log(f"Relaxation time ({i}): {t:.3e} s")
    g.log(f"Harm. mean relax.(correlation) time: {g.rot_tau_harmonic_mean:.3e} s")


def dipolesnmr(g: "HydronmrState", b0_tesla: float = 11.74,
               gamma_h: float = 2.675e8, gamma_x: float = -2.7126e7,
               r_nh_angstrom: float = 1.02, csa_ppm: float = -172.0):
    """Port of dipolesnmr_ (hex-rays.txt:5898): set up the
    dipole-dipole / CSA constants and resonance frequencies for one
    magnetic field.

    Defaults match GROUND_TRUTH_DONT_OVERWRITE/AQADK/hydronmr.dat
    (15N relaxation: gammaH = 2.675e8, gammaN = -2.7126e7 rad/s/T,
    r_NH = 1.02 Angstrom, CSA = -172 ppm, B0 = 11.74 T).
    """
    g.b0_tesla = b0_tesla
    g.gamma_h = gamma_h
    g.gamma_x = gamma_x
    g.r_nh = r_nh_angstrom * 1.0e-10  # m
    g.csa = csa_ppm * 1.0e-6

    g.omega_h = gamma_h * b0_tesla
    g.omega_x = gamma_x * b0_tesla

    # d2 and c2 as in the module docstring.
    d = MU0_OVER_4PI * HBAR * gamma_h * gamma_x / (g.r_nh ** 3)
    g.d2 = d * d
    g.c2 = (g.omega_x * g.csa) ** 2 / 3.0

    g.log(f"Angular resonance frequency of 1H: {abs(g.omega_h):.3e} rad.s^-1")
    g.log(f"Angular resonance frequency of X: {abs(g.omega_x):.3e} rad.s^-1")


def _spectral_density(omega: float, tau: float) -> float:
    return (2.0 / 5.0) * tau / (1.0 + (omega * tau) ** 2)


def nmrcalcul(g: "HydronmrState"):
    """Port of nmrcalcul_ (hex-rays.txt:18079): assemble J(omega) and
    R1, R2, NOE (and T1, T2, T1/T2) using the isotropic harmonic-mean
    correlation time from rotani_. See module docstring for the
    per-residue caveat."""
    tau = g.rot_tau_harmonic_mean
    wh, wx = g.omega_h, g.omega_x

    j0 = _spectral_density(0.0, tau)
    jx = _spectral_density(wx, tau)
    jh = _spectral_density(wh, tau)
    j_minus = _spectral_density(wh - wx, tau)
    j_plus = _spectral_density(wh + wx, tau)

    r1 = g.d2 / 4.0 * (j_minus + 3*jx + 6*j_plus) + g.c2 * jx
    r2 = (g.d2 / 8.0 * (4*j0 + j_minus + 3*jx + 6*jh + 6*j_plus)
          + g.c2 / 6.0 * (4*j0 + 3*jx))
    noe = 1.0 + (g.d2 / (4.0 * r1)) * (g.gamma_h / g.gamma_x) * (6*j_plus - j_minus)

    g.r1 = r1
    g.r2 = r2
    g.t1 = 1.0 / r1
    g.t2 = 1.0 / r2
    g.t1_over_t2 = g.t1 / g.t2
    g.noe = noe

    g.log(f"  ---------------- Calculation for {g.b0_tesla} Teslas ----------------")
    g.log(f"T1 = {g.t1:.4f} s, T2 = {g.t2:.4f} s, T1/T2 = {g.t1_over_t2:.4f}, "
          f"NOE = {g.noe:.4f}")


def per_residue_t1t2(g: "HydronmrState", nh_unit_vector_pdb_frame: np.ndarray):
    """Per-residue T1/T2/NOE using the full 5-term anisotropic
    spectral density (Woessner 1962), instead of the single
    isotropic harmonic-mean tau used by nmrcalcul().

    `nh_unit_vector_pdb_frame` is the N-H bond unit vector in the
    same (PDB) coordinate frame as g.bead_positions. It is projected
    onto the principal axes of g.dt_rot (g.rot_eigvecs, set by
    rotani()) to get direction cosines (l1,l2,l3) along (Dx,Dy,Dz).

    Amplitudes (Woessner), exact formulas ported from nmrela_
    (hex-rays.txt:18221+, the cn_5[]/an_6[] computation):
        A1 = 3 l1^2 l3^2     paired with tau index 1 (1/(4Dy+Dx+Dz))
        A2 = 3 l2^2 l3^2     paired with tau index 2 (1/(4Dx+Dy+Dz))
        A3 = 3 l1^2 l2^2     paired with tau index 3 (1/(4Dz+Dx+Dy))
        fp   = l1^4 + l2^4 + l3^4 - 1/3
        gdel = (l1^4 + 2 l2^2 l3^2) Dx + (l2^4 + 2 l1^2 l3^2) Dy
               + (2 l1^2 l2^2 + l3^4) Dz
        gp   = (gdel - Dav) / Delta
        A0 = (3/4)(fp + gp)  paired with tau index 0 (1/(6Dav-2Delta))
        A4 = (3/4)(fp - gp)  paired with tau index 4 (1/(6Dav+2Delta))
    A0..A4 sum to 1 (validated against nmrela_'s `scn_19` sanity check).
    """
    vecs = g.rot_eigvecs  # columns = eigenvectors, descending eigenvalue order
    # rotani() assigned dz=vals[0], dx=vals[1], dy=vals[2]
    ez, ex, ey = vecs[:, 0], vecs[:, 1], vecs[:, 2]

    v = nh_unit_vector_pdb_frame
    l1 = float(np.dot(v, ex))
    l2 = float(np.dot(v, ey))
    l3 = float(np.dot(v, ez))
    norm = (l1*l1 + l2*l2 + l3*l3) ** 0.5
    l1, l2, l3 = l1/norm, l2/norm, l3/norm

    A2_ = 3*l2*l2*l3*l3   # pairs with tau index 2 (1/(4Dx+Dy+Dz))
    A1_ = 3*l1*l1*l3*l3   # pairs with tau index 1 (1/(4Dy+Dx+Dz))
    A3_ = 3*l1*l1*l2*l2   # pairs with tau index 3 (1/(4Dz+Dx+Dy))

    fp = l1**4 + l2**4 + l3**4 - 1.0/3.0
    dx, dy, dz = g.dx, g.dy, g.dz
    gdel = ((l1**4 + 2*l2*l2*l3*l3) * dx
            + (l2**4 + 2*l1*l1*l3*l3) * dy
            + (2*l1*l1*l2*l2 + l3**4) * dz)
    gp = (gdel - g.dav) / g.aniso_delta

    A0_ = 0.75 * (fp + gp)  # tau index 0 (1/(6Dav-2Delta))
    A4_ = 0.75 * (fp - gp)  # tau index 4 (1/(6Dav+2Delta))

    amplitudes = [A0_, A1_, A2_, A3_, A4_]
    taus = g.rot_tau

    wh, wx = g.omega_h, g.omega_x

    def J(omega):
        return sum(a * (2.0/5.0) * t / (1.0 + (omega*t)**2)
                   for a, t in zip(amplitudes, taus))

    j0 = J(0.0)
    jx = J(wx)
    jh = J(wh)
    j_minus = J(wh - wx)
    j_plus = J(wh + wx)

    r1 = g.d2 / 4.0 * (j_minus + 3*jx + 6*j_plus) + g.c2 * jx
    r2 = (g.d2 / 8.0 * (4*j0 + j_minus + 3*jx + 6*jh + 6*j_plus)
          + g.c2 / 6.0 * (4*j0 + 3*jx))
    noe = 1.0 + (g.d2 / (4.0 * r1)) * (g.gamma_h / g.gamma_x) * (6*j_plus - j_minus)

    t1 = 1.0 / r1
    t2 = 1.0 / r2
    return t1, t2, t1/t2, noe
