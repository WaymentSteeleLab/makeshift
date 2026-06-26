"""
Core hydrodynamic-tensor routines.

Source: decompilations/hex-rays.txt
    covol_      11488  -- bead-pair hydrodynamic interaction tensor (Rotne-Prager-Yamakawa)
    eigen_       5370  -- eigen-decomposition of a symmetric 3x3 matrix
    hydro_      13635  -- central routine: assemble grand mobility matrix,
                          invert via spptrf_/spptri_ (routines/lapack.py),
                          sum to friction/diffusion tensors
    generdif66_  8280  -- assemble the full 6x6 generalized diffusion tensor
                          (3x3 translation, 3x3 rotation, 3x3 coupling)

NOTE on translation approach
-----------------------------
covol_ and eigen_ are, in the decompiled output, almost entirely x87
floating-point-stack bookkeeping (`fst7`, `fxch`, `fmulp`, dozens of
single-letter temporaries per expression) with no surviving variable
names or comments. A literal statement-by-statement port of that form
would be both unreadable and very likely to silently transpose an
operand somewhere in a 700-line function.

HYDRO-family programs (of which HYDRONMR is one; see Carrasco &
Garcia de la Torre, Garcia de la Torre et al.) are published, and the
underlying algorithms are standard:

  * covol_ computes the Rotne-Prager-Yamakawa (RPY) hydrodynamic
    interaction tensor T_ij between two beads i, j of radii a_i, a_j
    separated by r = r_j - r_i, in units of 1/(6 pi eta), used to
    build the off-diagonal 3x3 blocks of the grand mobility matrix
    (with the diagonal blocks being (1/a_i) * I from Stokes' law).

  * eigen_ diagonalizes a symmetric 3x3 matrix (the rotational
    diffusion tensor) to obtain principal axes / principal values.

These are reimplemented below directly from the standard formulas /
via numpy's robust symmetric eigensolver, rather than transliterated.
This is an algorithmic re-derivation of what those functions compute,
not a byte-for-byte port -- flagged here explicitly so it can be
checked against decompilations/hex-rays.txt if discrepancies show up
in validation against GROUND_TRUTH.
"""

import numpy as np
from scipy.linalg import lapack as _lapack


class LapackError(ValueError):
    """Raised in place of the original xerbla_'s STOP."""


def _xerbla(routine, info):
    raise LapackError(f"{routine}: illegal value in argument {info}")


def spptrf(uplo, n, ap):
    """Packed-symmetric Cholesky factorization (LAPACK SPPTRF/DPPTRF).
    Returns (factor, info); info > 0 means not positive definite."""
    if n < 0:
        _xerbla("SPPTRF", 2)
    fn = _lapack.dpptrf if ap.dtype == np.float64 else _lapack.spptrf
    return fn(n, ap, lower=(uplo.upper() == "L"))


def spptri(uplo, n, ap):
    """Inverse of a symmetric positive-definite matrix from its packed
    Cholesky factor (LAPACK SPPTRI/DPPTRI). Returns (ainv, info)."""
    if n < 0:
        _xerbla("SPPTRI", 2)
    fn = _lapack.dpptri if ap.dtype == np.float64 else _lapack.spptri
    ainv, info = fn(n, ap, lower=(uplo.upper() == "L"))
    if info < 0:
        _xerbla("SPPTRI", -info)
    return ainv, info


def eigen(matrix: np.ndarray):
    """Port of eigen_ (hex-rays.txt:5370): eigenvalues/eigenvectors of a
    symmetric 3x3 matrix, returned in descending order of eigenvalue
    (matching the convention used downstream by rotani_/oblate_)."""
    m = np.asarray(matrix, dtype=np.float64)
    vals, vecs = np.linalg.eigh(m)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def build_mobility_matrix(positions: np.ndarray, radii: np.ndarray, ind: int = 3) -> np.ndarray:
    """Assemble the 3N x 3N grand mobility matrix B used inside hydro_
    (hex-rays.txt:13635): diagonal 3x3 blocks (1/a_i) I from Stokes'
    law, off-diagonal 3x3 blocks T_ij from covol_oseen_rpy.

    Units: 1/(6*pi*eta). Caller multiplies by 1/(6*pi*eta) (eta =
    state.viscosity) to get physical mobility.

    Vectorized (numpy broadcasting) equivalent of the pairwise
    covol_oseen_rpy loop -- O(N^2) array ops instead of O(N^2) Python
    calls, which matters since N can be ~1500-2000 beads.
    """
    positions = np.asarray(positions, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    n = len(radii)
    eye3 = np.eye(3)

    # Pairwise differences/distances, shape (N,N,3) / (N,N)
    rij = positions[:, None, :] - positions[None, :, :]
    r2 = np.sum(rij * rij, axis=-1)
    np.fill_diagonal(r2, 1.0)  # avoid div-by-zero; diagonal overwritten later
    r = np.sqrt(r2)

    ai = radii[:, None]
    aj = radii[None, :]

    # rr_hat outer products, shape (N,N,3,3)
    rr_hat = rij[:, :, :, None] * rij[:, :, None, :] / r2[:, :, None, None]

    # --- LABEL_42 (non-overlap RPY) branch ---
    a2 = (ai * ai + aj * aj) / (3.0 * r2)
    coef = 6.0 / (8.0 * r)
    t_nonoverlap = coef[:, :, None, None] * (
        (1.0 + ind * a2)[:, :, None, None] * eye3
        + (1.0 - 3.0 * ind * a2)[:, :, None, None] * rr_hat
    )

    if ind == 1:
        # --- overlap branch: equivalent radius sigma = ((ai^3+aj^3)/2)^(1/3) ---
        sigma = ((ai**3 + aj**3) / 2.0) ** (1.0 / 3.0)
        diag_term = 1.0 / sigma - 9.0 * r / (32.0 * sigma * sigma)
        rr_overlap = rij[:, :, :, None] * rij[:, :, None, :] * (
            3.0 / (32.0 * sigma * sigma * r)
        )[:, :, None, None]
        t_overlap = diag_term[:, :, None, None] * eye3 + rr_overlap

        overlap_mask = (ai + aj) >= 1.01 * r
        t = np.where(overlap_mask[:, :, None, None], t_overlap, t_nonoverlap)
    else:
        t = t_nonoverlap

    # Assemble into 3N x 3N: t has shape (N,N,3,3) -> (N,3,N,3) -> (3N,3N)
    b = t.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)

    # Overwrite diagonal blocks with (1/a_i) I, zero everything else on
    # the diagonal blocks (the i==j entries above used r=1 placeholder).
    for i in range(n):
        b[3*i:3*i+3, 3*i:3*i+3] = eye3 / radii[i]

    return b


def invert_packed_symmetric(matrix: np.ndarray):
    """Pack a symmetric n x n matrix into LAPACK lower-packed form,
    invert via spptrf_/spptri_ (routines/lapack.py), and return the
    full n x n inverse. This mirrors the spptrf_/spptri_ call pair in
    hydro_ (hex-rays.txt:13635)."""
    n = matrix.shape[0]
    # LAPACK lower-packed (column-major): for column j, elements
    # (j,j)..(n-1,j) consecutively. np.tril_indices is row-major (it
    # yields, for each row, all columns <= row), which is the wrong
    # order; build the column-major lower-triangle index arrays via
    # np.triu_indices on the transpose (equivalent to column-major
    # traversal of the lower triangle) -- vectorized, O(n^2) array ops
    # instead of an O(n^2) Python double loop.
    # np.triu_indices(n) gives (row_idx, col_idx) with row_idx<=col_idx,
    # ordered by row_idx (outer) then col_idx (inner) ascending --
    # setting j=row_idx, i=col_idx gives exactly the (i>=j, j outer
    # ascending, i inner ascending) traversal LAPACK packed-lower wants.
    j_idx, i_idx = np.triu_indices(n)
    ap = matrix[i_idx, j_idx].astype(np.float64)

    chol, info = spptrf("L", n, ap.copy())
    if info != 0:
        raise ValueError(f"spptrf_: matrix not positive definite (info={info})")
    ainv_packed, info = spptri("L", n, chol.copy())
    if info != 0:
        raise ValueError(f"spptri_: inversion failed (info={info})")

    full = np.zeros((n, n))
    full[i_idx, j_idx] = ainv_packed
    full[j_idx, i_idx] = ainv_packed
    return full


def hydro(positions: np.ndarray, radii: np.ndarray, viscosity: float,
          center: np.ndarray = None, ind: int = 3):
    """Port of hydro_ (hex-rays.txt:13635): build the grand mobility
    matrix, invert it to get the grand friction matrix C (3N x 3N),
    then sum the 3x3 blocks of C about `center` (default: centroid)
    to get the 3x3 translational friction tensor Ct, 3x3 rotational
    friction tensor Cr, and 3x3 translation-rotation coupling Cc.

    Returns (Ct, Cr, Cc, C) where C is the full 3N x 3N friction
    matrix (used by generdif66_).
    """
    positions = np.asarray(positions, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    n = len(radii)

    if center is None:
        center = positions.mean(axis=0)

    b = build_mobility_matrix(positions, radii, ind=ind)
    b *= 1.0 / (6.0 * np.pi * viscosity)

    c = invert_packed_symmetric(b)

    ct = np.zeros((3, 3))
    cr = np.zeros((3, 3))
    cc = np.zeros((3, 3))

    def skew(r):
        return np.array([[0, -r[2], r[1]],
                          [r[2], 0, -r[0]],
                          [-r[1], r[0], 0]])

    def assemble(origin):
        # S: (N,3,3) skew matrices of (positions[i] - origin)
        rel = positions - origin
        s = np.zeros((n, 3, 3))
        s[:, 0, 1] = -rel[:, 2]; s[:, 0, 2] = rel[:, 1]
        s[:, 1, 0] = rel[:, 2];  s[:, 1, 2] = -rel[:, 0]
        s[:, 2, 0] = -rel[:, 1]; s[:, 2, 1] = rel[:, 0]

        # S_flat[(i*3+c), a] = s[i,a,c]  -- flattened "block-row" form,
        # so that block-matrix contractions become plain (3xN3)@(N3xN3)
        # @(N3x3) matmuls (BLAS), avoiding any (N,N,3,3)-sized tensor.
        s_flat = s.transpose(0, 2, 1).reshape(3 * n, 3)

        # Ct = sum_{i,j} C_ij : sum c's row-blocks then col-blocks.
        ct_ = c.reshape(n, 3, 3 * n).sum(axis=0).reshape(3, n, 3).sum(axis=1)

        # Cc = sum_{i,j} S_i @ C_ij
        #    -> Cc[a,b] = sum_{i,c,j} s[i,a,c] * c[3i+c, 3j+b]
        # c_colsum[3i+c, b] = sum_j c[3i+c, 3j+b]
        c_colsum = c.reshape(3 * n, n, 3).sum(axis=1)  # (3n, 3)
        cc_ = s_flat.T @ c_colsum

        # Cr = sum_{i,j} S_i @ C_ij @ S_j^T
        #    -> Cr[a,b] = sum_{i,c,j,d} s[i,a,c] * c[3i+c,3j+d] * s[j,b,d]
        cr_ = s_flat.T @ (c @ s_flat)

        return ct_, cr_, cc_

    ct, cr, cc = assemble(center)

    # Shift to the "center of diffusion" (a.k.a. center of resistance):
    # the origin about which the rotation-translation coupling tensor
    # Cc is symmetric / Tr(Cr) is minimized. Per Garcia de la Torre &
    # Bloomfield (1977), the shift vector d (from `center` to the
    # center of diffusion) solves A d = b, with
    #     A = Tr(Ct) I - Ct
    #     b_k = sum_{i,j} epsilon_kij Cc_ij  (i.e. the "vector" formed
    #           from the antisymmetric part of Cc)
    basis = [skew(np.eye(3)[k]) for k in range(3)]
    A = np.array([[np.trace(basis[k] @ ct @ basis[l].T) for l in range(3)]
                   for k in range(3)])
    b = -0.5 * np.array([np.trace(cc @ basis[k]) - np.trace(basis[k] @ cc.T)
                          for k in range(3)])
    d = np.linalg.solve(A, b)
    center_of_diffusion = center + d

    ct, cr, cc = assemble(center_of_diffusion)

    return ct, cr, cc, c, center_of_diffusion


def generdif66(ct: np.ndarray, cr: np.ndarray, cc: np.ndarray, temperature: float):
    """Port of generdif66_ (hex-rays.txt:8280): assemble the full 6x6
    friction supermatrix

        Xi = [[Ct, Cc^T],
              [Cc, Cr ]]

    and the corresponding 6x6 diffusion supermatrix D = kT * Xi^-1
    (Boltzmann constant k = 1.380649e-16 erg/K, lengths in cm, so D
    comes out in cgs units as in the original Fortran)."""
    k_boltzmann = 1.380649e-16  # erg/K

    xi = np.zeros((6, 6))
    xi[0:3, 0:3] = ct
    xi[3:6, 3:6] = cr
    xi[0:3, 3:6] = cc.T
    xi[3:6, 0:3] = cc

    d = k_boltzmann * temperature * np.linalg.inv(xi)
    return d, xi


def generdif66_state(g):
    """`generdif66` variant operating on a HydronmrState `g`: reads
    g.friction_translation/rotation/coupling and g.temperature, and
    stores the resulting 6x6 diffusion tensor (and its 3x3 trans/rot
    blocks) back onto g.dt_trans/g.dt_rot."""
    d, _xi = generdif66(g.friction_translation, g.friction_rotation,
                         g.friction_coupling, g.temperature)
    g.diffusion_66 = d
    g.dt_trans = d[0:3, 0:3]
    g.dt_rot = d[3:6, 3:6]
