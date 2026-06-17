import zipfile
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.optimize import least_squares
from collections import defaultdict
from tqdm import tqdm

from ..io.io import estimate_background
from .config import _yaml_set_cache


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_save(fit_results: pd.DataFrame, cache_path: Path) -> None:
    json_str = fit_results.to_json(orient="records")
    with zipfile.ZipFile(cache_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("fit_lineshapes.json", json_str)
    print(f"  Saved lineshape cache → {cache_path.name}")


def _cache_load(cache_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(cache_path, "r") as zf:
        json_str = zf.read("fit_lineshapes.json").decode()
    df = pd.read_json(json_str, orient="records")
    for col in ("ref_index", "plane"):
        if col in df.columns:
            df[col] = df[col].astype(int)
    if "converged" in df.columns:
        df["converged"] = df["converged"].astype(bool)
    print(f"  {len(df)} rows loaded from cache  "
          f"({df['ref_index'].nunique()} peaks × {df['plane'].nunique()} planes)")
    return df


# ---------------------------------------------------------------------------
# Lineshape math
# ---------------------------------------------------------------------------

def _gaussian_1d(x, x0, lw):
    sig = lw / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-0.5 * ((x - x0) / sig) ** 2)


def _lorentzian_1d(x, x0, lw):
    hwhm = lw / 2
    return hwhm**2 / ((x - x0)**2 + hwhm**2)


def _lineshape_1d(x, x0, lw, kind):
    return _gaussian_1d(x, x0, lw) if kind == "gaussian" else _lorentzian_1d(x, x0, lw)


def _analytical_volume_1d(lw, kind):
    if kind == "gaussian":
        return lw / (2 * np.sqrt(2 * np.log(2))) * np.sqrt(2 * np.pi)
    return np.pi * (lw / 2)


def _peak_shapes(rows, cols, peak_params, kind):
    """Vectorised 2D lineshape: (n_peaks, n_pix)."""
    r0 = peak_params[:, 0, np.newaxis]
    c0 = peak_params[:, 1, np.newaxis]
    lw_n = peak_params[:, 2, np.newaxis]
    lw_h = peak_params[:, 3, np.newaxis]
    return _lineshape_1d(rows, r0, lw_n, kind) * _lineshape_1d(cols, c0, lw_h, kind)


def _find_overlap_groups(peaks_df, n_col="N_axis", h_col="H_axis",
                         n_lw_col="N_lw", h_lw_col="H_lw",
                         kind="lorentzian", overlap_fraction=0.05):
    """
    Union-find overlap detection using lineshape contribution criterion.

    Two peaks are grouped when either one contributes more than overlap_fraction
    of its own peak height at the other's center position.
    """
    r0s = peaks_df[n_col].to_numpy()
    c0s = peaks_df[h_col].to_numpy()
    lw_ns = peaks_df[n_lw_col].to_numpy()
    lw_hs = peaks_df[h_lw_col].to_numpy()
    n = len(peaks_df)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def contribution(dr, dc, lw_n, lw_h):
        if kind == "lorentzian":
            fn = (lw_n / 2)**2 / (dr**2 + (lw_n / 2)**2)
            fh = (lw_h / 2)**2 / (dc**2 + (lw_h / 2)**2)
        else:
            sig_n = lw_n / (2 * np.sqrt(2 * np.log(2)))
            sig_h = lw_h / (2 * np.sqrt(2 * np.log(2)))
            fn = np.exp(-0.5 * (dr / sig_n) ** 2)
            fh = np.exp(-0.5 * (dc / sig_h) ** 2)
        return fn * fh

    for i in range(n):
        for j in range(i + 1, n):
            dr = abs(r0s[i] - r0s[j])
            dc = abs(c0s[i] - c0s[j])
            if max(contribution(dr, dc, lw_ns[i], lw_hs[i]),
                   contribution(dr, dc, lw_ns[j], lw_hs[j])) > overlap_fraction:
                parent[find(i)] = find(j)

    grp = defaultdict(list)
    for i in range(n):
        grp[find(i)].append(i)
    return list(grp.values())


def _fit_group(data_planes, peak_rows, peak_cols, lw_n_init, lw_h_init,
               noises, kind="gaussian", radius_pts=(20, 8)):
    n_peaks, n_planes = len(peak_rows), len(data_planes)
    nr, nc = data_planes[0].shape
    r_rad, c_rad = radius_pts
    r_min = max(0, int(min(peak_rows)) - r_rad)
    r_max = min(nr, int(max(peak_rows)) + r_rad + 1)
    c_min = max(0, int(min(peak_cols)) - c_rad)
    c_max = min(nc, int(max(peak_cols)) + c_rad + 1)
    rg = np.arange(r_min, r_max, dtype=float)
    cg = np.arange(c_min, c_max, dtype=float)
    rows_flat, cols_flat = [a.ravel() for a in np.meshgrid(rg, cg, indexing="ij")]
    obs = np.stack([data_planes[p][r_min:r_max, c_min:c_max].ravel()
                    for p in range(n_planes)])

    def unpack(params):
        return (params[:4 * n_peaks].reshape(n_peaks, 4),
                params[4 * n_peaks:].reshape(n_planes, n_peaks))

    def residuals(params):
        shared, amps = unpack(params)
        pred = amps @ _peak_shapes(rows_flat, cols_flat, shared, kind)
        return ((pred - obs) / noises[:, np.newaxis]).ravel()

    amp0 = np.array([[max(data_planes[p][np.clip(int(round(r)), 0, nr - 1),
                                        np.clip(int(round(c)), 0, nc - 1)], noises[p])
                      for r, c in zip(peak_rows, peak_cols)]
                     for p in range(n_planes)])
    p0 = np.concatenate([np.column_stack([peak_rows, peak_cols,
                                          lw_n_init, lw_h_init]).ravel(),
                         amp0.ravel()])
    lo = np.concatenate([np.column_stack([peak_rows - r_rad / 2, peak_cols - c_rad / 2,
                                          np.full(n_peaks, 0.5),
                                          np.full(n_peaks, 0.5)]).ravel(),
                         np.zeros(n_planes * n_peaks)])
    hi = np.concatenate([np.column_stack([peak_rows + r_rad / 2, peak_cols + c_rad / 2,
                                          np.full(n_peaks, r_rad * 2.),
                                          np.full(n_peaks, c_rad * 2.)]).ravel(),
                         np.full(n_planes * n_peaks, np.inf)])
    try:
        res = least_squares(residuals, p0, bounds=(lo, hi), method="trf",
                            ftol=1e-6, xtol=1e-6, max_nfev=2000 * len(p0))
        converged = res.success or res.cost < 1e6
    except Exception as e:
        return dict(converged=False, r0=peak_rows, c0=peak_cols,
                    lw_n=lw_n_init, lw_h=lw_h_init,
                    amps=amp0, vols=np.full((n_planes, n_peaks), np.nan), msg=str(e))

    shared_fit, amps_fit = unpack(res.x)
    v_n = np.array([_analytical_volume_1d(lw, kind) for lw in shared_fit[:, 2]])
    v_h = np.array([_analytical_volume_1d(lw, kind) for lw in shared_fit[:, 3]])
    return dict(converged=converged, msg=res.message,
                r0=shared_fit[:, 0], c0=shared_fit[:, 1],
                lw_n=shared_fit[:, 2], lw_h=shared_fit[:, 3],
                amps=amps_fit, vols=amps_fit * (v_n * v_h), cost=float(res.cost))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_peaks(planes, ref_df, uc, config, lineshape="gaussian",
              radius_ppm=(0.5, 0.04),
              overlap_fraction=0.05,
              n_col="N_axis", h_col="H_axis",
              n_lw_col="N_lw", h_lw_col="H_lw",
              cache_dir=None, overwrite_cache=False):
    """
    Fit lineshapes across all planes simultaneously (PINT-style).

    Peaks are grouped by lineshape contribution overlap (see overlap_fraction),
    then each group is fit jointly across all planes with shared positions and
    linewidths and per-plane amplitudes.

    Caching: checks for an existing cache in this order:
      1. fit_lineshapes_cache key in the YAML file
      2. default cache file (<stem>_fit_lineshapes.json.zip) in cache_dir
         (default: next to the YAML)
    After a successful fit the cache is saved and fit_lineshapes_cache is
    written into the YAML automatically.

    Parameters
    ----------
    planes : list of ndarray
        planes[0] is the reference; planes[1:] are CPMG planes.
    ref_df : DataFrame
        Output of spectrum.pick_peaks, with a ref_index column.
    uc : list
        Unit-conversion objects from spectrum.read_ucsf.
    config : dict
        Output of load_config.
    lineshape : {'lorentzian', 'gaussian'}
    radius_ppm : tuple of float
        (N, H) integration window half-width in ppm.
    overlap_fraction : float
        Lineshape contribution threshold for grouping peaks (0–1).
    cache_dir : str or Path, optional
        Directory for the default <stem>_fit_lineshapes.json.zip cache file.
        Defaults to yaml_path.parent.
    overwrite_cache : bool
        If True, re-fit even if a cached result exists.

    Returns
    -------
    DataFrame with columns: ref_index, plane, vol, vol_err, N_ppm, H_ppm,
    lw_n, lw_h, converged.
    """
    yaml_path = config["yaml_path"]
    cache_dir = Path(cache_dir) if cache_dir is not None else yaml_path.parent

    # ── cache check ─────────────────────────────────────────────────────────
    if not overwrite_cache:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        if "fit_lineshapes_cache" in cfg:
            configured = Path(cfg["fit_lineshapes_cache"])
            if configured.exists():
                print(f"  Loading lineshape cache: {configured.name}")
                return _cache_load(configured)

        default_cache = cache_dir / f"{yaml_path.stem}_fit_lineshapes.json.zip"
        if default_cache.exists():
            print(f"  Loading lineshape cache: {default_cache.name}")
            result = _cache_load(default_cache)
            _yaml_set_cache(yaml_path, default_cache)
            return result

    nr, nc = planes[0].shape
    n_planes = len(planes)
    r_rad = max(int(round(abs(uc[0].f(radius_ppm[0], "ppm") - uc[0].f(0, "ppm")))), 4)
    c_rad = max(int(round(abs(uc[1].f(radius_ppm[1], "ppm") - uc[1].f(0, "ppm")))), 4)
    radius_pts = (r_rad, c_rad)

    peaks_df = ref_df.reset_index(drop=True)
    groups = _find_overlap_groups(peaks_df, n_col, h_col, n_lw_col, h_lw_col,
                                   kind=lineshape, overlap_fraction=overlap_fraction)

    noises = np.array([estimate_background(p) for p in planes])
    print(f"  {len(peaks_df)} peaks | {len(groups)} groups "
          f"({sum(len(g) > 1 for g in groups)} overlapping)")
    print(f"  noise per plane: {noises.round(1)}")

    rows_out = []
    for group in tqdm(groups):
        sub = peaks_df.iloc[group]
        fit = _fit_group(planes,
                         sub[n_col].to_numpy(float),
                         sub[h_col].to_numpy(float),
                         sub[n_lw_col].to_numpy(float).clip(min=1.0, max=r_rad * 2.0),
                         sub[h_lw_col].to_numpy(float).clip(min=1.0, max=c_rad * 2.0),
                         noises, kind=lineshape, radius_pts=radius_pts)

        if not fit["converged"]:
            print(f"  WARNING group {group}: {fit['msg']}")

        for ki, peak_idx in enumerate(group):
            v_n = _analytical_volume_1d(float(fit["lw_n"][ki]), lineshape)
            v_h = _analytical_volume_1d(float(fit["lw_h"][ki]), lineshape)
            for p in range(n_planes):
                vol_err = noises[p] * v_n * v_h
                rows_out.append(dict(
                    ref_index=int(peak_idx),
                    plane=p,
                    vol=float(fit["vols"][p, ki]),
                    vol_err=vol_err,
                    N_ppm=float(uc[0].ppm(fit["r0"][ki])),
                    H_ppm=float(uc[1].ppm(fit["c0"][ki])),
                    lw_n=float(fit["lw_n"][ki]),
                    lw_h=float(fit["lw_h"][ki]),
                    converged=bool(fit["converged"]),
                ))

    result = pd.DataFrame(rows_out).sort_values(["ref_index", "plane"]).reset_index(drop=True)

    cache_path = cache_dir / f"{yaml_path.stem}_fit_lineshapes.json.zip"
    _cache_save(result, cache_path)
    _yaml_set_cache(yaml_path, cache_path)

    return result
