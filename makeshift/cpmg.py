import re
import zipfile
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.optimize import least_squares
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from .spectrum import (read_ucsf, estimate_background, pick_peaks,
                       plot_spectrum, plot_peaklist,
                       load_peaklist, map_peaklists, _AA_3TO1)
from .hydronmr.hydronmr.engine import run as run_hydronmr


# ---------------------------------------------------------------------------
# Cache helpers (private)
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


def _fetch_pdb(pdb_code: str) -> Path:
    """Download a PDB file from RCSB if not already cached, return its path."""
    import urllib.request
    cache_dir = Path.home() / ".makeshift" / "pdb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{pdb_code.upper()}.pdb"
    if not dest.exists():
        url = f"https://files.rcsb.org/download/{pdb_code.upper()}.pdb"
        print(f"  Downloading {pdb_code.upper()} from RCSB ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved → {dest}")
    return dest


def _yaml_set_cache(yaml_path: Path, cache_path: Path, key: str = "fit_lineshapes_cache") -> None:
    """Insert or replace a `<key>: <cache_path>` line in a YAML file."""
    text = yaml_path.read_text()
    new_line = f"{key}: {cache_path}\n"
    if f"{key}:" in text:
        lines = [
            new_line if line.startswith(f"{key}:") else line
            for line in text.splitlines(keepends=True)
        ]
        text = "".join(lines)
    else:
        text = text.rstrip("\n") + "\n" + new_line
    yaml_path.write_text(text)
    print(f"  Written {key} to {yaml_path.name}")

# ---------------------------------------------------------------------------
# Config / I/O
# ---------------------------------------------------------------------------

def load_config(yaml_path):
    """
    Load a CPMG experiment config from a YAML file.

    Expected format::

        time_T2: 0.05  # constant-time CPMG delay, seconds

        data_dir: /path/to/ucsf/files

        reference: ref.ucsf

        planes:
          - file: plane_0080.ucsf
            vcpmg: 80
          - file: plane_0120.ucsf
            vcpmg: 120
          ...

        sequence: MTEYKLVVVGA...   # optional, 1-indexed full construct sequence

    Parameters
    ----------
    yaml_path : str or Path

    Returns
    -------
    dict with keys: time_T2, reference, planes
        reference and planes[*].file are resolved to absolute Path objects.
        'sequence', if present, is returned as-is (str).
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    required = {"time_T2", "data_dir", "reference", "planes"}
    missing = required - set(cfg.keys())
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")
    if not isinstance(cfg["planes"], list) or len(cfg["planes"]) == 0:
        raise ValueError("'planes' must be a non-empty list")
    for i, plane in enumerate(cfg["planes"]):
        if "file" not in plane or "vcpmg" not in plane:
            raise ValueError(f"planes[{i}] is missing 'file' or 'vcpmg'")

    data_dir = Path(cfg["data_dir"]).expanduser()
    cfg["reference"] = data_dir / cfg["reference"]
    for plane in cfg["planes"]:
        plane["file"] = data_dir / plane["file"]
    if "pdb" in cfg:
        pdb_val = str(cfg["pdb"]).strip()
        if len(pdb_val) == 4 and pdb_val.isalnum():
            cfg["pdb"] = _fetch_pdb(pdb_val)
        else:
            cfg["pdb"] = Path(pdb_val).expanduser()

    cfg.setdefault("peak_mapping_tol", 1)
    cfg.setdefault("baseline_ref_plane", 10)
    cfg.setdefault("start_num", 1)
    cfg.setdefault("end_num", 3)
    cfg["yaml_path"] = Path(yaml_path)

    return cfg


def load_planes(config):
    """
    Read all UCSF planes described in a config dict (from load_config).

    Parameters
    ----------
    config : dict
        Output of load_config.

    Returns
    -------
    ref_spectrum : Spectrum — reference plane (data + unit-conversion)
    plane_data : list of ndarray — [ref_data, cpmg_plane_1, cpmg_plane_2, ...]
    vcpmg_values : list of float — νCPMG in Hz, one per CPMG plane
    time_T2 : float — constant-time delay in seconds
    """
    ref_spectrum = read_ucsf(config["reference"])
    sorted_planes = sorted(config["planes"], key=lambda p: float(p["vcpmg"]))
    plane_data = [ref_spectrum.data]
    vcpmg_values = []
    for plane in sorted_planes:
        plane_data.append(read_ucsf(plane["file"]).data)
        print(f'read data from {plane["file"]}')
        vcpmg_values.append(float(plane["vcpmg"]))
    return ref_spectrum, plane_data, vcpmg_values, float(config["time_T2"])


# ---------------------------------------------------------------------------
# HYDRONMR
# ---------------------------------------------------------------------------

def calc_rigid_R2(config, overwrite_cache=False):
    """
    Run HYDRONMR on the PDB structure in config["pdb"] and return per-residue
    predicted R2 values scaled to the experiment field/temperature.

    n_residues is inferred from config["sequence"] when present (trims
    multi-chain HYDRONMR output to the construct length). cache_dir defaults
    to next to the YAML file.

    Caching: checks for an existing cache in this order:
      1. hydronmr_r2_cache key in the YAML file
      2. <yaml_stem>_hydronmr_r2.csv next to the YAML

    After a successful run the cache is saved and hydronmr_r2_cache is
    written/updated in the YAML file.

    Parameters
    ----------
    config : dict
        Output of load_config. Must contain a "pdb" key.
    overwrite_cache : bool
        If True, re-run HYDRONMR even if a cached CSV exists.

    Returns
    -------
    DataFrame with columns: seqpos (int), R2_hydro (float)
    """
    if "pdb" not in config:
        raise ValueError("config is missing 'pdb' (path to PDB structure for HYDRONMR)")

    yaml_path = config["yaml_path"]
    cache_dir = yaml_path.parent
    sequence = config.get("sequence")
    n_residues = len(sequence) if sequence else None

    # ── cache check ─────────────────────────────────────────────────────────
    if not overwrite_cache:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        # 1. YAML-configured cache
        if "hydronmr_r2_cache" in cfg:
            configured = Path(cfg["hydronmr_r2_cache"])
            if configured.exists():
                df = pd.read_csv(configured, usecols=lambda c: c in ("seqpos", "R2_hydro"))
                print(f"  {len(df)} rows loaded from HYDRONMR cache → {configured.name}")
                if n_residues is not None:
                    df = df[df["seqpos"] <= n_residues].reset_index(drop=True)
                df.attrs["cache_path"] = configured
                return df

        # 2. Default cache name (handles cache present but YAML not yet updated)
        default_cache = cache_dir / f"{yaml_path.stem}_hydronmr_r2.csv"
        if default_cache.exists():
            df = pd.read_csv(default_cache, usecols=lambda c: c in ("seqpos", "R2_hydro"))
            print(f"  {len(df)} rows loaded from HYDRONMR cache → {default_cache.name}")
            _yaml_set_cache(yaml_path, default_cache, key="hydronmr_r2_cache")
            if n_residues is not None:
                df = df[df["seqpos"] <= n_residues].reset_index(drop=True)
            df.attrs["cache_path"] = default_cache
            return df

    # ── run HYDRONMR ────────────────────────────────────────────────────────
    pdb_path = Path(config["pdb"])
    result = run_hydronmr(pdb_path)
    rows = [dict(seqpos=resseq, R2_hydro=1.0 / t2)
            for (chain, resseq), (t1, t2, ratio, noe) in result.per_residue.items()]
    df = pd.DataFrame(rows)
    # Multi-chain structures (e.g. a crystallographic dimer) can have
    # multiple chains sharing the same residue numbering; average their
    # predictions so seqpos is unique.
    df = df.groupby("seqpos", as_index=False)["R2_hydro"].mean()
    df = df.sort_values("seqpos").reset_index(drop=True)

    cache_path = cache_dir / f"{yaml_path.stem}_hydronmr_r2.csv"
    df.to_csv(cache_path, index=False)
    print(f"  Saved HYDRONMR R2 cache → {cache_path.name}")
    _yaml_set_cache(yaml_path, cache_path, key="hydronmr_r2_cache")

    if n_residues is not None:
        df = df[df["seqpos"] <= n_residues].reset_index(drop=True)
    df.attrs["cache_path"] = cache_path
    return df


# ---------------------------------------------------------------------------
# Lineshape fitting (PINT-style)
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

        # 1. YAML-configured cache
        if "fit_lineshapes_cache" in cfg:
            configured = Path(cfg["fit_lineshapes_cache"])
            if configured.exists():
                print(f"  Loading lineshape cache: {configured.name}")
                return _cache_load(configured)

        # 2. Default cache name (handles cache present but YAML not yet updated)
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

    # ── cache save ───────────────────────────────────────────────────────────
    cache_path = cache_dir / f"{yaml_path.stem}_fit_lineshapes.json.zip"
    _cache_save(result, cache_path)
    _yaml_set_cache(yaml_path, cache_path)

    return result


# ---------------------------------------------------------------------------
# R2eff calculation
# ---------------------------------------------------------------------------

def compute_r2eff(fit_results, vcpmg_values, time_T2, duplicate_errors=True):
    """
    Compute effective R₂ from fitted peak volumes.

    R2eff(νCPMG) = -(1/T) × ln(I(νCPMG) / I₀)

    Noise errors are propagated from per-plane fit uncertainties. When
    duplicate_errors=True (default), any duplicated νCPMG points are also used
    to estimate a max-absolute-deviation error floor; whichever is larger
    (noise or duplicate-based) is kept in R2eff_err, and the source is recorded
    in the `larger_err` column ('noise' or 'duplicates').

    Parameters
    ----------
    fit_results : DataFrame
        Output of fit_peaks.
    vcpmg_values : list of float
        νCPMG frequencies in Hz, one per CPMG plane (not including the reference).
    time_T2 : float
        Constant-time CPMG delay in seconds.
    duplicate_errors : bool
        If True (default), apply duplicate-based error estimation on top of
        noise propagation.

    Returns
    -------
    DataFrame with columns: ref_index, vcpmg, t_cpmg, R2eff, R2eff_err,
    vol, vol_err, vol_ref, vol_ref_err, N_ppm, H_ppm,
    and (when duplicate_errors=True) R2eff_err_dup, larger_err.
    """
    ref_vols = fit_results[fit_results.plane == 0].set_index("ref_index")["vol"].to_dict()
    ref_errs = fit_results[fit_results.plane == 0].set_index("ref_index")["vol_err"].to_dict()

    rows = []
    for plane_idx, vc in enumerate(vcpmg_values, start=1):
        sub = fit_results[fit_results.plane == plane_idx].copy()
        sub["vol_ref"] = sub["ref_index"].map(ref_vols)
        sub["vol_ref_err"] = sub["ref_index"].map(ref_errs)
        sub["vcpmg"] = vc
        sub["t_cpmg"] = time_T2
        valid = (sub.vol > 0) & (sub.vol_ref > 0)
        sub["R2eff"] = np.nan
        sub["R2eff_err"] = np.nan
        I, I0 = sub.loc[valid, "vol"], sub.loc[valid, "vol_ref"]
        sI, sI0 = sub.loc[valid, "vol_err"], sub.loc[valid, "vol_ref_err"]
        sub.loc[valid, "R2eff"] = -1 / time_T2 * np.log(I / I0)
        sub.loc[valid, "R2eff_err"] = (1 / time_T2) * np.sqrt((sI / I)**2 + (sI0 / I0)**2)
        rows.append(sub[["ref_index", "vcpmg", "t_cpmg", "R2eff", "R2eff_err",
                          "vol", "vol_err", "vol_ref", "vol_ref_err", "N_ppm", "H_ppm"]])

    out = pd.concat(rows, ignore_index=True)
    if duplicate_errors:
        out = _apply_duplicate_errors(out)
    return out


# ---------------------------------------------------------------------------
# Duplicate-aware error estimation (internal)
# ---------------------------------------------------------------------------

def _apply_duplicate_errors(all_r2eff):
    """
    Replace noise-propagated errors with duplicate-based errors when larger.

    For each peak (ref_index):
      1. Find νCPMG values that appear more than once.
      2. Compute std(R2eff) for each duplicate group.
      3. Max std across all groups → R2eff_err_dup, applied
         uniformly to every νCPMG point for this peak.
      4. Use whichever is larger (dup or noise); record source in `larger_err`.

    Adds columns R2eff_err_dup and larger_err; updates R2eff_err in place.
    """
    df = all_r2eff.copy()
    df["R2eff_err_dup"] = np.nan
    df["larger_err"] = "noise"

    found_any = False

    for ref_idx, grp in df.groupby("ref_index"):
        idx = grp.index
        dup_vcpmg = grp["vcpmg"].value_counts()
        dup_vcpmg = dup_vcpmg[dup_vcpmg > 1].index

        if len(dup_vcpmg) == 0:
            continue

        found_any = True
        dup_rows = grp[grp["vcpmg"].isin(dup_vcpmg)].dropna(subset=["R2eff"])
        if len(dup_rows) == 0:
            continue

        max_dev = dup_rows.groupby("vcpmg")["R2eff"].std().max()

        df.loc[idx, "R2eff_err_dup"] = max_dev

        mean_noise = grp["R2eff_err"].dropna().mean()
        if max_dev > mean_noise:
            df.loc[idx, "larger_err"] = "duplicates"
            df.loc[idx, "R2eff_err"] = max_dev

    if not found_any:
        print("  No duplicate νCPMG points — using noise-propagated errors.")

    return df


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def validate_sequence(sequence, bmrb_peaks):
    """
    Check that a construct sequence agrees with BMRB-derived assignments.

    For each row in bmrb_peaks, sequence[Seq_ID - 1] must equal the
    one-letter code for Comp_ID (e.g. 'P' for PRO). Seq_ID (BMRB's
    1-indexed sequence position) is used rather than Auth_seq_ID (the
    author/PDB residue number), since the latter often has an arbitrary
    offset relative to a 1-indexed construct sequence.

    Parameters
    ----------
    sequence : str
        1-indexed sequence string (sequence[0] is residue 1).
    bmrb_peaks : DataFrame
        Output of fetch_bmrb_peaks; must contain Seq_ID and Comp_ID.

    Returns
    -------
    list of dict
        One entry per mismatch, with keys: seqpos, expected (from sequence),
        observed (from BMRB), assn_label. Empty if everything matches.

    Raises
    ------
    ValueError
        If any Seq_ID falls outside the sequence (1..len(sequence)).
    """
    mismatches = []
    for _, row in bmrb_peaks.iterrows():
        seqpos = int(row["Seq_ID"])
        if seqpos < 1 or seqpos > len(sequence):
            raise ValueError(
                f"BMRB residue {row['assn_label']} (Seq_ID={seqpos}) "
                f"is outside sequence (length {len(sequence)})"
            )
        expected = sequence[seqpos - 1]
        observed = _AA_3TO1.get(row["Comp_ID"], "?")
        if expected != observed:
            mismatches.append(dict(seqpos=seqpos, expected=expected,
                                   observed=observed, assn_label=row["assn_label"]))

    if mismatches:
        print(f"  WARNING: {len(mismatches)} sequence/BMRB mismatch(es):")
        for m in mismatches:
            print(f"    position {m['seqpos']}: sequence has '{m['expected']}', "
                  f"BMRB ({m['assn_label']}) has '{m['observed']}'")
    else:
        print(f"  Sequence matches BMRB assignments ({len(bmrb_peaks)} residues checked).")

    return mismatches


def _parse_seqpos(assn_label):
    if assn_label is None or (isinstance(assn_label, float) and np.isnan(assn_label)):
        return np.nan
    m = re.search(r"(\d+)", str(assn_label))
    return int(m.group(1)) if m else np.nan


def flatten_r2eff(all_r2eff, ref_df, max_r2err_threshold=100.0, start_num=2, end_num=3):
    """
    Reduce all_r2eff to one row per residue with R2first, R2last, and spread stats.
    Only well-fit peaks (>=4 points, all errors below threshold) are kept.
    Returns DataFrame with ref_index, seqpos, assn_label, R2first, R2last,
    std_first, std_last.
    """
    rows = []
    for ref_idx, grp in all_r2eff.groupby("ref_index"):
        grp = grp.dropna(subset=["R2eff", "R2eff_err"]).sort_values("vcpmg")
        if len(grp) < 4 or grp["R2eff_err"].max() >= max_r2err_threshold:
            continue
        firstfew = grp.iloc[:start_num]["R2eff"]
        lastfew  = grp.iloc[-end_num:]["R2eff"]

        rows.append(dict(
            ref_index=ref_idx,
            R2first=firstfew.mean(),
            R2last=lastfew.mean(),
            std_first=grp.iloc[:start_num]["R2eff_err"].mean(),
            std_last=lastfew.max() - lastfew.min(),
            std_total=grp["R2eff"].max() - grp["R2eff"].min(),
        ))
    df = pd.DataFrame(rows)
    df = df.merge(ref_df[["ref_index", "assn_label"]], on="ref_index", how="left")
    df["seqpos"] = df["assn_label"].apply(_parse_seqpos)
    return df


def fit_R2_rigid(df, hydro_df):
    """
    Merge HYDRONMR R2 predictions onto df (by seqpos), fit a scale factor
    C so that C*R2_hydro ≈ R2last for the most rigid residues, and return
    df with scaled_R2_pred added.
    """
    df = df.merge(hydro_df[["seqpos", "R2_hydro"]], on="seqpos", how="left")
    ma = (df["std_last"] < np.nanmedian(df["std_last"])) & df["R2_hydro"].notna()
    scalar = float((df.loc[ma, "R2last"] * df.loc[ma, "R2_hydro"]).sum()
                   / (df.loc[ma, "R2_hydro"] ** 2).sum())
    df["scaled_R2_pred"] = scalar * df["R2_hydro"]
    residuals = df.loc[ma, "R2last"] - df.loc[ma, "scaled_R2_pred"]
    df["rigid_rmse"] = float(np.sqrt(np.mean(residuals**2)))
    print(f"  HYDRONMR scale factor: {scalar:.3f}  ({ma.sum()} rigid residues used, RMSE={df['rigid_rmse'].iloc[0]:.3f})")
    return df


def classify_peaks(all_r2eff, ref_df, config, max_r2err_threshold=100.0):
    """Classify each peak as 'Rex', 'elevated_R2', or 'flat'."""

    start_num = config["start_num"]
    end_num = config["end_num"]
    df = flatten_r2eff(all_r2eff, ref_df, max_r2err_threshold, start_num, end_num)
    hydro_df = calc_rigid_R2(config)
    df = fit_R2_rigid(df, hydro_df)

    base_uncertainty = float(df["std_total"].median())
    print(f"  base_uncertainty: {base_uncertainty:.3f} s⁻¹ (median std_total)")

    df["Rex_val"] = df["R2first"] - df["R2last"]
    df["Rex_err"] = df["std_first"] + df["std_last"]
    df["Rex"] = ((df["Rex_val"] - df["Rex_err"]) > base_uncertainty)
    df["label"] = np.where(df["Rex"], "Rex", "flat")

    rigid_rmse = float(df["rigid_rmse"].iloc[0])
    df["elevated_R2"] = (df["scaled_R2_pred"].notna()
        & (df["R2last"] - df["std_last"] > df["scaled_R2_pred"] + rigid_rmse + base_uncertainty))
    df["label"] = np.where(df["elevated_R2"], "elevated_R2", df["label"])

    return df.sort_values("seqpos").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-peak dispersion plot
# ---------------------------------------------------------------------------

def plot_r2eff_per_peak(all_r2eff, classifications, pdf_path, ref_df=None):
    """
    Write a multi-page PDF with one page per peak showing R2eff vs νCPMG.

    Each page mirrors the plot_peak style from process_CPMG_err.py:
      - Errorbars from whichever source won (duplicates or noise), read from
        the `larger_err` column added by apply_duplicate_errors (falls back to
        R2eff_err if the column is absent).
      - Dotted horizontal line at R2last with a grey shaded ±std_last band.
      - Title: BMRB label (if ref_df has assn_label), peak index, (N, H) ppm,
        Rex ± Rex_err, label, error method.

    Parameters
    ----------
    all_r2eff : DataFrame
        Output of compute_r2eff.
    classifications : DataFrame
        Output of classify_peaks.
    pdf_path : str or Path
    ref_df : DataFrame, optional
        If provided and contains a 'assn_label' column, the BMRB label is
        prepended to each page title.
    """
    class_map = classifications.set_index("ref_index")
    has_dup_col = "larger_err" in all_r2eff.columns and "R2eff_err_dup" in all_r2eff.columns

    bmrb_map = {}
    if ref_df is not None and "assn_label" in ref_df.columns:
        bmrb_map = ref_df.set_index("ref_index")["assn_label"].dropna().to_dict()

    with PdfPages(pdf_path) as pdf:
        for ref_idx, grp in all_r2eff.groupby("ref_index"):
            grp = grp.dropna(subset=["R2eff"]).sort_values("vcpmg")
            if len(grp) == 0 or ref_idx not in class_map.index:
                continue

            cls = class_map.loc[ref_idx]

            # choose error bars
            larger_err = grp["larger_err"].iloc[0] if has_dup_col else "noise"
            if larger_err == "duplicates":
                yerr = grp["R2eff_err_dup"].fillna(grp["R2eff_err"]).values
            else:
                yerr = grp["R2eff_err"].values

            N_ppm = grp["N_ppm"].iloc[0]
            H_ppm = grp["H_ppm"].iloc[0]
            bmrb_lbl = bmrb_map.get(ref_idx)
            prefix = f"{bmrb_lbl}  " if bmrb_lbl else ""

            fig, ax = plt.subplots(figsize=(4, 3))
            ax.errorbar(grp["vcpmg"].values, grp["R2eff"].values, yerr=yerr,
                        capsize=2, fmt=".", color="tab:blue")

            if cls is not None and not np.isnan(cls["R2last"]):
                ax.axhline(cls["R2last"], linestyle=":", zorder=0, color="grey")
                x0, x1 = ax.get_xlim()
                ax.fill_between(
                    [x0, x1],
                    cls["R2last"] - cls["std_last"],
                    cls["R2last"] + cls["std_last"],
                    alpha=0.1, zorder=0, color="grey", linewidth=0,
                )
                ax.set_xlim([x0, x1])

            ax.set_xlabel(r"$\nu_{CPMG}$ (Hz)")
            ax.set_ylabel(r"$R_{2,\mathrm{eff}}$ (s$^{-1}$)")

            if cls is not None:
                title = (
                    f"{prefix}peak {ref_idx}  ({N_ppm:.1f}, {H_ppm:.2f} ppm)\n"
                    f"Rex = {cls['Rex_val']:.2f} ± {cls['Rex_err']:.2f} s⁻¹"
                    f"  |  {cls['label']}  |  err: {larger_err}"
                )
            else:
                title = f"{prefix}peak {ref_idx}  ({N_ppm:.1f}, {H_ppm:.2f} ppm)"
            ax.set_title(title, fontsize=7)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"  Saved per-peak PDF → {Path(pdf_path).name}")


def plot_r2eff_grid(all_r2eff, classifications, ref_df=None,
                    color_map=None, ncols=10,
                    figsize_per_panel=(2.5, 2.0),
                    title_fontsize=6, axis_fontsize=7):
    """
    Plot all R2eff dispersion curves in a compact grid, one panel per peak.

    Peaks are sorted N→C by residue number when BMRB labels are available
    (parsed from ref_df['assn_label']).  Unassigned peaks appear at the end.

    Parameters
    ----------
    all_r2eff : DataFrame
        Output of compute_r2eff.
    classifications : DataFrame
        Output of classify_peaks.
    ref_df : DataFrame, optional
        If provided and contains a 'assn_label' column, labels are used as
        panel titles and peaks are sorted N→C.
    color_map : dict, optional
        Maps classification labels to matplotlib colours. Missing keys → 'black'.
    ncols : int
        Number of columns in the grid (default 10).
    figsize_per_panel : tuple
        (width, height) in inches per panel.
    title_fontsize, axis_fontsize : int
        Font sizes for panel titles and tick labels.

    Returns
    -------
    fig : matplotlib Figure
    """
    if color_map is None:
        color_map = {
            "Rex": "tab:orange",
            "elevated_R2": "#8ACE00",
            "flat": "black",
            "unfit": "red",
        }

    class_map = classifications.set_index("ref_index")

    # ── build label map and N→C sort order ──────────────────────────────────
    has_bmrb = (ref_df is not None
                and "assn_label" in ref_df.columns
                and ref_df["assn_label"].notna().any())

    ref_indices = list(all_r2eff["ref_index"].unique())

    if has_bmrb:
        bmrb_map = ref_df.set_index("ref_index")["assn_label"].to_dict()

        def _res_num(idx):
            lbl = bmrb_map.get(idx)
            if lbl is None or (isinstance(lbl, float) and np.isnan(lbl)):
                return 99999
            m = re.search(r"(\d+)", str(lbl))
            return int(m.group(1)) if m else 99999

        ref_indices = sorted(ref_indices, key=_res_num)
    else:
        bmrb_map = {}
        ref_indices = sorted(ref_indices)

    # ── figure layout ────────────────────────────────────────────────────────
    n_peaks = len(ref_indices)
    nrows   = int(np.ceil(n_peaks / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * figsize_per_panel[0], nrows * figsize_per_panel[1]),
        squeeze=False,
    )

    for i, ref_idx in enumerate(ref_indices):
        row, col = divmod(i, ncols)
        ax = axes[row][col]

        grp = (all_r2eff[all_r2eff["ref_index"] == ref_idx]
               .dropna(subset=["R2eff"])
               .sort_values("vcpmg"))

        cls   = class_map.loc[ref_idx] if ref_idx in class_map.index else None
        label = cls["label"] if cls is not None else "unfit"
        color = color_map.get(label, "black")

        if len(grp) > 0:
            ax.errorbar(grp["vcpmg"], grp["R2eff"], yerr=grp["R2eff_err"],
                        fmt=".", color=color, capsize=1.5, ms=3, lw=0.8, elinewidth=0.8)

            if cls is not None and not np.isnan(cls["R2last"]):
                ax.axhline(cls["R2last"], linestyle=":", lw=0.8, color="grey", zorder=0)
                x0, x1 = ax.get_xlim()
                ax.fill_between([x0, x1],
                                cls["R2last"] - cls["std_last"],
                                cls["R2last"] + cls["std_last"],
                                alpha=0.15, color="grey", linewidth=0, zorder=0)
                ax.set_xlim([x0, x1])

            if (cls is not None and "scaled_R2_pred" in cls.index
                    and not pd.isna(cls["scaled_R2_pred"])):
                x0, x1 = ax.get_xlim()
                ax.axhline(cls["scaled_R2_pred"], linestyle="-", lw=0.8,
                          color="k", zorder=1)
                ax.set_xlim([x0, x1])

        # panel title: BMRB label, Rex value for Rex peaks
        bmrb_lbl = bmrb_map.get(ref_idx)
        if bmrb_lbl and not (isinstance(bmrb_lbl, float) and np.isnan(bmrb_lbl)):
            title = str(bmrb_lbl)
        else:
            title = str(ref_idx)

        if cls is not None and cls["label"] == "Rex" and not np.isnan(cls["Rex_val"]):
            title += f"\n{cls['Rex_val']:.1f}±{cls['Rex_err']:.1f} s⁻¹"

        ax.set_title(title, fontsize=title_fontsize, color="black")
        ax.tick_params(labelsize=title_fontsize - 1)

        ymin, ymax = ax.get_ylim()
        if ymax - ymin < 2:
            mid = (ymin + ymax) / 2
            ax.set_ylim(mid - 1, mid + 1)

    # hide unused panels
    for j in range(n_peaks, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row][col].set_visible(False)

    # shared axis labels on outer edges only
    for r in range(nrows):
        axes[r][0].set_ylabel(r"$R_{2,\mathrm{eff}}$ (s$^{-1}$)", fontsize=axis_fontsize)
    for c in range(ncols):
        axes[-1][c].set_xlabel(r"$\nu_\mathrm{CPMG}$ (Hz)", fontsize=axis_fontsize)

    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Waterfall plot
# ---------------------------------------------------------------------------

def plot_waterfall(all_r2eff, classifications, ref_df, color_map=None, sequence=None, missing_df=None):
    """
    Plot R2eff vs sequence position for every νCPMG point (rainbow-coloured
    by νCPMG), with vertical bands marking peak classification, prolines,
    and missing residues.

    Parameters
    ----------
    all_r2eff : DataFrame
        Output of compute_r2eff.
    classifications : DataFrame
        Output of classify_peaks; scaled_R2_pred is overlaid as a black line.
    ref_df : DataFrame
        Must contain ref_index and assn_label columns (used to map peaks to
        sequence positions).
    color_map : dict, optional
        Maps classification labels to matplotlib colours for the axvline
        bands. Missing keys are not shaded.
    sequence : str, optional
        One-letter sequence string, 1-indexed by position, with 'P' marking
        prolines and '.' marking residues missing from the spectrum (e.g.
        unassigned/overlapped). When given, the x-axis spans the full
        sequence and prolines/missing residues are annotated. When omitted,
        the x-axis spans only the residues present in ref_df.
    missing_df : DataFrame, optional
        Columns: seqpos, label, where label is "proline" or
        "no_peaklist_assignment" (see run_protein). Drawn as vertical bands —
        prolines always purple, no_peaklist_assignment coloured via
        color_map (default tab:red). Takes precedence over the 'P'/'.'
        characters in `sequence`.

    Returns
    -------
    fig, ax
    """
    if color_map is None:
        color_map = {
            "Rex": "tab:orange",
            "elevated_R2": "#8ACE00",
            "no_peaklist_assignment": "tab:red",
        }

    df = classifications.dropna(subset=["seqpos"]).copy()
    df["seqpos"] = df["seqpos"].astype(int)
    seqpos_map = df.set_index("seqpos")
    ref_index_to_seqpos = df.set_index("ref_index")["seqpos"]

    n_res = len(sequence) if sequence is not None else int(df["seqpos"].max())

    fig, ax = plt.subplots(figsize=(0.06 * n_res + 1, 1.5))

    points = all_r2eff.dropna(subset=["R2eff"]).copy()
    points["seqpos"] = points["ref_index"].map(ref_index_to_seqpos)
    points = points.dropna(subset=["seqpos"])
    points["seqpos"] = points["seqpos"].astype(int)

    vcpmg_values = sorted(points["vcpmg"].unique())
    palette = sns.color_palette("rainbow_r", len(vcpmg_values))
    for vc, color in zip(vcpmg_values, palette):
        sub = points[points["vcpmg"] == vc]
        ax.scatter(sub["seqpos"], sub["R2eff"], s=5, color=color, zorder=2)

    pred_line = None
    if "scaled_R2_pred" in df.columns and df["scaled_R2_pred"].notna().any():
        pred_line = df[["seqpos", "scaled_R2_pred", "rigid_rmse"]].dropna(subset=["scaled_R2_pred"]).sort_values("seqpos")

    if pred_line is not None and len(pred_line):
        y = pred_line["scaled_R2_pred"]
        x = pred_line["seqpos"]
        rmse = pred_line["rigid_rmse"].iloc[0]
        ax.plot(x, y, color="k", zorder=3)
        ax.fill_between(x, y - rmse, y + rmse, color="k", alpha=0.15, zorder=2, linewidth=0)

    ax.set_xlim(0, n_res + 1)
    ax.set_xlabel("Residue")
    ax.set_ylabel(r"$R_2$ (s$^{-1}$)")

    ymin, ymax = ax.get_ylim()
    p_pos = ymin + 0.05 * (ymax - ymin)
    star_pos = ymin + 0.9 * (ymax - ymin)

    missing_map = {}
    if missing_df is not None:
        missing_map = missing_df.set_index("seqpos")["label"].to_dict()

    if sequence is not None:
        for i, char in enumerate(sequence):
            seqpos = i + 1
            m_label = missing_map.get(seqpos)
            if m_label == "proline" or char == "P":
                ax.axvline(seqpos, color="tab:purple", zorder=0, alpha=0.5, linewidth=0.5)
                ax.text(seqpos - 0.5, p_pos, "P", color="tab:purple", fontsize=5, weight="bold")
            elif m_label == "no_peaklist_assignment":
                # missing from original peaklist — red star
                color = color_map.get("no_peaklist_assignment", "tab:red")
                ax.axvline(seqpos, color=color, zorder=0, alpha=0.5, linewidth=0.5)
                ax.scatter([seqpos], [star_pos], marker="*", color=color)
            elif seqpos not in seqpos_map.index:
                # in sequence but no fit data (overlap, poor fit, etc.) — grey line
                ax.axvline(seqpos, color="grey", zorder=0, alpha=0.3, linewidth=0.5)
            else:
                label = seqpos_map.loc[seqpos, "label"]
                if label in ("Rex", "elevated_R2"):
                    color = color_map.get(label)
                    if color is not None:
                        ax.axvline(seqpos, color=color, zorder=0, alpha=0.3)
    else:
        for seqpos, row in seqpos_map.iterrows():
            if row["label"] in ("Rex", "elevated_R2"):
                color = color_map.get(row["label"])
                if color is not None:
                    ax.axvline(seqpos, color=color, zorder=0, alpha=0.3)

    ax.set_ylim(ymin, ymax)
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------

def save_results_csv(ref_df, classifications, csv_path, fit_results=None, missing_df=None):
    """
    Write a per-peak summary CSV combining peak assignments and classification.

    Columns: ref_index, assn_label (if present), N_ppm, H_ppm, vol, vol_err,
    lw_n, lw_h (if fit_results given, from the reference plane),
    R2first, R2last, std_first, std_last, std_total, Rex_val, Rex_err, Rex,
    elevated_R2, label, seqpos, R2_hydro, scaled_R2_pred, rigid_rmse, missing_assignment.

    Rows are sorted by seqpos (peaks without a seqpos are placed last).

    Parameters
    ----------
    ref_df : DataFrame
        Picked peaks, optionally with assn_label.
    classifications : DataFrame
        Output of classify_peaks.
    csv_path : str or Path
    fit_results : DataFrame, optional
        Output of fit_peaks. If given, the reference-plane (plane 0) vol,
        vol_err, lw_n, lw_h are merged in.
    missing_df : DataFrame, optional
        Columns: seqpos, label ("proline" or "no_peaklist_assignment"), as
        computed in run_protein. Appended as extra rows (with NaN peak
        columns) and carried forward in the new "missing_assignment" column.

    Returns
    -------
    DataFrame written to csv_path.
    """
    classifications = classifications.drop(columns=["assn_label"], errors="ignore")
    ref_cols = [c for c in ["ref_index", "assn_label", "N_ppm", "H_ppm"] if c in ref_df.columns]
    out = ref_df[ref_cols].merge(classifications, on="ref_index", how="left")

    if fit_results is not None:
        ref_fit = fit_results[fit_results["plane"] == 0][
            ["ref_index", "vol", "vol_err", "lw_n", "lw_h"]
        ]
        out = out.merge(ref_fit, on="ref_index", how="left")

    out["missing_assignment"] = np.nan

    if missing_df is not None and len(missing_df):
        existing_seqpos = set(out["seqpos"].dropna().astype(int)) if "seqpos" in out.columns else set()
        extra = missing_df[~missing_df["seqpos"].isin(existing_seqpos)].copy()
        extra = extra.rename(columns={"label": "missing_assignment"})
        out = pd.concat([out, extra], ignore_index=True)

    if "seqpos" in out.columns:
        out = out.sort_values("seqpos", na_position="last").reset_index(drop=True)

    out.to_csv(csv_path, index=False)
    print(f"  Saved results CSV → {Path(csv_path).name}")
    return out


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def run_protein(yaml_path, out_dir, lineshape="gaussian",
                max_r2err_threshold=10000.0, color_map=None,
                xlim=(10.5, 6.0), ylim=(135, 100),
                zoom_xlim=(9, 7), zoom_ylim=(125, 112)):
    """
    Run the full CPMG pipeline for one protein and write standard outputs.

    Steps: load config/planes -> pick peaks -> (optional) BMRB assignment ->
    fit lineshapes -> compute R2eff -> classify peaks (with HYDRONMR
    elevated_R2 classification) -> write plots and CSV.

    Parameters
    ----------
    yaml_path : str or Path
        CPMG config YAML (see load_config).
    out_dir : str or Path
        Output directory; created if absent. Files are named
        <yaml_stem>_<thing>.<ext>.
    lineshape, max_r2err_threshold :
        Passed to fit_peaks / classify_peaks. Peak-picking baseline,
        matching tolerance, start_num, and end_num are read from config
        (see load_config for defaults).
    color_map : dict, optional
        Classification -> colour, used for the classified HSQC and waterfall
        plots. Defaults to a built-in CPMG_COLORS-style map.
    xlim, ylim : tuple
        ppm window for HSQC plots.
    zoom_xlim, zoom_ylim : tuple
        ppm window for the zoomed-in panel of the peak-mapping plot.

    Returns
    -------
    dict with keys: ref_df, fit_results, all_r2eff, classifications,
    hydro_df.
    """
    if color_map is None:
        color_map = {
            "Rex": "tab:orange",
            "elevated_R2": "#8ACE00",
            "flat": "black",
            "unfit": "red",
        }

    yaml_path = Path(yaml_path)
    out_dir = Path(out_dir) / "outputs"
    cache_dir = Path(out_dir).parent / "caches"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = yaml_path.stem

    print("== Loading config and planes ==")
    config = load_config(yaml_path)
    ref_spectrum, plane_data, vcpmg_values, time_T2 = load_planes(config)

    print("== Picking peaks ==")
    ref_df = pick_peaks(ref_spectrum, baseline=config["baseline_ref_plane"])
    ref_df["ref_index"] = ref_df.index
    print(f"  {len(ref_df)} peaks picked")

    fig, ax = plot_spectrum(ref_spectrum, baseline=config["baseline_ref_plane"], xlim=xlim, ylim=ylim,cmap='Greys_r')
    ax.set_title(f"{stem} — reference HSQC")
    fig.savefig(out_dir / f"{stem}_hsqc_unlabeled.pdf", bbox_inches="tight")
    plt.close(fig)

    print("== Peak assignment ==")
    sequence = config.get("sequence")
    peaklist = config.get("peaklist")
    missing_df = None
    if peaklist and sequence:
        peaks_for_missing = load_peaklist(peaklist)
        assigned_seqpos = set(
            peaks_for_missing["assn_label"].apply(_parse_seqpos).dropna().astype(int)
        )
        rows = []
        for i, char in enumerate(sequence):
            seqpos = i + 1
            if char == "P":
                rows.append(dict(seqpos=seqpos, label="proline"))
            elif seqpos not in assigned_seqpos:
                rows.append(dict(seqpos=seqpos, label="no_peaklist_assignment"))
        missing_df = pd.DataFrame(rows)
    if peaklist:
        picked_df = ref_df.copy()
        peaks = load_peaklist(peaklist)
        if sequence:
            validate_sequence(sequence, peaks)

        ref_df, peaks_shifted = map_peaklists(ref_df, peaks, tol=config["peak_mapping_tol"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, panel_xlim, panel_ylim in zip(axes, (xlim, zoom_xlim), (ylim, zoom_ylim)):
            plot_spectrum(ref_spectrum, baseline=config["baseline_ref_plane"],
                          xlim=panel_xlim, ylim=panel_ylim, cmap="Greys_r", ax=ax)
            plot_peaklist(ax, picked_df, text=None, color="tab:blue", label="picked peaks")
            plot_peaklist(ax, peaks_shifted, text="assn_label", color="tab:green", label="ref peaklist, translated")
            plot_peaklist(ax, ref_df, text="assn_label", color="magenta", label="mapped peaks")
        fig.suptitle(f"{stem} — peak mapping")
        plt.tight_layout()
        fig.savefig(out_dir / f"{stem}_peak_mapping.pdf", bbox_inches="tight")
        plt.close(fig)
    else:
        print("  No peaklist in config — skipping.")

    print("== Fitting lineshapes ==")
    fit_results = fit_peaks(plane_data, ref_df, ref_spectrum.uc, config, lineshape=lineshape, cache_dir=cache_dir)

    print("== Computing R2eff ==")
    all_r2eff = compute_r2eff(fit_results, vcpmg_values, time_T2)

    print("== Classifying peaks ==")
    classifications = classify_peaks(all_r2eff, ref_df, config,
                                     max_r2err_threshold=max_r2err_threshold)
    print(classifications["label"].value_counts().to_string())

    merged = ref_df.merge(classifications[["ref_index", "label"]], on="ref_index", how="left")
    merged["label"] = merged["label"].fillna("unlabeled")

    text_col = ("assn_label"
                if "assn_label" in merged.columns and merged["assn_label"].notna().any()
                else "ref_index")

    fig, ax = plot_spectrum(ref_spectrum, baseline=config["baseline_ref_plane"],
                            contour_color="black", xlim=xlim, ylim=ylim,
                            figsize=(9, 8))
    plot_peaklist(ax, merged, marker=".", markersize=4,
                  text=text_col, label_fontsize=6,
                  hue="label", palette=color_map)
    ax.set_title(f"{stem} — CPMG classification", fontsize=14)
    fig.savefig(out_dir / f"{stem}_hsqc_annotated_assigned.pdf", bbox_inches="tight")
    plt.close(fig)

    print("== Plotting dispersion grid ==")
    fig, _ = plot_r2eff_grid(all_r2eff, classifications, ref_df=ref_df, color_map=color_map)
    fig.savefig(out_dir / f"{stem}_grid.pdf", bbox_inches="tight")
    plt.close(fig)


    print("== Plotting waterfall ==")
    fig, ax = plot_waterfall(all_r2eff, classifications, ref_df, color_map=color_map,
                             sequence=sequence, missing_df=missing_df)
    fig.savefig(out_dir / f"{stem}_waterfall.pdf", bbox_inches="tight")
    plt.close(fig)

    print("== Writing CSV ==")
    save_results_csv(ref_df, classifications, out_dir / f"{stem}.csv", fit_results=fit_results, missing_df=missing_df)

    return dict(ref_df=ref_df, fit_results=fit_results, all_r2eff=all_r2eff,
                classifications=classifications)
