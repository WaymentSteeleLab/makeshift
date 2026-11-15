"""
Stateless format helpers for the ChemEx subpackage: build CPMG profiles from
makeshift's lineshape fits, write ChemEx's input files (``.out`` profiles and
the experiment / parameters / method TOMLs), and parse its output files back
into DataFrames.

ChemEx file conventions used here (checked against ChemEx 2026.9):

* A CPMG profile is named after its spin system, e.g. ``Y4N-H``, and holds
  ``ncyc  intensity  error`` rows; ncyc 0 is the reference plane.
  ChemEx defines nu_CPMG = ncyc / time_t2.
* ``carrier`` in the experiment file is in **ppm** (ChemEx's docstring says
  Hz, but the engine treats it as ppm), ``pw90`` and ``time_t2`` in seconds,
  ``h_larmor_frq`` in MHz, ``temperature`` in °C.
* Parameter values may be a number or ``[value, min, max]``.
"""

import math
import re
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROFILE_HEADER = "#    ncyc_cp        Intensity      Esd(Int.)\n"


# --------------------------------------------------------------------------
# settings (the ``chemex:`` section of a CPMG config YAML)
# --------------------------------------------------------------------------

_NUMERIC_SETTINGS = {
    "h_larmor_frq": "1H Larmor frequency, MHz",
    "carrier": "15N carrier, ppm",
    "temperature": "temperature, °C",
    "pw90": "15N 90° pulse width, s",
}
_SETTINGS_DEFAULTS = {"experiment": "cpmg_15n_ip", "model": "2st"}
_DICT_SETTINGS = ("experiment_settings", "conditions", "parameters")
SETTINGS_KEYS = (*_SETTINGS_DEFAULTS, *_NUMERIC_SETTINGS, *_DICT_SETTINGS, "method")
REQUIRED_SETTINGS = ("temperature", "pw90")


def normalize_settings(settings, source="chemex settings"):
    """
    Validate a ``chemex:`` settings dict and fill in defaults.

    Keys (all flat; see the ChemEx guide for the YAML form):

    ``experiment`` (ChemEx experiment name, default ``cpmg_15n_ip``),
    ``h_larmor_frq`` (MHz), ``carrier`` (15N, ppm), ``temperature`` (°C),
    ``pw90`` (s), ``model`` (default ``2st``), and optional
    ``experiment_settings`` (extra ``[experiment]`` keys),
    ``conditions`` (extra ``[conditions]`` keys), ``parameters`` and
    ``method``. ``temperature`` and ``pw90`` are required;
    ``h_larmor_frq`` and ``carrier`` may be left out when they can be read
    from the reference spectrum's header.

    Raises ValueError naming ``source`` for unknown keys, missing required
    keys, or non-numeric values.
    """
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError(f"{source} must be a mapping, got {type(settings).__name__}")
    unknown = sorted(set(settings) - set(SETTINGS_KEYS))
    if unknown:
        raise ValueError(f"{source}: unknown key(s) {unknown}; allowed keys are "
                         f"{list(SETTINGS_KEYS)}")
    missing = [k for k in REQUIRED_SETTINGS if settings.get(k) is None]
    if missing:
        raise ValueError(f"{source}: missing required key(s) " + ", ".join(
            f"{k} ({_NUMERIC_SETTINGS[k]})" for k in missing)
            + "; add them to the chemex section or give `acqus:` in the config")

    out = {**_SETTINGS_DEFAULTS, **{k: v for k, v in settings.items() if v is not None}}
    for key, meaning in _NUMERIC_SETTINGS.items():
        if key in out:
            try:
                out[key] = float(out[key])
            except (TypeError, ValueError):
                raise ValueError(f"{source}: {key} ({meaning}) must be a number, "
                                 f"got {out[key]!r}") from None
    for key in _DICT_SETTINGS:
        if key in out and not isinstance(out[key], dict):
            raise ValueError(f"{source}: {key} must be a mapping")
    for section, values in (out.get("parameters") or {}).items():
        if str(section).upper() == "GLOBAL" and isinstance(values, dict):
            clash = [k for k in values if str(k).upper() in ("KEX_AB", "PB")]
            if clash:
                raise ValueError(f"{source}: set {clash} with ChemEx.run(kex=..., "
                                 "kex_bounds=..., pb=..., pb_bounds=...), not in "
                                 "parameters")
    if "method" in out and not isinstance(out["method"], (list, dict, str)):
        raise ValueError(f"{source}: method must be a list of parameters, "
                         "a mapping of steps, or a method-file path")
    return out


# --------------------------------------------------------------------------
# profiles
# --------------------------------------------------------------------------

def profile_name(label, atoms=("N", "H")):
    """'Y4' + ('N', 'H') -> 'Y4N-H'."""
    return f"{label}{atoms[0]}-" + "-".join(atoms[1:])


def spin_name(label, atoms=("N", "H")):
    """'Y4' + ('N', 'H') -> 'Y4N', the name ChemEx uses in per-spin sections."""
    return f"{label}{atoms[0]}"


def cpmg_profiles(fit_results, vcpmg_values, time_T2, ref_df=None,
                  label_col="assn_label", atoms=("N", "H"),
                  include_unassigned=False):
    """
    Turn :func:`~makeshift.relaxation.fit_peaks` output into ChemEx profiles.

    Parameters
    ----------
    fit_results : DataFrame
        Output of fit_peaks (``ref_index, plane, vol, vol_err``); plane 0 is
        the reference.
    vcpmg_values : list of float
        νCPMG in Hz for planes 1..n.
    time_T2 : float
        Constant-time CPMG delay in seconds.
    ref_df : DataFrame, optional
        Peak table with ``ref_index`` and ``label_col`` (e.g. ``Y4``). Peaks
        without a label are skipped unless ``include_unassigned=True`` (then
        named ``peak<ref_index>``, which ChemEx cannot parse as a residue).

    Returns
    -------
    dict mapping profile name (``Y4N-H``) -> DataFrame with columns
    ``ncyc, intensity, error, ref_index``.
    """
    ncyc = np.asarray(vcpmg_values, float) * float(time_T2)
    bad = ~np.isclose(ncyc, np.round(ncyc), atol=1e-3)
    if bad.any():
        pairs = ", ".join(f"{v:g} Hz -> {n:g}" for v, n in
                          zip(np.asarray(vcpmg_values, float)[bad], ncyc[bad]))
        raise ValueError(
            f"νCPMG × time_T2 must be a whole number of CPMG cycles, but with "
            f"time_T2 = {time_T2:g} s: {pairs}. time_T2 (the CPMG constant-time "
            "period) or the planes' vcpmg values are wrong.")
    ncyc = np.round(ncyc)
    plane_ncyc = {0: 0.0, **{i + 1: n for i, n in enumerate(ncyc)}}

    n_planes = fit_results["plane"].max()
    if n_planes != len(vcpmg_values):
        raise ValueError(f"fit_results has {n_planes} CPMG planes but "
                         f"{len(vcpmg_values)} vcpmg values were given")

    labels = {}
    if ref_df is not None and label_col in ref_df.columns:
        labels = ref_df.set_index("ref_index")[label_col].dropna().to_dict()

    profiles = {}
    n_skipped = 0
    for ref_index, grp in fit_results.groupby("ref_index"):
        label = labels.get(ref_index)
        if label is None or (isinstance(label, float) and np.isnan(label)):
            if not include_unassigned:
                n_skipped += 1
                continue
            name = f"peak{ref_index}"
        else:
            name = profile_name(str(label), atoms)
        if name in profiles:
            warnings.warn(f"duplicate profile name {name} (ref_index {ref_index}); "
                          "keeping the first.", UserWarning)
            continue
        grp = grp.dropna(subset=["vol", "vol_err"]).sort_values("plane")
        profiles[name] = pd.DataFrame({
            "ncyc": [plane_ncyc[int(p)] for p in grp["plane"]],
            "intensity": grp["vol"].to_numpy(float),
            "error": grp["vol_err"].to_numpy(float),
            "ref_index": ref_index,
        })
    if n_skipped:
        print(f"  skipped {n_skipped} unassigned peaks")
    return profiles


def write_profile(profile, path):
    """Write one profile DataFrame (``ncyc, intensity, error``) as a ChemEx ``.out``."""
    lines = [f"{n:12.3e}  {v:15.7e}  {e:13.7e}\n"
             for n, v, e in zip(profile["ncyc"], profile["intensity"], profile["error"])]
    Path(path).write_text(PROFILE_HEADER + "".join(lines))


def write_profiles(profiles, out_dir):
    """Write every profile to ``out_dir/<name>.out``; returns name -> Path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, profile in profiles.items():
        path = out_dir / f"{name}.out"
        write_profile(profile, path)
        written[name] = path
    return written


def save_chemex(fit_results, vcpmg_values, time_T2, out_dir, ref_df=None,
                label_col="assn_label", atoms=("N", "H"),
                include_unassigned=False, profiles_file="profiles.toml"):
    """
    Write fit_peaks output as ChemEx CPMG ``.out`` files, one per peak, for
    running ChemEx by hand. To run ChemEx from Python use
    :class:`makeshift.chemex.ChemEx` instead.

    ``profiles_file`` additionally writes a ``[data.profiles]`` TOML snippet
    to paste into a ChemEx experiment file (None to skip).

    Returns a dict mapping profile name -> written Path.
    """
    profiles = cpmg_profiles(fit_results, vcpmg_values, time_T2, ref_df=ref_df,
                             label_col=label_col, atoms=atoms,
                             include_unassigned=include_unassigned)
    written = write_profiles(profiles, out_dir)
    if profiles_file and written:
        snippet = "[data.profiles]\n" + "".join(
            f'{_toml_key(name)} = "{p.name}"\n' for name, p in written.items())
        (Path(out_dir) / profiles_file).write_text(snippet)
    print(f"  Wrote {len(written)} ChemEx profiles → {out_dir}")
    return written


# --------------------------------------------------------------------------
# TOML writing (small subset: tables of scalars / lists / inline tables)
# --------------------------------------------------------------------------

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(key):
    key = str(key)
    return key if _BARE_KEY.match(key) else '"' + key.replace('"', '\\"') + '"'


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            raise ValueError("cannot write NaN to TOML")
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(float(value))
    if isinstance(value, (str, Path)):
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_toml_key(k)} = {_toml_value(v)}"
                                for k, v in value.items()) + " }"
    raise TypeError(f"cannot write {type(value).__name__} to TOML")


def dumps_toml(doc):
    """
    Serialize ``{key: scalar, table: {key: value, subtable: {...}}}``.
    Top-level scalars come first; nested dicts become ``[table.subtable]``.
    """
    out = []

    def emit_table(prefix, table):
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        tables = {k: v for k, v in table.items() if isinstance(v, dict)}
        if prefix and (scalars or not tables):
            out.append(f"\n[{'.'.join(_toml_key(p) for p in prefix)}]")
        for k, v in scalars.items():
            out.append(f"{_toml_key(k)} = {_toml_value(v)}")
        for k, v in tables.items():
            emit_table(prefix + [k], v)

    emit_table([], doc)
    return "\n".join(out).lstrip("\n") + "\n"


def write_experiment_toml(path, experiment, conditions, profiles, data_path,
                          error="file"):
    """
    Write a ChemEx experiment file.

    ``experiment`` holds the ``[experiment]`` settings (``name``, ``time_t2``,
    ``carrier`` in ppm, ``pw90``, ...); ``conditions`` the ``[conditions]``
    (``h_larmor_frq`` in MHz, ``temperature`` in °C, ...); ``profiles`` maps
    profile name -> ``.out`` filename relative to ``data_path``.
    """
    doc = {
        "experiment": dict(experiment),
        "conditions": {k: v for k, v in conditions.items() if v is not None},
        "data": {"path": str(data_path), "error": error,
                 "profiles": {name: str(fn) for name, fn in profiles.items()}},
    }
    Path(path).write_text(dumps_toml(doc))


def write_parameters_toml(path, parameters):
    """
    Write a ChemEx parameters file from ``{section: {key: value}}``, e.g.
    ``{"GLOBAL": {"PB": 0.05, "KEX_AB": 500.0}, "DW_AB": {"Y4N": 1.0}}``.
    Values may be ``[value, min, max]``.
    """
    Path(path).write_text(dumps_toml({s: dict(v) for s, v in parameters.items()}))


def write_method_toml(path, steps):
    """
    Write a ChemEx v2 method file from ``{step_name: step_settings}``, e.g.
    ``{"STEP1": {"ROLES": [{"FIT": ["PB", "KEX_AB", "DW_AB", "R2_A"]}]}}``.
    """
    Path(path).write_text(dumps_toml({"FORMAT_VERSION": 2, **steps}))


# --------------------------------------------------------------------------
# output parsing
# --------------------------------------------------------------------------

_SECTION = re.compile(r'^\[\s*"?([^"\]]+)"?\s*\]\s*$')
_ENTRY = re.compile(r'^\s*("?)([^"=]+?)\1\s*=\s*([^#]+?)\s*(?:#\s*(.*))?$')
_ERR = re.compile(r"^±\s*([-+0-9.eEinfa]+)\s*(.*)$")


def _split_qualified(name):
    """'KEX_AB, T->25.0C' -> ('KEX_AB', 'T->25.0C')."""
    head, _, rest = name.partition(",")
    return head.strip(), rest.strip()


def parse_parameters_toml(path, kind=None):
    """
    Parse a ChemEx ``Parameters/*.toml`` output (fitted / fixed / constrained)
    into a long DataFrame.

    Columns: ``param`` (e.g. ``KEX_AB``), ``spin`` (``Y4N`` / ``Y4N-H``, or
    None for global parameters), ``value``, ``error`` (NaN when not reported),
    ``conditions`` (ChemEx's qualifier string, e.g. ``T->25.0C, B0->800.0MHZ``),
    ``note`` (any other trailing comment, e.g. a constraint expression), and
    ``kind`` (the file stem unless given).
    """
    path = Path(path)
    kind = kind or path.stem
    rows = []
    section = None
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _SECTION.match(line.strip())
        if m:
            section = m.group(1).strip()
            continue
        m = _ENTRY.match(line)
        if not m or section is None:
            continue
        key, raw_value, comment = m.group(2).strip(), m.group(3), m.group(4) or ""
        try:
            value = float(raw_value)
        except ValueError:
            continue
        error, note = np.nan, comment.strip()
        e = _ERR.match(note)
        if e:
            error, note = float(e.group(1)), e.group(2).strip()
        if note == "(fixed)":
            note = ""
        if section.upper() == "GLOBAL":
            param, cond = _split_qualified(key)
            spin = None
        else:
            param, cond = _split_qualified(section)
            spin = key
        rows.append(dict(param=param, spin=spin, value=value, error=error,
                         conditions=cond, note=note, kind=kind))
    return pd.DataFrame(rows, columns=["param", "spin", "value", "error",
                                       "conditions", "note", "kind"])


def parse_dat(path, time_t2=None):
    """
    Parse a ChemEx ``Data/<experiment>.dat`` output into a long DataFrame
    with columns ``profile, ncyc, intensity, error, calc``. For fits all
    four are present; for simulations only ``calc``.

    With ``time_t2`` (s), also adds ``vcpmg`` (Hz) and R2eff columns
    ``r2eff``, ``r2eff_err`` and ``r2eff_calc`` (s⁻¹) computed against each
    profile's ncyc 0 reference.
    """
    frames = []
    for block in Path(path).read_text().strip().split("\n\n"):
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        if not lines or not lines[0].startswith("["):
            continue
        name = lines[0].strip().strip("[]")
        header = lines[1].lstrip("#").upper() if len(lines) > 1 else ""
        data = np.array([[float(x) for x in ln.split()]
                         for ln in lines[1:] if not ln.lstrip().startswith("#")])
        if data.size == 0:
            continue
        if "EXP" in header:
            df = pd.DataFrame(data[:, :4], columns=["ncyc", "intensity", "error", "calc"])
        else:
            df = pd.DataFrame({"ncyc": data[:, 0], "calc": data[:, 1]})
        df.insert(0, "profile", name)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["profile", "ncyc", "intensity", "error", "calc"])
    out = pd.concat(frames, ignore_index=True)
    if time_t2:
        out["vcpmg"] = out["ncyc"] / time_t2
        out = pd.concat([_add_r2eff(g, time_t2)
                         for _, g in out.groupby("profile", sort=False)])
    return out.reset_index(drop=True)


def _add_r2eff(g, time_t2):
    g = g.copy()
    ref = g[g["ncyc"] == 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        if "intensity" in g and len(ref):
            i0 = ref["intensity"].mean()
            g["r2eff"] = -np.log(g["intensity"] / i0) / time_t2
            g["r2eff_err"] = g["error"] / (g["intensity"] * time_t2)
        if len(ref):
            c0 = ref["calc"].mean()
            g["r2eff_calc"] = -np.log(g["calc"] / c0) / time_t2
    ref_mask = g["ncyc"] == 0
    for col in ("r2eff", "r2eff_err", "r2eff_calc"):
        if col in g:
            g.loc[ref_mask, col] = np.nan
    return g


def parse_statistics(path):
    """Read ``statistics.toml`` into a plain dict."""
    with open(path, "rb") as f:
        return tomllib.load(f)
