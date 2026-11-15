"""
ChemEx engine: the :class:`ChemEx` object that fits CPMG relaxation-dispersion
data by running the ChemEx program (https://github.com/gbouvignies/ChemEx).

:class:`ChemEx` holds per-residue CPMG profiles plus the experiment settings
and, on :meth:`ChemEx.run`, writes ChemEx's input files (via :mod:`.utils`),
runs ``chemex fit`` as a subprocess, and parses the fitted parameters,
statistics and fitted profiles back into DataFrames. ChemEx has no stable
Python API, so the command line is the boundary; stateless file-format helpers
live in :mod:`makeshift.chemex.utils`.

ChemEx must be installed in the same environment (``pip install chemex``).
"""

import importlib.util
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import utils

CHEMEX_INFO_URL = "https://github.com/gbouvignies/ChemEx"

# Parameters each model needs fitted when no method is given.
DEFAULT_FIT = {
    "1st": ["R2_A"],
    # R2_A isn't listed: ChemEx always fits it for CPMG experiments. CS_A is
    # left out (fixed at the peak shifts); add it to float within cs_window.
    "2st": ["PB", "KEX_AB", "DW_AB"],
}

# Default kex / pB starting values and (min, max) bounds for ChemEx.run();
# None leaves that side unbounded. pB <= 0.2: a minor state of at most 20%.
DEFAULT_KEX = 500.0
DEFAULT_KEX_BOUNDS = (10.0, 10000.0)
DEFAULT_PB = 0.05
DEFAULT_PB_BOUNDS = (0.0, 0.2)
# CS_A starts at each peak's 15N shift. If a method fits it, it is bounded to
# ±DEFAULT_CS_WINDOW ppm: CPMG barely constrains it, and unbounded fits fail
# to converge.
DEFAULT_CS_WINDOW = 0.5
_RUN_ARG_PARAMS = {"KEX_AB": "kex / kex_bounds", "PB": "pb / pb_bounds"}
DEFAULT_DW = 1.0  # ppm; ChemEx's own default of 0 sits on a flat gradient


def chemex_command():
    """The command prefix that runs ChemEx, preferring this interpreter's."""
    if importlib.util.find_spec("chemex") is not None:
        return [sys.executable, "-m", "chemex"]
    exe = shutil.which("chemex")
    if exe:
        return [exe]
    raise RuntimeError(
        "ChemEx is not installed in this environment. Install it with "
        f"`pip install chemex` (see {CHEMEX_INFO_URL})."
    )


def _run_chemex(experiment_file, parameters_file, method_file, output_dir,
                model="2st", plot="nothing", include=None, exclude=None,
                workers=None, extra_args=None, cwd=None):
    cmd = [*chemex_command(), "fit",
           "-e", str(experiment_file),
           "-p", str(parameters_file),
           "-o", str(output_dir),
           "-d", model,
           "--plot", plot]
    if method_file is not None:
        cmd.extend(["-m", str(method_file)])
    if include:
        cmd.extend(["--include", *map(str, include)])
    if exclude:
        cmd.extend(["--exclude", *map(str, exclude)])
    if workers is not None:
        cmd.extend(["--workers", str(workers)])
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env.setdefault("NO_COLOR", "1")
    env.setdefault("COLUMNS", "120")
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


def _collect_step(step_dir, time_t2):
    """Parse one result directory (the run root, or ``STEPn`` for multi-step)."""
    step_dir = Path(step_dir)
    out = {"dir": str(step_dir)}

    params = []
    for kind in ("fitted", "constrained", "fixed"):
        path = step_dir / "Parameters" / f"{kind}.toml"
        if path.is_file():
            params.append(utils.parse_parameters_toml(path, kind=kind))
    out["parameters"] = (pd.concat(params, ignore_index=True) if params else
                         pd.DataFrame(columns=["param", "spin", "value", "error",
                                               "conditions", "note", "kind"]))

    stats = step_dir / "statistics.toml"
    out["statistics"] = utils.parse_statistics(stats) if stats.is_file() else None

    dats = sorted((step_dir / "Data").glob("*.dat"))
    out["profiles"] = (pd.concat([utils.parse_dat(p, time_t2=time_t2).assign(
        experiment=p.stem) for p in dats], ignore_index=True) if dats else None)
    return out


def _collect_outputs(output_dir, time_t2):
    """Parse a ``chemex fit`` output directory; the last step is the result."""
    output_dir = Path(output_dir)
    step_dirs = sorted(p for p in output_dir.iterdir()
                       if p.is_dir() and (p / "Parameters").is_dir())
    if (output_dir / "Parameters").is_dir():
        steps = {"": _collect_step(output_dir, time_t2)}
    else:
        steps = {p.name: _collect_step(p, time_t2) for p in step_dirs}
    if not steps:
        return {"output_dir": str(output_dir), "steps": {}}
    final = steps[list(steps)[-1]]
    return {"output_dir": str(output_dir), "steps": steps, **final}



def _fit_line(parameters):
    """'kex 812 ± 95 s⁻¹, pB 0.048 ± 0.006, dw 1.95 ± 0.15 ppm' from one fit."""
    fitted = parameters[parameters["kind"] == "fitted"]

    def fmt(param, digits, unit=""):
        row = fitted[fitted["param"] == param]
        if row.empty:
            return None
        value, err = row["value"].iloc[0], row["error"].iloc[0]
        err_txt = f" ± {err:.{digits}f}" if np.isfinite(err) else ""
        return f"{value:.{digits}f}{err_txt}{unit}"

    parts = [(label, fmt(param, digits, unit)) for label, param, digits, unit in (
        ("kex", "KEX_AB", 0, " s⁻¹"), ("pB", "PB", 3, ""), ("dw", "DW_AB", 2, " ppm"),
        ("R2B", "R2_B", 1, " s⁻¹"))]
    return ", ".join(f"{label} {text}" for label, text in parts if text)


def _pm(value, err, digits):
    err_txt = f" ± {err:.{digits}f}" if np.isfinite(err) else ""
    return f"{value:.{digits}f}{err_txt}"


def _panel_values(params, spin, individual):
    """Short per-residue fit summary for a plot panel title."""
    if params is None or spin not in params.index:
        return ""
    r = params.loc[spin]
    def txt(items):
        return ", ".join(f"{label} {_pm(r[col], r.get(col + '_err', np.nan), d)}"
                         for label, col, d in items if col in r and np.isfinite(r[col]))
    lines = [txt([("kex", "KEX_AB", 0), ("pB", "PB", 3)])] if individual else []
    if "R2_B" in r and np.isfinite(r["R2_B"]):
        lines.append(txt([("Δω", "DW_AB", 2)]))
        lines.append(txt([("R₂A", "R2_A", 1), ("R₂B", "R2_B", 1)]))
    else:
        lines.append(txt([("Δω", "DW_AB", 2), ("R₂", "R2_A", 1)]))
    return "\n".join(line for line in lines if line)


def _global_values(g):
    parts = []
    for label, param, digits, unit in (("kex", "KEX_AB", 0, " s⁻¹"), ("pB", "PB", 3, "")):
        if param in g.index:
            parts.append(f"{label} {_pm(g.loc[param, 'value'], g.loc[param, 'error'], digits)}{unit}")
    return ", ".join(parts)


def _combine_individual(per):
    """Merge per-residue run outputs into one results dict."""
    params = [o["parameters"].assign(group=spin) for spin, o in per.items()]
    profiles = [o["profiles"] for o in per.values() if o.get("profiles") is not None]
    stats = {spin: o["statistics"] for spin, o in per.items() if o.get("statistics")}
    return {
        "parameters": pd.concat(params, ignore_index=True),
        "profiles": pd.concat(profiles, ignore_index=True) if profiles else None,
        "statistics": pd.DataFrame.from_dict(stats, orient="index") if stats else None,
        "runs": per,
    }



def exchange_start(kex=DEFAULT_KEX, kex_bounds=DEFAULT_KEX_BOUNDS,
                   pb=DEFAULT_PB, pb_bounds=DEFAULT_PB_BOUNDS):
    """
    ChemEx ``[value, min, max]`` entries for ``KEX_AB`` and ``PB``.

    Bounds are ``(min, max)``; None (or the whole bounds argument None) leaves
    that side unbounded. Raises ValueError for a start outside its bounds or
    pB bounds outside [0, 1].
    """
    def entry(name, value, bounds, lo_floor=None, hi_ceiling=None):
        value = float(value)
        lo, hi = (None, None) if bounds is None else bounds
        lo = lo_floor if lo is None else float(lo)
        hi = hi_ceiling if hi is None else float(hi)
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError(f"{name} bounds must have min < max, got ({lo}, {hi})")
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            raise ValueError(f"{name} start {value} is outside its bounds ({lo}, {hi})")
        if lo is None and hi is None:
            return value
        return [value, -math.inf if lo is None else lo, math.inf if hi is None else hi]

    pb_entry = entry("pb", pb, pb_bounds, lo_floor=0.0, hi_ceiling=1.0)
    if isinstance(pb_entry, list) and (pb_entry[1] < 0 or pb_entry[2] > 1):
        raise ValueError(f"pb bounds must lie within [0, 1], got {tuple(pb_entry[1:])}")
    return {"PB": pb_entry, "KEX_AB": entry("kex", kex, kex_bounds)}


class ChemEx:
    """
    Fit CPMG relaxation dispersion with ChemEx.

    Usually built from a :class:`~makeshift.relaxation.CPMGExperiment` after
    its :meth:`~makeshift.relaxation.CPMGExperiment.run`. The ChemEx settings
    come from its config YAML: the Bruker ``acqus`` file supplies the field,
    carrier, temperature, pw90 and delays, and an optional ``chemex:``
    section picks the experiment/model or overrides them::

        acqus: acqus
        chemex:
          experiment: cpmg_15n_tr   # default cpmg_15n_ip

    then::

        cx = cpmg.to_chemex()     # or ChemEx.from_cpmg(cpmg)
        cx.run(residues=["Y4", "G10", "L22"], fit="global")   # or "individual"
        cx.global_parameters      # kex, pB, ...
        cx.residue_parameters     # per-residue dw, R2 (+ errors)
        cx.fit_profiles           # experimental vs fitted intensities / R2eff

    or from lineshape fits directly with
    ``ChemEx.from_fit_results(fit_results, ref_df, config)``.

    Parameters
    ----------
    profiles : dict
        Profile name (``Y4N-H``) -> DataFrame with ``ncyc, intensity, error``.
    experiment : dict
        ChemEx ``[experiment]`` settings: ``name`` (e.g. ``"cpmg_15n_ip"``),
        ``time_t2`` (s), ``carrier`` (15N, ppm), ``pw90`` (s), ...
    conditions : dict
        ChemEx ``[conditions]``: ``h_larmor_frq`` (MHz), ``temperature`` (°C).
    shifts : dict, optional
        Spin (``Y4N``) -> 15N shift in ppm, used as each residue's ``CS_A``.
        Strongly recommended: with no shifts ChemEx assumes 0 ppm, i.e. every
        residue ~118 ppm off resonance, which badly biases the fit.
    model : str
        ChemEx kinetic model (``"2st"``, ``"1st"``, ``"3st"``, ...).
    parameters : dict, optional
        Overrides for the starting parameters, by ChemEx section, e.g.
        ``{"DW_AB": 2.0, "R2_A": [12.0, 0.0, 50.0]}``. A scalar for a
        per-spin section applies to every spin. Values may be
        ``[value, min, max]``. kex and pB are not set here but with
        :meth:`run`'s ``kex`` / ``kex_bounds`` / ``pb`` / ``pb_bounds``.
    method : list, dict, str or Path, optional
        What to fit. A list of parameter names fits them in one step; a dict
        is written as ChemEx v2 method steps
        (``{"STEP1": {"ROLES": [{"FIT": [...]}]}}``); a path is passed through
        as an existing method file. Default: ``DEFAULT_FIT[model]``.
    """

    def __init__(self, profiles, experiment, conditions, shifts=None,
                 model="2st", parameters=None, method=None, name="cpmg"):
        self.profiles = dict(profiles)
        self.experiment = dict(experiment)
        self.conditions = dict(conditions)
        self.shifts = dict(shifts or {})
        self.model = model
        self.parameters = parameters or {}
        self.method = method
        self.name = name
        self.results = None

        if not self.profiles:
            raise ValueError("no profiles to fit (were any peaks assigned?)")
        for section, values in self.parameters.items():
            if section.upper() == "GLOBAL" and isinstance(values, dict):
                clash = [k for k in values if k.upper() in _RUN_ARG_PARAMS]
                if clash:
                    raise ValueError(
                        f"set {clash} with ChemEx.run("
                        + ", ".join(_RUN_ARG_PARAMS[k.upper()] for k in clash)
                        + "), not in parameters / the yaml")
        if "name" not in self.experiment:
            raise ValueError("experiment must include a ChemEx experiment 'name'")

    # ---------------------------------------------------------------- build

    @classmethod
    def from_fit_results(cls, fit_results, ref_df, config, *, ref_spectrum=None,
                         label_col="assn_label", atoms=("N", "H"),
                         shift_col="N_ppm", **overrides):
        """
        Build from :func:`~makeshift.relaxation.fit_peaks` output.

        Parameters
        ----------
        fit_results : DataFrame
            Output of ``fit_peaks``.
        ref_df : DataFrame
            The assigned peak table passed to ``fit_peaks``: ``ref_index``,
            ``label_col`` (e.g. ``Y4``) and ``shift_col`` (15N ppm).
        config : dict, str or Path
            The CPMG config (from :func:`~makeshift.relaxation.load_config`,
            or the YAML path). Supplies ``time_T2``, the νCPMG values of the
            planes, and the ChemEx settings (from ``acqus`` and/or the
            ``chemex:`` section).
        ref_spectrum : Spectrum, optional
            The already-loaded reference spectrum, used for the 1H field and
            15N carrier when neither acqus nor the ``chemex:`` section gives
            them; otherwise ``config["reference"]`` is read for its header.
        **overrides
            Replace individual ChemEx settings, e.g. ``model="1st"``.
        """
        if isinstance(config, (str, Path)):
            from ..relaxation.config import load_config
            config = load_config(config)
        time_T2 = float(config["time_T2"])
        # load_planes orders the planes by vcpmg; fit_results' planes follow it
        vcpmg_values = sorted(float(p["vcpmg"]) for p in config["planes"])

        settings = {**(config.get("chemex") or {}), **overrides}
        if any(settings.get(k) is None for k in ("h_larmor_frq", "carrier")):
            if ref_spectrum is None and config.get("reference") is not None:
                from ..spectra.spectrum import Spectrum
                ref_spectrum = Spectrum.from_ucsf(config["reference"])
            hdr_h, hdr_carrier = _header_field_and_carrier(ref_spectrum)
            for key, value, what in (("h_larmor_frq", hdr_h, "1H field, MHz"),
                                     ("carrier", hdr_carrier, "15N carrier, ppm")):
                if settings.get(key) is None:
                    if value is None:
                        raise ValueError(
                            f"could not read the {what} from the reference spectrum "
                            f"header; give `acqus:` or add {key} to the chemex section")
                    print(f"  chemex: {key} = {value:.3f} ({what}) from the spectrum header")
                    settings[key] = value
        s = utils.normalize_settings(settings, source="chemex settings")

        profiles = utils.cpmg_profiles(fit_results, vcpmg_values, time_T2,
                                       ref_df=ref_df, label_col=label_col,
                                       atoms=atoms)
        shifts = {}
        if shift_col in ref_df.columns and label_col in ref_df.columns:
            for label, ppm in ref_df[[label_col, shift_col]].dropna().itertuples(index=False):
                shifts[utils.spin_name(str(label), atoms)] = float(ppm)
        experiment = {"name": s["experiment"], "time_t2": time_T2,
                      "carrier": s["carrier"], "pw90": s["pw90"],
                      **s.get("experiment_settings", {})}
        conditions = {"h_larmor_frq": s["h_larmor_frq"],
                      "temperature": s["temperature"], **s.get("conditions", {})}
        return cls(profiles, experiment, conditions, shifts=shifts,
                   model=s["model"], parameters=s.get("parameters"),
                   method=s.get("method"))

    @classmethod
    def from_cpmg(cls, cpmg, **overrides):
        """
        Build from a :class:`~makeshift.relaxation.CPMGExperiment` that has
        been :meth:`run`, using its config (see :meth:`from_fit_results`).
        Keyword ``overrides`` (e.g. ``model="1st"``) replace individual
        ChemEx settings.
        """
        if cpmg.fit_results is None:
            raise RuntimeError("CPMGExperiment has no fit results; call run() first.")
        return cls.from_fit_results(cpmg.fit_results, cpmg.ref_df, cpmg.config,
                                    ref_spectrum=cpmg.ref_spectrum, **overrides)

    # ---------------------------------------------------------------- inputs

    @property
    def spins(self):
        """Spin names (``Y4N``) of the profiles, in order."""
        return [name.split("-")[0] for name in self.profiles]

    def select(self, residues=None):
        """
        Profile names for ``residues``, in profile order.

        Each entry may be a residue label (``"Y4"``), spin (``"Y4N"``),
        profile name (``"Y4N-H"``) or residue number (``4``). None selects
        every profile. Raises ValueError for residues with no profile.
        """
        names = list(self.profiles)
        if residues is None:
            return names
        if isinstance(residues, (str, int)):
            residues = [residues]
        lookup = {}
        for name in names:
            spin = name.split("-")[0]
            m = re.match(r"^([A-Za-z]*)(\d+)", spin)
            keys = {name.upper(), spin.upper()}
            if m:
                keys |= {(m.group(1) + m.group(2)).upper(), int(m.group(2))}
            for key in keys:
                lookup.setdefault(key, []).append(name)
        chosen, missing = set(), []
        for r in residues:
            key = int(r) if isinstance(r, (int, np.integer)) else str(r).strip().upper()
            if isinstance(key, str) and key.isdigit():
                key = int(key)
            if key in lookup:
                chosen.update(lookup[key])
            else:
                missing.append(r)
        if missing:
            raise ValueError(f"no assigned profile for residue(s) {missing}; "
                             f"available: {[n.split('-')[0] for n in names]}")
        return [n for n in names if n in chosen]

    def build_parameters(self, names=None, exchange=None, cs_window=DEFAULT_CS_WINDOW):
        """The starting-parameter sections that will be written for ChemEx
        (for the profiles in ``names``; default all). ``exchange`` holds the
        ``KEX_AB`` / ``PB`` entries (default :func:`exchange_start`);
        ``CS_A`` starts at each peak's 15N shift, bounded to ±``cs_window``
        ppm (None: unbounded)."""
        names = list(self.profiles) if names is None else names
        spins = [n.split("-")[0] for n in names]
        if self.model == "1st":
            sections = {"GLOBAL": {}}
        else:
            sections = {"GLOBAL": dict(exchange_start() if exchange is None else exchange)}
        if self.shifts:
            sections["CS_A"] = {
                s: (self.shifts[s] if cs_window is None else
                    [self.shifts[s], self.shifts[s] - cs_window, self.shifts[s] + cs_window])
                for s in spins if s in self.shifts}
            missing = [s for s in spins if s not in self.shifts]
            if missing:
                print(f"  no 15N shift for {len(missing)} spins (CS_A left at "
                      f"ChemEx's default): {missing[:5]}")
        if self.model != "1st":
            sections["DW_AB"] = {s: DEFAULT_DW for s in spins}
        for section, values in self.parameters.items():
            if isinstance(values, dict):
                if section.upper() == "GLOBAL":
                    sections["GLOBAL"].update(values)
                else:
                    sections.setdefault(section, {}).update(
                        {k: v for k, v in values.items() if k in spins})
            elif section.upper() == "GLOBAL":
                raise ValueError("GLOBAL parameters must be a dict")
            else:
                sections[section] = {s: values for s in spins}
        return {k: v for k, v in sections.items() if v}

    def build_method(self, separate_r2=False):
        """
        The v2 method steps, or a Path to a user method file.

        With ``separate_r2`` the minor state's ``R2_B`` is added to the
        parameters fitted (in the last step), so it is fitted on its own
        instead of being held equal to ``R2_A``.
        """
        method = self._base_method()
        if not separate_r2:
            return method
        if isinstance(method, Path):
            raise ValueError("separate_r2 can't be added to a method file; "
                             "list R2_B under FIT in the file instead")
        if self.model == "1st":
            raise ValueError("separate_r2 needs a two-state model (there is no R2_B)")
        method = {step: dict(body) for step, body in method.items()}
        last = method[list(method)[-1]]
        roles = [dict(r) for r in last.get("ROLES", [{}])]
        fit = list(roles[0].get("FIT", []))
        if "R2_B" not in [str(f).upper() for f in fit]:
            fit.append("R2_B")
        roles[0]["FIT"] = fit
        last["ROLES"] = roles
        return method

    def _base_method(self):
        method = self.method
        if method is None:
            if self.model not in DEFAULT_FIT:
                raise ValueError(f"no default method for model {self.model!r}; "
                                 "pass method=[...parameters to fit...]")
            method = DEFAULT_FIT[self.model]
        if isinstance(method, (str, Path)):
            return Path(method)
        if isinstance(method, (list, tuple)):
            return {"STEP1": {"ROLES": [{"FIT": list(method)}]}}
        return dict(method)

    def write_inputs(self, input_dir, names=None, exchange=None,
                     cs_window=DEFAULT_CS_WINDOW, separate_r2=False):
        """Write the ChemEx input files for the profiles in ``names``
        (default all) into ``input_dir``; returns their paths."""
        names = list(self.profiles) if names is None else names
        input_dir = Path(input_dir)
        data_dir = input_dir / "data"
        written = utils.write_profiles({n: self.profiles[n] for n in names}, data_dir)

        experiment_file = input_dir / f"{self.name}.toml"
        utils.write_experiment_toml(
            experiment_file, self.experiment, self.conditions,
            profiles={n: p.name for n, p in written.items()}, data_path="data")

        parameters_file = input_dir / "parameters.toml"
        utils.write_parameters_toml(parameters_file, self.build_parameters(names, exchange, cs_window))

        method = self.build_method(separate_r2)
        if isinstance(method, Path):
            method_file = method.resolve()
        else:
            method_file = input_dir / "method.toml"
            utils.write_method_toml(method_file, method)

        return {"experiment": experiment_file, "parameters": parameters_file,
                "method": method_file, "data": data_dir}

    # ---------------------------------------------------------------- run

    def run(self, residues=None, fit="global", kex=DEFAULT_KEX,
            kex_bounds=DEFAULT_KEX_BOUNDS, pb=DEFAULT_PB, pb_bounds=DEFAULT_PB_BOUNDS,
            cs_window=DEFAULT_CS_WINDOW, separate_r2=False, output_dir=None,
            plot=False, cleanup=False):
        """
        Fit the selected residues with ChemEx and parse the results.

        Parameters
        ----------
        residues : list, optional
            Residues to fit, e.g. ``["Y4", "G10", 22]`` (labels, spin names
            ``Y4N``, profile names ``Y4N-H`` or residue numbers). Default:
            every assigned residue.
        fit : {"global", "individual"}
            ``"global"``: one fit in which all residues share kex and pB
            (per-residue Δω and R₂). ``"individual"``: each residue is fitted
            on its own, with its own kex and pB.
        kex : float
            Starting kex (s⁻¹). Default 500.
        kex_bounds : (min, max)
            Bounds on kex; None on either side (or for both) leaves it open.
            Default ``(10, 1e4)``.
        pb : float
            Starting pB. Default 0.05.
        pb_bounds : (min, max)
            Bounds on pB, within [0, 1]. Default ``(0, 0.2)``: a minor state
            of at most 20%.
        cs_window : float or None
            Only used when the method fits ``CS_A`` (the default method keeps
            it fixed at each peak's 15N shift): it then floats within ± this
            many ppm of the shift. Default 0.5. None removes the bounds, which
            CPMG data usually can't support (the fit may not converge).

            For individual fits every residue starts from the same kex and pB.
        separate_r2 : bool
            Fit the minor state's intrinsic R₂ (``R2_B``) separately from the
            major state's (``R2_A``). By default ChemEx holds R2_B = R2_A.
            CPMG data constrain R2_B weakly, especially when pB is small, so
            expect larger errors (or none, at a bound) and less certain kex
            and pB. Compare chi²/dof with and without it.
        output_dir : str or Path, optional
            Where to write ChemEx's inputs and output (for ``"individual"``,
            one sub-folder per residue). Default: a temporary directory, whose
            path is in ``results["output_dir"]``.
        plot : bool
            Show the fitted dispersion profiles with matplotlib once the fit
            is done (see :meth:`plot`). Default False.
        cleanup : bool
            Delete the temporary directory once the results are parsed (it is
            kept if ChemEx fails, so the inputs can be inspected).
        """
        if fit not in ("global", "individual"):
            raise ValueError(f"fit must be 'global' or 'individual', got {fit!r}")
        names = self.select(residues)
        exchange = exchange_start(kex, kex_bounds, pb, pb_bounds)
        if cs_window is not None and not cs_window > 0:
            raise ValueError(f"cs_window must be > 0 ppm or None, got {cs_window}")

        temp_dir = None
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="chemex_")
            output_dir = Path(temp_dir)
        else:
            # absolute, because ChemEx runs with cwd set to the inputs folder
            output_dir = Path(output_dir).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

        if fit == "global":
            print(f"  ChemEx: global fit of {len(names)} residues")
            results = self._run_once(names, output_dir, exchange, cs_window,
                                     separate_r2)
            results["parameters"] = results["parameters"].assign(group="global")
        else:
            per = {}
            for i, name in enumerate(names, 1):
                spin = name.split("-")[0]
                print(f"  ChemEx: fitting {spin} ({i}/{len(names)})", end="", flush=True)
                per[spin] = self._run_once([name], output_dir / spin, exchange,
                                           cs_window, separate_r2)
                print(f"  {_fit_line(per[spin]['parameters'])}")
            results = _combine_individual(per)
        results.update(fit=fit, residues=[n.split("-")[0] for n in names],
                       initial={**(exchange if self.model != "1st" else {}),
                                "cs_window": cs_window},
                       separate_r2=separate_r2,
                       output_dir=str(output_dir))
        self.results = results
        if cleanup and temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            self.results["output_dir"] = None
        if plot:
            self.plot()
        return self

    def _run_once(self, names, output_dir, exchange=None,
                  cs_window=DEFAULT_CS_WINDOW, separate_r2=False):
        """One ``chemex fit`` on the profiles in ``names``."""
        output_dir.mkdir(parents=True, exist_ok=True)
        input_dir = output_dir / "inputs"
        files = self.write_inputs(input_dir, names, exchange, cs_window, separate_r2)
        chemex_dir = output_dir / "chemex"
        proc = _run_chemex(
            files["experiment"].name, files["parameters"].name,
            files["method"], chemex_dir.resolve(), model=self.model,
            plot="nothing", cwd=input_dir,
        )
        if proc.returncode != 0 or not chemex_dir.is_dir():
            raise RuntimeError(
                f"ChemEx failed with return code {proc.returncode}.\n"
                f"inputs: {input_dir}\n"
                f"output (last 3000 chars):\n{(proc.stdout + proc.stderr)[-3000:]}"
            )
        outputs = _collect_outputs(chemex_dir, self.experiment.get("time_t2"))
        outputs["chemex_dir"] = str(chemex_dir)
        outputs["input_files"] = {k: str(v) for k, v in files.items()}
        outputs["stdout"] = proc.stdout
        outputs["stderr"] = proc.stderr
        return outputs

    # ---------------------------------------------------------------- results

    @property
    def parameters_table(self):
        """Every parameter ChemEx reported (fitted, constrained and fixed),
        with a ``group`` column: ``"global"``, or the residue's spin name for
        individual fits."""
        return None if self.results is None else self.results.get("parameters")

    @property
    def global_parameters(self):
        """For a global fit: the shared parameters (kex, pB, pA, kAB, kBA, ...)
        as a DataFrame indexed by parameter with ``value, error, kind``.
        None for individual fits, whose kex and pB are per residue in
        :attr:`residue_parameters`."""
        df = self.parameters_table
        if df is None or self.results.get("fit") == "individual":
            return None
        g = df[df["spin"].isna() & df["kind"].isin(["fitted", "constrained"])]
        return g.set_index("param")[["value", "error", "kind"]]

    @property
    def residue_parameters(self):
        """Fitted parameters, one row per residue spin (``Y4N``), each with a
        ``_err`` column: ``DW_AB``, ``R2_A`` (and ``R2_B`` with
        ``run(separate_r2=True)``), and for individual fits also
        that residue's own ``KEX_AB`` and ``PB``. Includes ``ref_index``
        back into the peak table when known."""
        df = self.parameters_table
        if df is None:
            return None
        df = df[df["kind"] == "fitted"]
        if self.results.get("fit") == "individual":
            df = df.assign(spin=df["spin"].fillna(df["group"]))
        r = df[df["spin"].notna()]
        if r.empty:
            return pd.DataFrame()
        values = r.pivot_table(index="spin", columns="param", values="value", sort=False)
        errors = r.pivot_table(index="spin", columns="param", values="error",
                               sort=False, dropna=False).add_suffix("_err")
        wide = values.join(errors)
        wide = wide[[c for p in values.columns for c in (p, f"{p}_err")]]
        ref_index = {name.split("-")[0]: int(p["ref_index"].iloc[0])
                     for name, p in self.profiles.items() if "ref_index" in p}
        if ref_index:
            wide.insert(0, "ref_index", [ref_index.get(s) for s in wide.index])
        order = {s: i for i, s in enumerate(self.spins)}
        wide = wide.loc[sorted(wide.index, key=lambda s: order.get(s, len(order)))]
        wide.columns.name = None
        return wide

    def plot(self, residues=None, ncols=4, show=True):
        """
        Plot the fitted dispersion profiles with matplotlib, one panel per
        residue: R₂,eff vs νCPMG with error bars and the ChemEx fit, and the
        residuals (data − fit) above. Panel titles give the residue's fitted
        values (kex, pB and Δω for individual fits; Δω for global fits, with
        the shared kex and pB in the figure title).

        Parameters
        ----------
        residues : list, optional
            Subset to plot (same forms as :meth:`run`); default all fitted.
        ncols : int
            Panels per row.
        show : bool
            Display the figure (``plt.show()``) and return None. With False
            the figure is returned instead, e.g. to save it.
        """
        import matplotlib.pyplot as plt

        prof = self.fit_profiles
        if prof is None or prof.empty:
            raise RuntimeError("nothing to plot; call run() first")
        def key(name):
            return str(name).split("-")[0].strip().upper()

        fitted = {key(p): p for p in prof["profile"].unique()}
        wanted = (self.select(residues) if residues is not None
                  else self.results.get("residues") or list(self.profiles))
        # plot under the names ChemEx wrote, matched on the spin (Y92N)
        names = [fitted[key(n)] for n in wanted if key(n) in fitted]

        def resnum(name):
            m = re.search(r"\d+", str(name))
            return (int(m.group()) if m else float("inf"), str(name))

        names = sorted(dict.fromkeys(names), key=resnum)
        if not names:
            raise ValueError(
                f"none of the requested residues {[key(n) for n in wanted]} are in the "
                f"last run's fitted profiles {sorted(fitted)}")
        params = self.residue_parameters
        individual = self.results.get("fit") == "individual"

        ncols = max(1, min(ncols, len(names)))
        nrows = -(-len(names) // ncols)
        fig = plt.figure(figsize=(3.8 * ncols, 3.9 * nrows), layout="constrained")
        outer = fig.add_gridspec(nrows, ncols)
        for i, name in enumerate(names):
            row, col = divmod(i, ncols)
            cell = outer[row, col].subgridspec(2, 1, height_ratios=[1, 3], hspace=0.05)
            ax_res = fig.add_subplot(cell[0])
            ax = fig.add_subplot(cell[1], sharex=ax_res)
            d = prof[(prof["profile"] == name) & (prof["ncyc"] > 0)].sort_values("vcpmg")
            fit = d.drop_duplicates("vcpmg")
            ax.errorbar(d["vcpmg"], d["r2eff"], yerr=d["r2eff_err"], fmt="o", ms=4,
                        color="C3", ecolor="C3", elinewidth=1, capsize=0, zorder=3)
            ax.plot(fit["vcpmg"], fit["r2eff_calc"], "-", color="C0", lw=1.5, zorder=2)
            ax_res.errorbar(d["vcpmg"], d["r2eff"] - d["r2eff_calc"], yerr=d["r2eff_err"],
                            fmt="o", ms=3, color="C3", elinewidth=1, capsize=0)
            ax_res.axhline(0, color="k", lw=0.8)
            ax_res.tick_params(labelbottom=False)
            ax.set_xlim(left=0)
            if row == nrows - 1 or i + ncols >= len(names):
                ax.set_xlabel("νCPMG (Hz)")
            if col == 0:
                ax.set_ylabel("R₂,eff (s⁻¹)")
                ax_res.set_ylabel("resid.")
            spin = name.split("-")[0]
            ax_res.set_title(f"{spin}\n{_panel_values(params, spin, individual)}",
                             fontsize=8.5)
        if not individual and self.global_parameters is not None:
            fig.suptitle(f"global fit: {_global_values(self.global_parameters)}")
        if show:
            plt.show()
            return None
        return fig

    def exchange_summary(self):
        """
        After ``run(fit="individual")``: how well the residues' own kex and
        pB values agree, as a guide for the starting values of a global fit.

        Returns a DataFrame indexed by ``kex`` and ``pb`` with columns

        - ``mean``, ``mean_err``: error-weighted mean (weights 1/err²) and
          its standard error;
        - ``median``, ``min``, ``max``: unweighted;
        - ``chi2_dof``: weighted scatter about the mean, Σ((x-mean)/err)² /
          (n-1). Near 1 means the values agree within their errors; much
          larger means they don't, and they may not share one process;
        - ``n``: residues used (those with a finite, non-zero error).

        The per-residue values are in :attr:`residue_parameters`.
        """
        if self.results is None or self.results.get("fit") != "individual":
            raise RuntimeError('exchange_summary() needs results from run(fit="individual")')
        r = self.residue_parameters
        rows = {}
        for key, col in (("kex", "KEX_AB"), ("pb", "PB")):
            if col not in r.columns:
                continue
            x, e = r[col].to_numpy(float), r[f"{col}_err"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(e) & (e > 0)
            x, e = x[ok], e[ok]
            n = int(ok.sum())
            if n == 0:
                rows[key] = dict(mean=np.nan, mean_err=np.nan, median=np.nan,
                                 min=np.nan, max=np.nan, chi2_dof=np.nan, n=0)
                continue
            w = 1.0 / e ** 2
            mean = float(np.sum(w * x) / np.sum(w))
            rows[key] = dict(
                mean=mean, mean_err=float(1.0 / np.sqrt(np.sum(w))),
                median=float(np.median(x)), min=float(x.min()), max=float(x.max()),
                chi2_dof=(float(np.sum(w * (x - mean) ** 2) / (n - 1)) if n > 1 else np.nan),
                n=n)
        return pd.DataFrame.from_dict(rows, orient="index")

    @property
    def fit_profiles(self):
        """Experimental vs fitted profiles (``profile, ncyc, vcpmg, intensity,
        error, calc, r2eff, r2eff_err, r2eff_calc``)."""
        return None if self.results is None else self.results.get("profiles")

    @property
    def statistics(self):
        """χ², reduced χ², AIC, BIC, ... from ChemEx: a dict for a global fit,
        a DataFrame with one row per residue for individual fits."""
        return None if self.results is None else self.results.get("statistics")

    def __repr__(self):
        state = "run" if self.results is not None else "not run"
        return (f"<ChemEx {self.experiment.get('name')} model={self.model} "
                f"profiles={len(self.profiles)} ({state})>")


def _header_field_and_carrier(spectrum):
    """(1H MHz, 15N carrier ppm) from a Spectrum's nmrglue unit converters."""
    try:
        ucs = list(spectrum.uc)
        obs = [float(u._obs) for u in ucs]
        h = int(np.argmax(obs))
        n = int(np.argmin(obs))
        return obs[h], float(ucs[n]._car) / obs[n]
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return None, None
