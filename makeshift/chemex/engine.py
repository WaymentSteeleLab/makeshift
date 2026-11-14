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
import os
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
    "2st": ["PB", "KEX_AB", "DW_AB", "R2_A"],
}

# Starting values; PB is bounded to (0, 0.5) so A stays the major state.
DEFAULT_GLOBAL = {
    "1st": {},
    "2st": {"PB": [0.05, 0.0, 0.5], "KEX_AB": 500.0},
}
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
    out["plots"] = [str(p) for p in sorted((step_dir / "Plots").glob("*.pdf"))]
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


class ChemEx:
    """
    Fit CPMG relaxation dispersion with ChemEx.

    Build from a :class:`~makeshift.relaxation.CPMGExperiment` after its
    :meth:`~makeshift.relaxation.CPMGExperiment.run`, which reads the 1H
    field and 15N carrier from the reference spectrum's header::

        cx = ChemEx.from_cpmg(cpmg, pw90=35e-6, temperature=25.0)
        cx.run()
        cx.global_parameters      # kex, pB, ...
        cx.residue_parameters     # per-residue dw, R2 (+ errors)
        cx.fit_profiles           # experimental vs fitted intensities / R2eff

    or from lineshape fits directly with :meth:`from_fit_results`.

    Parameters
    ----------
    profiles : dict
        Profile name (``Y4N-H``) -> DataFrame with ``ncyc, intensity, error``.
    experiment : dict
        ChemEx ``[experiment]`` settings: ``name`` (e.g. ``"cpmg_15n_tr"``),
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
        ``{"GLOBAL": {"KEX_AB": 1500.0}, "DW_AB": 2.0}``. A scalar for a
        per-spin section applies to every spin. Values may be
        ``[value, min, max]``.
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
        if "name" not in self.experiment:
            raise ValueError("experiment must include a ChemEx experiment 'name'")

    # ---------------------------------------------------------------- build

    @classmethod
    def from_fit_results(cls, fit_results, vcpmg_values, time_T2, ref_df, *,
                         pw90, temperature, h_larmor_frq, carrier=None,
                         experiment="cpmg_15n_tr", label_col="assn_label",
                         atoms=("N", "H"), shift_col="N_ppm",
                         experiment_settings=None, **kwargs):
        """
        Build from :func:`~makeshift.relaxation.fit_peaks` output.

        ``ref_df`` names the peaks (``label_col``, e.g. ``Y4``) and supplies
        each residue's 15N shift (``shift_col``). ``carrier`` (15N, ppm)
        defaults to the midpoint of the observed 15N shifts; pass the real
        value when you know it. ``experiment_settings`` adds extra
        ``[experiment]`` keys (e.g. ``{"antitrosy": True}``); other keyword
        arguments go to :class:`ChemEx`.
        """
        profiles = utils.cpmg_profiles(fit_results, vcpmg_values, time_T2,
                                       ref_df=ref_df, label_col=label_col,
                                       atoms=atoms)
        shifts = {}
        if shift_col in ref_df.columns and label_col in ref_df.columns:
            for label, ppm in ref_df[[label_col, shift_col]].dropna().itertuples(index=False):
                shifts[utils.spin_name(str(label), atoms)] = float(ppm)
        if carrier is None:
            if not shifts:
                raise ValueError("pass carrier= (15N, ppm): no shifts to infer it from")
            vals = np.array(list(shifts.values()))
            carrier = float((vals.min() + vals.max()) / 2)
        settings = {"name": experiment, "time_t2": float(time_T2),
                    "carrier": float(carrier), "pw90": float(pw90),
                    **(experiment_settings or {})}
        conditions = {"h_larmor_frq": float(h_larmor_frq),
                      "temperature": float(temperature)}
        return cls(profiles, settings, conditions, shifts=shifts, **kwargs)

    @classmethod
    def from_cpmg(cls, cpmg, *, pw90, temperature, h_larmor_frq=None,
                  carrier=None, **kwargs):
        """
        Build from a :class:`~makeshift.relaxation.CPMGExperiment` that has
        been :meth:`run`. ``h_larmor_frq`` (MHz) and the 15N ``carrier`` (ppm)
        default to the values in the reference spectrum's header.
        """
        if cpmg.fit_results is None:
            raise RuntimeError("CPMGExperiment has no fit results; call run() first.")
        hdr_h, hdr_carrier = _header_field_and_carrier(cpmg.ref_spectrum)
        h_larmor_frq = h_larmor_frq if h_larmor_frq is not None else hdr_h
        carrier = carrier if carrier is not None else hdr_carrier
        if h_larmor_frq is None:
            raise ValueError("could not read the 1H field from the spectrum; "
                             "pass h_larmor_frq= (MHz)")
        return cls.from_fit_results(
            cpmg.fit_results, cpmg.vcpmg_values, cpmg.time_T2, cpmg.ref_df,
            pw90=pw90, temperature=temperature, h_larmor_frq=h_larmor_frq,
            carrier=carrier, **kwargs)

    # ---------------------------------------------------------------- inputs

    @property
    def spins(self):
        """Spin names (``Y4N``) of the profiles, in order."""
        return [name.split("-")[0] for name in self.profiles]

    def build_parameters(self):
        """The starting-parameter sections that will be written for ChemEx."""
        spins = self.spins
        sections = {"GLOBAL": dict(DEFAULT_GLOBAL.get(self.model, {}))}
        if self.shifts:
            sections["CS_A"] = {s: self.shifts[s] for s in spins if s in self.shifts}
            missing = [s for s in spins if s not in self.shifts]
            if missing:
                print(f"  no 15N shift for {len(missing)} spins (CS_A left at "
                      f"ChemEx's default): {missing[:5]}")
        if self.model != "1st":
            sections["DW_AB"] = {s: DEFAULT_DW for s in spins}
        for section, values in self.parameters.items():
            if isinstance(values, dict):
                sections.setdefault(section, {}).update(values)
            elif section.upper() == "GLOBAL":
                raise ValueError("GLOBAL parameters must be a dict")
            else:
                sections[section] = {s: values for s in spins}
        return {k: v for k, v in sections.items() if v}

    def build_method(self):
        """The v2 method steps, or a Path to a user method file."""
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

    def write_inputs(self, input_dir):
        """Write every ChemEx input file into ``input_dir``; returns their paths."""
        input_dir = Path(input_dir)
        data_dir = input_dir / "data"
        written = utils.write_profiles(self.profiles, data_dir)

        experiment_file = input_dir / f"{self.name}.toml"
        utils.write_experiment_toml(
            experiment_file, self.experiment, self.conditions,
            profiles={n: p.name for n, p in written.items()}, data_path="data")

        parameters_file = input_dir / "parameters.toml"
        utils.write_parameters_toml(parameters_file, self.build_parameters())

        method = self.build_method()
        if isinstance(method, Path):
            method_file = method.resolve()
        else:
            method_file = input_dir / "method.toml"
            utils.write_method_toml(method_file, method)

        return {"experiment": experiment_file, "parameters": parameters_file,
                "method": method_file, "data": data_dir}

    # ---------------------------------------------------------------- run

    def run(self, output_dir=None, plot="nothing", include=None, exclude=None,
            workers=None, extra_args=None, cleanup=False):
        """
        Write inputs, run ``chemex fit`` and parse the results.

        Inputs go to ``output_dir/inputs`` and ChemEx's own output to
        ``output_dir/chemex``. With no ``output_dir`` a temporary directory
        is used (its path is in ``results["output_dir"]``); ``cleanup=True``
        deletes it once the results are parsed (it is kept if ChemEx fails,
        so the inputs can be inspected).

        ``plot`` is ChemEx's plotting level (``"nothing"``, ``"normal"``,
        ``"all"``); ``include`` / ``exclude`` take spin names (``Y4N``) to
        restrict the fit; ``workers`` sets ChemEx worker processes.
        """
        temp_dir = None
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="chemex_")
            output_dir = Path(temp_dir)
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        input_dir = output_dir / "inputs"
        files = self.write_inputs(input_dir)
        chemex_dir = output_dir / "chemex"

        proc = _run_chemex(
            files["experiment"].name, files["parameters"].name,
            files["method"], chemex_dir.resolve(), model=self.model,
            plot=plot, include=include, exclude=exclude, workers=workers,
            extra_args=extra_args, cwd=input_dir,
        )
        if proc.returncode != 0 or not chemex_dir.is_dir():
            raise RuntimeError(
                f"ChemEx failed with return code {proc.returncode}.\n"
                f"inputs: {input_dir}\n"
                f"output (last 3000 chars):\n{(proc.stdout + proc.stderr)[-3000:]}"
            )

        outputs = _collect_outputs(chemex_dir, self.experiment.get("time_t2"))
        outputs["output_dir"] = str(output_dir)
        outputs["chemex_dir"] = str(chemex_dir)
        outputs["input_files"] = {k: str(v) for k, v in files.items()}
        outputs["returncode"] = proc.returncode
        outputs["stdout"] = proc.stdout
        outputs["stderr"] = proc.stderr
        self.results = outputs
        if cleanup and temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            self.results["output_dir"] = None
        return self

    # ---------------------------------------------------------------- results

    @property
    def parameters_table(self):
        """Every parameter ChemEx reported (fitted, constrained and fixed)."""
        return None if self.results is None else self.results.get("parameters")

    @property
    def global_parameters(self):
        """Fitted and derived global parameters (kex, pB, pA, kAB, kBA, ...)
        as a DataFrame indexed by parameter with ``value, error, kind``."""
        df = self.parameters_table
        if df is None:
            return None
        g = df[df["spin"].isna() & df["kind"].isin(["fitted", "constrained"])]
        return g.set_index("param")[["value", "error", "kind"]]

    @property
    def residue_parameters(self):
        """Fitted per-spin parameters, one row per spin: e.g. ``DW_AB``,
        ``DW_AB_err``, ``R2_A``, ``R2_A_err``, plus ``ref_index`` back into
        the peak table when known."""
        df = self.parameters_table
        if df is None:
            return None
        r = df[df["spin"].notna() & (df["kind"] == "fitted")]
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

    @property
    def fit_profiles(self):
        """Experimental vs fitted profiles (``profile, ncyc, vcpmg, intensity,
        error, calc, r2eff, r2eff_err, r2eff_calc``)."""
        return None if self.results is None else self.results.get("profiles")

    @property
    def statistics(self):
        """χ², reduced χ², AIC, BIC, ... from ChemEx's statistics.toml."""
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
