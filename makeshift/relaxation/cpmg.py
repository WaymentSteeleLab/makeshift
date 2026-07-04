"""
CPMG pipeline.

:class:`CPMGExperiment` ties the stages together for one protein: load
config/planes -> pick peaks -> (optional) map a reference assignment peaklist
-> fit lineshapes -> compute R2eff -> classify -> write a results CSV.

    exp = CPMGExperiment.from_config("exp.yml")
    exp.run("out/")
    exp.classifications   # results also available on the object
"""

import numpy as np
import pandas as pd
from pathlib import Path

from .config import load_config, load_planes
from .lineshape import fit_peaks
from .r2eff import compute_r2eff
from .classify import classify_peaks, validate_sequence, _parse_seqpos
from ..spectra.matching import map_peaklists

_DEFAULT_COLORS = {
    "Rex": "tab:orange",
    "elevated_R2": "#8ACE00",
    "flat": "black",
    "unfit": "red",
}


def _peaklist_df(peaklist):
    """Resolve a peaklist input to a DataFrame of assigned amides
    (Seq_ID, Auth_seq_ID, Comp_ID, H_ppm, N_ppm, assn_label).

    Accepts a :class:`PeakList`, a DataFrame, a BMRB id (int or digit-string),
    or a CSV path.
    """
    from ..peaklist import PeakList
    if isinstance(peaklist, PeakList):
        return peaklist.data
    if isinstance(peaklist, pd.DataFrame):
        return peaklist
    s = str(peaklist).strip()
    if s.isdigit():
        return PeakList.from_bmrb(int(s)).data
    return PeakList.from_csv(peaklist).data


def save_results_csv(ref_df, classifications, csv_path, fit_results=None, missing_df=None):
    """
    Write a per-peak summary CSV combining peak assignments and classification.

    Columns: ref_index, assn_label (if present), N_ppm, H_ppm, vol, vol_err,
    lw_n, lw_h (if fit_results given, from the reference plane),
    R2first, R2last, std_first, std_last, std_total, Rex_val, Rex_err, Rex,
    elevated_R2, label, seqpos, R2_hydro, scaled_R2_pred, rigid_rmse,
    missing_assignment. Rows are sorted by seqpos (peaks without a seqpos last).
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
    print(f"  Saved results CSV \u2192 {Path(csv_path).name}")
    return out


class CPMGExperiment:
    """
    One CPMG relaxation-dispersion experiment: its config, the loaded planes,
    and (after :meth:`run`) the fitted/classified results.

    Construct with :meth:`from_config`, then call :meth:`run`.
    """

    def __init__(self, config, ref_spectrum, plane_data, vcpmg_values, time_T2):
        self.config = config
        self.ref_spectrum = ref_spectrum
        self.plane_data = plane_data
        self.vcpmg_values = vcpmg_values
        self.time_T2 = time_T2

        # populated by run()
        self.ref_df = None
        self.fit_results = None
        self.all_r2eff = None
        self.classifications = None
        self.missing_df = None

    @classmethod
    def from_config(cls, yaml_path):
        """Load a CPMG config YAML (see :func:`load_config`) and read all of
        its UCSF planes into a ready-to-run experiment."""
        print("== Loading config and planes ==")
        config = load_config(yaml_path)
        ref_spectrum, plane_data, vcpmg_values, time_T2 = load_planes(config)
        return cls(config, ref_spectrum, plane_data, vcpmg_values, time_T2)

    def run(self, out_dir, lineshape="gaussian", max_r2err_threshold=10000.0,
            color_map=None, make_plots=False, peaklist=None,
            xlim=(10.5, 6.0), ylim=(135, 100),
            zoom_xlim=(9, 7), zoom_ylim=(125, 112)):
        """
        Run the full pipeline and write a results CSV to ``out_dir``.

        Steps: pick peaks -> (optional) reference-peaklist assignment -> fit
        lineshapes -> compute R2eff -> classify peaks (with HYDRONMR
        elevated_R2 classification) -> write the CSV. With ``make_plots=True``
        the diagnostic HSQC / dispersion / waterfall PDFs are also written
        (requires the plotting modules).

        Results are stored on the object (``ref_df``, ``fit_results``,
        ``all_r2eff``, ``classifications``) and ``self`` is returned.
        """
        color_map = color_map or dict(_DEFAULT_COLORS)
        config = self.config
        out_dir = Path(out_dir) / "outputs"
        cache_dir = out_dir.parent / "caches"
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = config["yaml_path"].stem

        print("== Picking peaks ==")
        ref_df = self.ref_spectrum.pick_peaks(baseline=config["baseline_ref_plane"])
        ref_df["ref_index"] = ref_df.index
        print(f"  {len(ref_df)} peaks picked")

        print("== Peak assignment ==")
        sequence = config.get("sequence")
        peaklist = peaklist if peaklist is not None else config.get("peaklist")
        missing_df = None
        if peaklist is not None and sequence:
            peaks_for_missing = _peaklist_df(peaklist)
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

        picked_df = ref_df.copy()
        peaks_shifted = None
        if peaklist is not None:
            peaks = _peaklist_df(peaklist)
            if sequence:
                validate_sequence(sequence, peaks)
            ref_df, peaks_shifted = map_peaklists(ref_df, peaks, tol=config["peak_mapping_tol"])
        else:
            print("  No peaklist provided or in config \u2014 skipping.")

        print("== Fitting lineshapes ==")
        fit_results = fit_peaks(self.plane_data, ref_df, self.ref_spectrum.uc, config,
                                lineshape=lineshape, cache_dir=cache_dir)

        print("== Computing R2eff ==")
        all_r2eff = compute_r2eff(fit_results, self.vcpmg_values, self.time_T2)

        print("== Classifying peaks ==")
        classifications = classify_peaks(all_r2eff, ref_df, config,
                                         max_r2err_threshold=max_r2err_threshold)
        print(classifications["label"].value_counts().to_string())

        # store results on the object
        self.ref_df = ref_df
        self.fit_results = fit_results
        self.all_r2eff = all_r2eff
        self.classifications = classifications
        self.missing_df = missing_df

        if make_plots:
            self._render_plots(stem, out_dir, picked_df, peaks_shifted,
                               color_map, sequence,
                               xlim, ylim, zoom_xlim, zoom_ylim)

        print("== Writing CSV ==")
        save_results_csv(ref_df, classifications, out_dir / f"{stem}.csv",
                         fit_results=fit_results, missing_df=missing_df)

        return self

    def _render_plots(self, stem, out_dir, picked_df, peaks_shifted,
                      color_map, sequence, xlim, ylim, zoom_xlim, zoom_ylim):
        """Render the diagnostic PDFs. Imports plotting lazily so the data
        pipeline does not depend on the plotting modules."""
        import matplotlib.pyplot as plt
        from .plotting import plot_r2eff_grid, plot_waterfall
        from ..spectra.plotting import plot_spectrum, plot_peaklist

        ref_spectrum = self.ref_spectrum
        ref_df = self.ref_df
        baseline = self.config["baseline_ref_plane"]

        fig, ax = plot_spectrum(ref_spectrum, baseline=baseline, xlim=xlim, ylim=ylim, cmap="Greys_r")
        ax.set_title(f"{stem} \u2014 reference HSQC")
        fig.savefig(out_dir / f"{stem}_hsqc_unlabeled.pdf", bbox_inches="tight")
        plt.close(fig)

        if peaks_shifted is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, panel_xlim, panel_ylim in zip(axes, (xlim, zoom_xlim), (ylim, zoom_ylim)):
                plot_spectrum(ref_spectrum, baseline=baseline,
                              xlim=panel_xlim, ylim=panel_ylim, cmap="Greys_r", ax=ax)
                plot_peaklist(ax, picked_df, text=None, color="tab:blue", label="picked peaks")
                plot_peaklist(ax, peaks_shifted, text="assn_label", color="tab:green",
                              label="ref peaklist, translated")
                plot_peaklist(ax, ref_df, text="assn_label", color="magenta", label="mapped peaks")
            fig.suptitle(f"{stem} \u2014 peak mapping")
            plt.tight_layout()
            fig.savefig(out_dir / f"{stem}_peak_mapping.pdf", bbox_inches="tight")
            plt.close(fig)

        merged = ref_df.merge(self.classifications[["ref_index", "label"]], on="ref_index", how="left")
        merged["label"] = merged["label"].fillna("unlabeled")
        text_col = ("assn_label"
                    if "assn_label" in merged.columns and merged["assn_label"].notna().any()
                    else "ref_index")

        fig, ax = plot_spectrum(ref_spectrum, baseline=baseline, contour_color="black",
                                xlim=xlim, ylim=ylim, figsize=(9, 8))
        plot_peaklist(ax, merged, marker=".", markersize=4, text=text_col,
                      label_fontsize=6, hue="label", palette=color_map)
        ax.set_title(f"{stem} \u2014 CPMG classification", fontsize=14)
        fig.savefig(out_dir / f"{stem}_hsqc_annotated_assigned.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, _ = plot_r2eff_grid(self.all_r2eff, self.classifications, ref_df=ref_df, color_map=color_map)
        fig.savefig(out_dir / f"{stem}_grid.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plot_waterfall(self.all_r2eff, self.classifications, ref_df, color_map=color_map,
                                 sequence=sequence, missing_df=self.missing_df)
        fig.savefig(out_dir / f"{stem}_waterfall.pdf", bbox_inches="tight")
        plt.close(fig)


def run_protein(yaml_path, out_dir, **kwargs):
    """Thin backward-compatible wrapper around
    ``CPMGExperiment.from_config(yaml_path).run(out_dir, ...)``. Returns the
    results dict (ref_df, fit_results, all_r2eff, classifications)."""
    exp = CPMGExperiment.from_config(yaml_path).run(out_dir, **kwargs)
    return dict(ref_df=exp.ref_df, fit_results=exp.fit_results,
                all_r2eff=exp.all_r2eff, classifications=exp.classifications)