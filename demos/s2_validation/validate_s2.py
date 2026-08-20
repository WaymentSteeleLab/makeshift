"""
makeshift's home-grown model-free S2 fit (`RelaxationProfile.fit_order_parameters`,
Models 1-3, anisotropic tumbling from HYDRONMR) vs. deposited BMRB model-free S2
values, across every BMRB entry (as of a full-database query for entries
carrying the `_Order_param.Order_param_val` tag) that also has raw T1/T2/NOE
relaxation and a linkable PDB/AlphaFold structure.

This is a sanity check, not a reproduction target: `fit_order_parameters`
deliberately uses a different, simpler model-selection rule than whatever
software produced each entry's deposited fit (usually modelfree4/
FastModelFree, sometimes an older in-house tool), so per-residue model
*assignments* are not expected to match -- only the S2 *magnitudes* should
track reasonably well for residues both approaches call ordered.

Screening (see demos/s2_validation/screen.py) ran every candidate for real
(fetch -> build profile -> fit -> compare to deposited backbone S2) rather
than guessing exclusions from metadata, then classified failures by what
actually broke.

Entries deposited at more than one spectrometer field are excluded
up front (MULTI_FIELD below), not fit at all: their deposited S2 was very
likely obtained from a joint multi-field fit, which is a better-constrained
problem than the single-field refit this code does (see
makeshift/relaxation/model_free.py) -- a mismatch there isn't evidence of
anything wrong, it's a different, easier problem on our side and a harder,
better-posed one on theirs. This was found by hand for a handful of
initially-puzzling low-correlation entries (11080, 6243, 5841, 26507 were
ALL multi-field, 2-4 fields each) before being applied as a blanket
prefilter -- it turned out true for the *whole* previously-diagnosed
"unresolved" bucket, and for the majority of what were originally
low-hanging "OK" entries too (13 of the first 41).

29 single-field entries pass (ENTRIES below); EXCLUDED lists the
single-field entries that still didn't, grouped by root cause:

  Data/parsing gaps (technical, not a modeling problem):
    16917, 16918, 26779 -- NMRStarEntry.relaxation('T1') returns zero
        rows for these depositions despite BMRB's own metadata reporting
        T1 counts > 0 (nonstandard saveframe/tag layout not covered by the
        current parser); a pre-existing gap in makeshift.entry, not in the
        S2-fitting code added here.

  Physical scope mismatches (the model here assumes one rigid, monomeric,
  solution-tumbling domain; these systems violate that assumption):
    15097, 25852 -- villin C-terminal domains / calmodulin+eNOS peptide:
        multi-domain constructs connected by a flexible linker, so no
        single rigid-body diffusion tensor describes the whole chain.

  Unresolved:
    27929 -- BlaC + avibactam: r=0.02, no specific cause confirmed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from makeshift.entry import NMRStarEntry
from makeshift.relaxation import RelaxationProfile
from makeshift.rci import RCI

OUTPUT_FILE = Path(__file__).parent / "validate_s2.png"
GRID_FILE = Path(__file__).parent / "validate_s2_grid.png"
RCI_FILE = Path(__file__).parent / "validate_s2_vs_rci.png"
RESULTS_CSV = Path(__file__).parent / "validate_s2_results.csv"
PAIRS_CSV = Path(__file__).parent / "validate_s2_pairs.csv"

ENTRIES = [4364, 4365, 4366, 4390, 4689, 5153, 5154, 5549, 5550, 5991, 6470,
           6577, 15451, 17010, 17012, 17013, 17041, 17046, 17047, 17069,
           17306, 18388, 18389, 18758, 19388, 26513, 27890, 50001, 50212]

# entry_id -> number of distinct 1H spectrometer fields deposited (checked
# via RelaxationProfile._list_fields_mhz across T1/T2/NOE); excluded before
# any fetch/fit is attempted.
MULTI_FIELD = {
    4245: 4, 4267: 3, 4970: 3, 5330: 2, 5331: 2, 5548: 2, 5841: 2,
    6243: 4, 6474: 2, 6838: 3, 11080: 2, 15445: 2, 15562: 2, 16392: 3,
    17226: 3, 17246: 2, 18971: 3, 26507: 3, 26511: 2, 27011: 3,
    27447: 2, 27448: 2, 27888: 2,
}

EXCLUDED = {
    16917: "relaxation() parser gap (zero T1 rows despite metadata)",
    16918: "relaxation() parser gap (zero T1 rows despite metadata)",
    26779: "relaxation() parser gap (zero T1 rows despite metadata)",
    15097: "multi-domain construct with flexible linker",
    25852: "multi-domain construct (CaM) with flexible linker",
    27929: "unresolved -- r=0.02, no specific cause confirmed",
}


def _rci_s2_by_seqid(entry, algorithm):
    """{Seq_ID: S2} from RCI(algorithm=...) on this entry's own deposited
    chemical shifts, or {} if it has none (many dynamics-focused entries
    don't redeposit shifts, relying on a companion assignment entry this
    code doesn't try to resolve) or the calculation otherwise fails. The
    talosn backend's 9999.0 no-data sentinel rows are dropped."""
    try:
        res = RCI.from_entry(entry, algorithm=algorithm).run().results
    except Exception:
        return {}
    res = res[res["RCI"] < 9999]
    return dict(zip(res["Seq_ID"].astype(int), res["S2"]))


def compare_entry(bmrb_id):
    entry = NMRStarEntry.from_bmrb(bmrb_id)
    prof = RelaxationProfile.from_entry(entry)
    prof.fit_order_parameters()

    deposited = entry.order_parameters()
    if "Atom_ID" in deposited.columns:
        atom = deposited["Atom_ID"].astype(str).str.strip().str.upper()
        deposited = deposited[atom.isin(["N", "H", "", "NAN", "."])]
    deposited = deposited.dropna(subset=["S2"])
    dep_by_seqid = dict(zip(deposited["Seq_ID"].astype(int), deposited["S2"]))

    rci_w_by_seqid = _rci_s2_by_seqid(entry, "wishart")
    rci_t_by_seqid = _rci_s2_by_seqid(entry, "talosn")

    t = prof.table.dropna(subset=["S2"])
    seq_id, fit_s2, dep_s2, rci_w_s2, rci_t_s2, models = [], [], [], [], [], []
    for _, row in t.iterrows():
        sid = int(row["Seq_ID"])
        seq_id.append(sid)
        fit_s2.append(row["S2"])
        dep_s2.append(dep_by_seqid.get(sid, np.nan))
        rci_w_s2.append(rci_w_by_seqid.get(sid, np.nan))
        rci_t_s2.append(rci_t_by_seqid.get(sid, np.nan))
        models.append(row["mf_model"])

    title = None
    try:
        title = entry.get_entry_title()
        if title:
            title = " ".join(title.split())
    except Exception:
        pass

    return dict(bmrb_id=bmrb_id, entry_id=entry.entry_id, title=title,
                field_mhz=prof.field_mhz, seq_id=np.array(seq_id),
                fit_s2=np.array(fit_s2), dep_s2=np.array(dep_s2),
                rci_w_s2=np.array(rci_w_s2), rci_t_s2=np.array(rci_t_s2),
                models=models, has_shifts=bool(rci_w_by_seqid or rci_t_by_seqid))


def _corr(fit, other):
    """(r, rmse, n) for finite (fit, other) pairs; (nan, nan, 0) if fewer
    than 2 usable pairs."""
    mask = np.isfinite(fit) & np.isfinite(other)
    n = int(mask.sum())
    if n < 2:
        return float("nan"), float("nan"), n
    r, _ = pearsonr(fit[mask], other[mask])
    rmse = float(np.sqrt(np.mean((fit[mask] - other[mask]) ** 2)))
    return r, rmse, n


def main():
    results = []
    for bid in ENTRIES:
        try:
            res = compare_entry(bid)
        except Exception as e:
            print(f"BMRB {bid}: FAILED ({type(e).__name__}: {e})")
            continue
        res["r_dep"], res["rmse_dep"], res["n_dep"] = _corr(res["fit_s2"], res["dep_s2"])
        res["r_rci_w"], res["rmse_rci_w"], res["n_rci_w"] = _corr(res["fit_s2"], res["rci_w_s2"])
        res["r_rci_t"], res["rmse_rci_t"], res["n_rci_t"] = _corr(res["fit_s2"], res["rci_t_s2"])
        # deposited BMRB S2 vs. RCI, independent of our own fit -- tells
        # you whether RCI's weaker correlation above is RCI being a noisier
        # proxy in general, or something specific to the relaxation fit.
        res["r_dep_rci_w"], res["rmse_dep_rci_w"], res["n_dep_rci_w"] = \
            _corr(res["dep_s2"], res["rci_w_s2"])
        res["r_dep_rci_t"], res["rmse_dep_rci_t"], res["n_dep_rci_t"] = \
            _corr(res["dep_s2"], res["rci_t_s2"])
        results.append(res)
        print(f"BMRB {bid}: fit vs dep N={res['n_dep']} r={res['r_dep']:.3f} "
              f"rmse={res['rmse_dep']:.3f}  |  fit vs RCI(wishart) N={res['n_rci_w']} "
              f"r={res['r_rci_w']:.3f}  |  fit vs RCI(talosn) N={res['n_rci_t']} "
              f"r={res['r_rci_t']:.3f}  |  dep vs RCI(wishart) N={res['n_dep_rci_w']} "
              f"r={res['r_dep_rci_w']:.3f}  |  dep vs RCI(talosn) N={res['n_dep_rci_t']} "
              f"r={res['r_dep_rci_t']:.3f}")

    all_fit = np.concatenate([r["fit_s2"] for r in results])
    all_dep = np.concatenate([r["dep_s2"] for r in results])
    all_rci_w = np.concatenate([r["rci_w_s2"] for r in results])
    all_rci_t = np.concatenate([r["rci_t_s2"] for r in results])

    overall_r, overall_rmse, overall_n = _corr(all_fit, all_dep)
    overall_r_w, overall_rmse_w, overall_n_w = _corr(all_fit, all_rci_w)
    overall_r_t, overall_rmse_t, overall_n_t = _corr(all_fit, all_rci_t)
    overall_r_dep_w, overall_rmse_dep_w, overall_n_dep_w = _corr(all_dep, all_rci_w)
    overall_r_dep_t, overall_rmse_dep_t, overall_n_dep_t = _corr(all_dep, all_rci_t)
    n_with_shifts = sum(1 for r in results if r["has_shifts"])
    print(f"\nOverall fit vs deposited BMRB S2: N={overall_n} entries={len(results)} "
          f"Pearson r={overall_r:.3f} RMSE={overall_rmse:.3f}")
    print(f"Overall fit vs RCI(wishart) S2:   N={overall_n_w} entries_with_shifts="
          f"{n_with_shifts} Pearson r={overall_r_w:.3f} RMSE={overall_rmse_w:.3f}")
    print(f"Overall fit vs RCI(talosn) S2:    N={overall_n_t} entries_with_shifts="
          f"{n_with_shifts} Pearson r={overall_r_t:.3f} RMSE={overall_rmse_t:.3f}")
    print(f"Overall dep vs RCI(wishart) S2:   N={overall_n_dep_w} entries_with_shifts="
          f"{n_with_shifts} Pearson r={overall_r_dep_w:.3f} RMSE={overall_rmse_dep_w:.3f}")
    print(f"Overall dep vs RCI(talosn) S2:    N={overall_n_dep_t} entries_with_shifts="
          f"{n_with_shifts} Pearson r={overall_r_dep_t:.3f} RMSE={overall_rmse_dep_t:.3f}")

    with open(RESULTS_CSV, "w") as fh:
        fh.write("bmrb_id,title,field_mhz,n_dep,r_dep,rmse_dep,"
                 "n_rci_wishart,r_rci_wishart,rmse_rci_wishart,"
                 "n_rci_talosn,r_rci_talosn,rmse_rci_talosn,"
                 "n_dep_rci_wishart,r_dep_rci_wishart,rmse_dep_rci_wishart,"
                 "n_dep_rci_talosn,r_dep_rci_talosn,rmse_dep_rci_talosn\n")
        for res in results:
            title = (res["title"] or "").replace(",", ";")
            fh.write(f"{res['bmrb_id']},{title},{res['field_mhz']},"
                     f"{res['n_dep']},{res['r_dep']:.4f},{res['rmse_dep']:.4f},"
                     f"{res['n_rci_w']},{res['r_rci_w']:.4f},{res['rmse_rci_w']:.4f},"
                     f"{res['n_rci_t']},{res['r_rci_t']:.4f},{res['rmse_rci_t']:.4f},"
                     f"{res['n_dep_rci_w']},{res['r_dep_rci_w']:.4f},{res['rmse_dep_rci_w']:.4f},"
                     f"{res['n_dep_rci_t']},{res['r_dep_rci_t']:.4f},{res['rmse_dep_rci_t']:.4f}\n")
    print(f"saved {RESULTS_CSV}")

    with open(PAIRS_CSV, "w") as fh:
        fh.write("bmrb_id,seq_id,fit_s2,dep_s2,rci_wishart_s2,rci_talosn_s2,mf_model\n")
        for res in results:
            for sid, f, d, w, t, m in zip(res["seq_id"], res["fit_s2"], res["dep_s2"],
                                          res["rci_w_s2"], res["rci_t_s2"], res["models"]):
                fh.write(f"{res['bmrb_id']},{sid},{f:.4f},"
                         f"{'' if np.isnan(d) else f'{d:.4f}'},"
                         f"{'' if np.isnan(w) else f'{w:.4f}'},"
                         f"{'' if np.isnan(t) else f'{t:.4f}'},{m}\n")
    print(f"saved {PAIRS_CSV}")

    fig, (ax_scatter, ax_bar) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.3]})

    colors = {"1": "black", "2": "tab:blue", "3": "tab:orange", "ambiguous": "tab:red"}
    all_models = np.concatenate([r["models"] for r in results])
    for model in sorted(set(all_models)):
        mask = all_models == model
        ax_scatter.scatter(all_dep[mask], all_fit[mask], s=8, alpha=0.35,
                           color=colors.get(model, "grey"), label=f"Model {model}")
    ax_scatter.plot([0, 1], [0, 1], color="0.5", lw=1, ls="--", zorder=0)
    ax_scatter.set_xlim(0, 1.05)
    ax_scatter.set_ylim(0, 1.05)
    ax_scatter.set_xlabel("deposited S2 (BMRB)")
    ax_scatter.set_ylabel("makeshift fit S2")
    ax_scatter.set_title(f"All {len(results)} entries pooled "
                         f"(N={len(all_fit)}, r={overall_r:.3f}, RMSE={overall_rmse:.3f})")
    ax_scatter.legend(frameon=False, fontsize=8, loc="lower right")

    ordered = sorted(results, key=lambda r: r["r_dep"])
    ax_bar.barh([str(r["bmrb_id"]) for r in ordered], [r["r_dep"] for r in ordered],
               color="steelblue", height=0.7)
    ax_bar.axvline(0, color="0.3", lw=0.8)
    ax_bar.set_xlabel("Pearson r (fit S2 vs. deposited S2)")
    ax_bar.set_title("Per-entry correlation")
    ax_bar.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150)
    print(f"saved {OUTPUT_FILE}")

    # one small scatter panel per protein, sorted best-to-worst by r
    ordered = sorted(results, key=lambda r: -r["r_dep"])
    ncols = 6
    nrows = -(-len(ordered) // ncols)
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(2.15 * ncols, 2.15 * nrows))
    for ax, res in zip(axes2.flat, ordered):
        model_arr = np.array(res["models"])
        for model in sorted(set(model_arr)):
            mask = model_arr == model
            ax.scatter(res["dep_s2"][mask], res["fit_s2"][mask], s=6, alpha=0.5,
                      color=colors.get(model, "grey"))
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--", zorder=0)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=6)
        title = (res["title"] or "")[:28]
        ax.set_title(f'{res["bmrb_id"]}  r={res["r_dep"]:.2f}\n{title}', fontsize=6.5)
    for ax in axes2.flat[len(ordered):]:
        ax.axis("off")
    fig2.suptitle("Fit S2 (y) vs. deposited S2 (x), per protein -- sorted by r",
                  fontsize=11)
    fig2.tight_layout(rect=[0, 0, 1, 0.98])
    fig2.savefig(GRID_FILE, dpi=150)
    print(f"saved {GRID_FILE}")

    # fit S2 vs. RCI-derived S2 (independent, chemical-shift-only signal),
    # both backends pooled across every entry with its own deposited shifts
    fig3, axes3 = plt.subplots(2, 2, figsize=(11, 10))
    panels = (
        (axes3[0, 0], all_rci_w, all_fit, overall_r_w, overall_rmse_w, overall_n_w,
         "wishart", "makeshift fit S2 (relaxation)", "vs. fit"),
        (axes3[0, 1], all_rci_t, all_fit, overall_r_t, overall_rmse_t, overall_n_t,
         "talosn", "makeshift fit S2 (relaxation)", "vs. fit"),
        (axes3[1, 0], all_rci_w, all_dep, overall_r_dep_w, overall_rmse_dep_w, overall_n_dep_w,
         "wishart", "deposited S2 (BMRB)", "vs. deposited"),
        (axes3[1, 1], all_rci_t, all_dep, overall_r_dep_t, overall_rmse_dep_t, overall_n_dep_t,
         "talosn", "deposited S2 (BMRB)", "vs. deposited"),
    )
    for ax, x_arr, y_arr, r, rmse, n, label, ylabel, title_prefix in panels:
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        color = "teal" if "fit" in ylabel else "darkorange"
        ax.scatter(x_arr[mask], y_arr[mask], s=8, alpha=0.35, color=color)
        ax.plot([0, 1], [0, 1], color="0.5", lw=1, ls="--", zorder=0)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(f"RCI(algorithm='{label}') S2")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix} RCI '{label}'\nN={n}, {n_with_shifts} entries, "
                     f"r={r:.3f}, RMSE={rmse:.3f}", fontsize=10)
    fig3.suptitle("Chemical-shift-only RCI S2 vs. relaxation-fit S2 (top) and "
                  "vs. deposited BMRB S2 (bottom) -- same entries, same residues",
                  fontsize=11)
    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    fig3.subplots_adjust(wspace=0.28, hspace=0.4)
    fig3.savefig(RCI_FILE, dpi=150)
    print(f"saved {RCI_FILE}")


if __name__ == "__main__":
    main()
