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

OUTPUT_FILE = Path(__file__).parent / "validate_s2.png"
GRID_FILE = Path(__file__).parent / "validate_s2_grid.png"
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

    t = prof.table.dropna(subset=["S2"])
    seq_id, fit_s2, dep_s2, models = [], [], [], []
    for _, row in t.iterrows():
        sid = int(row["Seq_ID"])
        s2_dep = dep_by_seqid.get(sid)
        if s2_dep is None:
            continue
        seq_id.append(sid)
        fit_s2.append(row["S2"])
        dep_s2.append(s2_dep)
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
                fit_s2=np.array(fit_s2), dep_s2=np.array(dep_s2), models=models)


def main():
    results = []
    for bid in ENTRIES:
        try:
            res = compare_entry(bid)
        except Exception as e:
            print(f"BMRB {bid}: FAILED ({type(e).__name__}: {e})")
            continue
        n = len(res["fit_s2"])
        r, _ = pearsonr(res["fit_s2"], res["dep_s2"]) if n >= 2 else (float("nan"), None)
        rmse = float(np.sqrt(np.mean((res["fit_s2"] - res["dep_s2"]) ** 2))) if n else float("nan")
        res["r"], res["n"], res["rmse"] = r, n, rmse
        results.append(res)
        print(f"BMRB {bid}: N={n} Pearson r={r:.3f} RMSE={rmse:.3f}")

    all_fit = np.concatenate([r["fit_s2"] for r in results])
    all_dep = np.concatenate([r["dep_s2"] for r in results])
    overall_r, _ = pearsonr(all_fit, all_dep)
    overall_rmse = float(np.sqrt(np.mean((all_fit - all_dep) ** 2)))
    print(f"\nOverall: N={len(all_fit)} entries={len(results)} "
          f"Pearson r={overall_r:.3f} RMSE={overall_rmse:.3f}")

    with open(RESULTS_CSV, "w") as fh:
        fh.write("bmrb_id,title,field_mhz,n,pearson_r,rmse\n")
        for res in results:
            title = (res["title"] or "").replace(",", ";")
            fh.write(f"{res['bmrb_id']},{title},{res['field_mhz']},{res['n']},"
                     f"{res['r']:.4f},{res['rmse']:.4f}\n")
    print(f"saved {RESULTS_CSV}")

    with open(PAIRS_CSV, "w") as fh:
        fh.write("bmrb_id,seq_id,fit_s2,dep_s2,mf_model\n")
        for res in results:
            for sid, f, d, m in zip(res["seq_id"], res["fit_s2"], res["dep_s2"], res["models"]):
                fh.write(f"{res['bmrb_id']},{sid},{f:.4f},{d:.4f},{m}\n")
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

    ordered = sorted(results, key=lambda r: r["r"])
    ax_bar.barh([str(r["bmrb_id"]) for r in ordered], [r["r"] for r in ordered],
               color="steelblue", height=0.7)
    ax_bar.axvline(0, color="0.3", lw=0.8)
    ax_bar.set_xlabel("Pearson r (fit S2 vs. deposited S2)")
    ax_bar.set_title("Per-entry correlation")
    ax_bar.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150)
    print(f"saved {OUTPUT_FILE}")

    # one small scatter panel per protein, sorted best-to-worst by r
    ordered = sorted(results, key=lambda r: -r["r"])
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
        ax.set_title(f'{res["bmrb_id"]}  r={res["r"]:.2f}\n{title}', fontsize=6.5)
    for ax in axes2.flat[len(ordered):]:
        ax.axis("off")
    fig2.suptitle("Fit S2 (y) vs. deposited S2 (x), per protein -- sorted by r",
                  fontsize=11)
    fig2.tight_layout(rect=[0, 0, 1, 0.98])
    fig2.savefig(GRID_FILE, dpi=150)
    print(f"saved {GRID_FILE}")


if __name__ == "__main__":
    main()
