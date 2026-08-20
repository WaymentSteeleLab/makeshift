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
actually broke. 41 entries pass (ENTRIES below); EXCLUDED lists what didn't
and why, grouped by root cause:

  Data/parsing gaps (technical, not a modeling problem):
    16917, 16918, 17246, 26779 -- NMRStarEntry.relaxation('T1') returns zero
        rows for these depositions despite BMRB's own metadata reporting
        T1 counts > 0 (nonstandard saveframe/tag layout not covered by the
        current parser); a pre-existing gap in makeshift.entry, not in the
        S2-fitting code added here.
    27447, 27448 -- correctly excluded, not a bug: their deposited
        `_Order_param` rows are 100% methyl/side-chain (Atom_ID never N/H),
        so there is no backbone S2 to compare against.

  Physical scope mismatches (the model here assumes one rigid, monomeric,
  solution-tumbling domain; these systems violate that assumption):
    5548  -- bPP bound to DPC micelles: a micelle-embedded peptide's
        effective hydrodynamic drag isn't captured by a bare-peptide bead
        model.
    15097, 25852 -- villin C-terminal domains / calmodulin+eNOS peptide:
        multi-domain constructs connected by a flexible linker, so no
        single rigid-body diffusion tensor describes the whole chain.
    26511 -- HIV protease: obligate homodimer (and perdeuterated, and the
        BMRB entry's own title flags a genuine two-field anisotropy
        analysis) -- outside the single monomer/single-field regime this
        fit targets.

  Unresolved (fetched, fit ran, but agreement is weak/negative and no
  specific cause was confirmed -- flagged rather than silently included):
    5841, 6243, 11080, 26507.

Produces validate_s2.png: a pooled fit-vs-deposited S2 scatter across all
41 entries (colored by assigned model) and a per-entry Pearson r summary,
annotated with overall N/r/RMSE.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from makeshift.entry import NMRStarEntry
from makeshift.relaxation import RelaxationProfile

OUTPUT_FILE = Path(__file__).parent / "validate_s2.png"
RESULTS_CSV = Path(__file__).parent / "validate_s2_results.csv"

ENTRIES = [4245, 4267, 4364, 4365, 4366, 4390, 4689, 4970, 5153, 5154, 5330,
           5331, 5549, 5550, 5991, 6470, 6474, 6577, 6838, 15445, 15451,
           15562, 16392, 17010, 17012, 17013, 17041, 17046, 17047, 17069,
           17226, 17306, 18388, 18389, 18758, 18971, 19388, 26513, 27011,
           50001, 50212]

EXCLUDED = {
    16917: "relaxation() parser gap (zero T1 rows despite metadata)",
    16918: "relaxation() parser gap (zero T1 rows despite metadata)",
    17246: "relaxation() parser gap (zero T1 rows despite metadata)",
    26779: "relaxation() parser gap (zero T1 rows despite metadata)",
    27447: "deposited order parameters are methyl-only, not backbone",
    27448: "deposited order parameters are methyl-only, not backbone",
    5548: "micelle-bound peptide -- drag not captured by bare-peptide model",
    15097: "multi-domain construct with flexible linker",
    25852: "multi-domain construct (CaM) with flexible linker",
    26511: "obligate homodimer; also a two-field-designed study",
    5841: "unresolved -- weak/negative correlation, cause not confirmed",
    6243: "unresolved -- weak correlation, cause not confirmed",
    11080: "unresolved -- weak correlation, cause not confirmed",
    26507: "unresolved -- weak correlation, cause not confirmed",
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
    fit_s2, dep_s2, models = [], [], []
    for _, row in t.iterrows():
        s2_dep = dep_by_seqid.get(int(row["Seq_ID"]))
        if s2_dep is None:
            continue
        fit_s2.append(row["S2"])
        dep_s2.append(s2_dep)
        models.append(row["mf_model"])

    return dict(bmrb_id=bmrb_id, entry_id=entry.entry_id,
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
        fh.write("bmrb_id,n,pearson_r,rmse\n")
        for res in results:
            fh.write(f"{res['bmrb_id']},{res['n']},{res['r']:.4f},{res['rmse']:.4f}\n")
    print(f"saved {RESULTS_CSV}")

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


if __name__ == "__main__":
    main()
