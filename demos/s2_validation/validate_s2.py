"""
makeshift's home-grown model-free S2 fit (`RelaxationProfile.fit_order_parameters`,
Models 1-3, anisotropic tumbling from HYDRONMR) vs. deposited BMRB model-free S2
values, for three entries chosen because they have raw R1/R2/NOE, deposited
order parameters, and a cited PDB structure, all at a single field strength
(the regime `fit_order_parameters` targets -- see makeshift/relaxation/model_free.py):

    4390  eotaxin            74 residues   PDB 1EOT   500 MHz
    6577  (135-residue)     135 residues   PDB 1Z9B   600.13 MHz
    5991  (140-residue)     140 residues   PDB 2STW   500 MHz

(chosen by screening BMRB's `_Order_param.Order_param_val` tag search for
entries that also carry T1/T2/NOE loops and a cited PDB id, then checking
each candidate's relaxation lists were a single field -- see conversation/
scratch notes; not hand-picked for agreement.)

This is a sanity check, not a reproduction target: `fit_order_parameters`
deliberately uses a different, simpler model-selection rule than whatever
software produced each entry's deposited fit (often modelfree4/FastModelFree,
sometimes an older in-house tool), so per-residue model *assignments* are not
expected to match -- only the S2 *magnitudes* should track reasonably well
for residues both approaches call ordered.

Produces validate_s2.png: one scatter panel per entry (fit S2 vs. deposited
S2, y=x reference line), annotated with N and Pearson r.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from makeshift.entry import NMRStarEntry
from makeshift.relaxation import RelaxationProfile

OUTPUT_FILE = Path(__file__).parent / "validate_s2.png"

ENTRIES = [4390, 6577, 5991]


def compare_entry(bmrb_id):
    entry = NMRStarEntry.from_bmrb(bmrb_id)
    prof = RelaxationProfile.from_entry(entry)
    prof.fit_order_parameters()

    deposited = entry.order_parameters()
    deposited = deposited.dropna(subset=["S2"])
    dep_by_seqid = dict(zip(deposited["Seq_ID"].astype(int), deposited["S2"]))

    t = prof.table.dropna(subset=["S2"])
    fit_s2 = []
    dep_s2 = []
    models = []
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
    panels = []
    for bid in ENTRIES:
        res = compare_entry(bid)
        n = len(res["fit_s2"])
        if n >= 2:
            r, _ = pearsonr(res["fit_s2"], res["dep_s2"])
        else:
            r = float("nan")
        rmse = float(np.sqrt(np.mean((res["fit_s2"] - res["dep_s2"]) ** 2))) if n else float("nan")
        res["r"] = r
        res["n"] = n
        res["rmse"] = rmse
        panels.append(res)
        print(f"BMRB {bid}: N={n} Pearson r={r:.3f} RMSE={rmse:.3f}")

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    for ax, p in zip(axes, panels):
        colors = {"1": "black", "2": "tab:blue", "3": "tab:orange",
                  "ambiguous": "tab:red"}
        for model in sorted(set(p["models"])):
            mask = [m == model for m in p["models"]]
            ax.scatter(np.array(p["dep_s2"])[mask], np.array(p["fit_s2"])[mask],
                       s=14, alpha=0.75, color=colors.get(model, "grey"),
                       label=f"Model {model}")
        ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", zorder=0)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("deposited S2 (BMRB)")
        ax.set_ylabel("makeshift fit S2")
        ax.set_title(f'BMRB {p["bmrb_id"]}  (N={p["n"]}, r={p["r"]:.3f})')
        ax.legend(frameon=False, fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150)
    print(f"saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
