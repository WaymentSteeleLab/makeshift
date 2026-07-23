"""
Figure of merit: makeshift's RCI implementation vs. the reference
implementation (Berjanskii & Wishart 2005, rci_v_1c.py), run on the *exact*
input/output pair bundled with the reference script itself -- BMRB 4403,
the J domain of murine polyomavirus T antigen ("PyJCScorr"), from the
RCI/ folder distributed alongside rci_v_1c.py.

makeshift's RCI is computed through the public API:

    cs = ms.ChemicalShifts(shifts_df)   # parsed from PyJCScorr directly
    rci = RCI.calc(cs, sequence=SEQUENCE)

`ChemicalShifts.from_bmrb`/`NMRStarEntry` can't parse PyJCScorr -- it's a
pre-3.0 NMR-STAR file -- so the chemical-shift loop is parsed directly
from tests/data/rci/PyJCScorr (a byte-identical copy of the file shipped
in the RCI/ folder) and handed to ChemicalShifts manually; RCI.calc() is
otherwise exercised exactly as a user would call it.

Because this feeds the port the identical input rci_v_1c.py itself
consumed (rather than a fresh, differently-referenced BMRB deposit), the
match is bit-exact modulo floating point noise -- this is the same pair
tests/test_rci_regression.py pins for CI.

Produces rci_validation.png: a per-residue trace and a reference-vs-
makeshift scatter, annotated with Pearson r and RMSD.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import makeshift as ms
from makeshift.rci import RCI

DATA_DIR = Path(__file__).parent.parent / "tests" / "data" / "rci"
INPUT_FILE = DATA_DIR / "PyJCScorr"
REFERENCE_FILE = DATA_DIR / "PyJCScorr_RCI.txt_compare"
OUTPUT_FILE = Path(__file__).parent / "rci_validation.png"

# From PyJCScorr's own _Mol_residue_sequence / _Residue_seq_code loop
# (BMRB 4403, first residue = 1).
SEQUENCE = (
    "MDRVLSRADKERLLELLKLPRQLWGDFGRMQQAYKQQSLLLHPDKGGSHALMQELNSLWGTFKTEVYNLRMNLGGTGFQHHHHHH"
)


def _parse_shift_loop(path):
    """Chemical-shift loop of an old-style (pre-3.0) NMR-STAR file, in the
    tidy Seq_ID/Comp_ID/Atom_ID/Val form ChemicalShifts expects."""
    rows = []
    in_loop = False
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "_Chem_shift_ambiguity_code":
            in_loop = True
            continue
        if in_loop and parts[0] == "stop_":
            break
        if in_loop and len(parts) >= 7:
            try:
                seq_id, comp_id, atom_id, val = int(parts[1]), parts[2], parts[3], float(parts[5])
            except ValueError:
                continue
            rows.append((seq_id, comp_id, atom_id, val))
    return pd.DataFrame(rows, columns=["Seq_ID", "Comp_ID", "Atom_ID", "Val"])


def main():
    shifts_df = _parse_shift_loop(INPUT_FILE)
    cs = ms.ChemicalShifts(shifts_df)
    rci = RCI.calc(cs, sequence=SEQUENCE)

    reference = pd.read_csv(
        REFERENCE_FILE, sep=" ", header=None, names=["Seq_ID", "RCI", "Comp_ID"],
    )

    merged = rci.results[["Seq_ID", "Comp_ID", "RCI"]].merge(
        reference[["Seq_ID", "RCI"]], on="Seq_ID", suffixes=("_makeshift", "_reference"),
    ).sort_values("Seq_ID")

    r, _ = pearsonr(merged["RCI_reference"], merged["RCI_makeshift"])
    rmsd = np.sqrt(np.mean((merged["RCI_makeshift"] - merged["RCI_reference"]) ** 2))
    max_abs_diff = (merged["RCI_makeshift"] - merged["RCI_reference"]).abs().max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(merged["Seq_ID"], merged["RCI_reference"], color="0.4", lw=2.5,
              label="reference (rci_v_1c.py)")
    ax1.plot(merged["Seq_ID"], merged["RCI_makeshift"], color="steelblue", lw=1.2,
              ls="--", label="makeshift")
    ax1.set_xlabel("residue")
    ax1.set_ylabel("RCI")
    ax1.set_title("BMRB 4403 (PyJCScorr)")
    ax1.legend(frameon=False, fontsize=8)

    lims = [0, max(merged["RCI_reference"].max(), merged["RCI_makeshift"].max()) * 1.05]
    ax2.plot(lims, lims, color="0.7", lw=1, zorder=0)
    ax2.scatter(merged["RCI_reference"], merged["RCI_makeshift"], s=18,
                 color="steelblue", edgecolor="white", linewidth=0.4, zorder=1)
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel("reference RCI")
    ax2.set_ylabel("makeshift RCI")
    ax2.set_aspect("equal")
    ax2.text(0.05, 0.92, f"r = {r:.6f}\nRMSD = {rmsd:.2e}\nmax |Δ| = {max_abs_diff:.2e}",
              transform=ax2.transAxes, va="top", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=300)
    print(f"n_residues={len(merged)} pearson_r={r:.10f} rmsd={rmsd:.3e} max_abs_diff={max_abs_diff:.3e}")
    print(f"saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
