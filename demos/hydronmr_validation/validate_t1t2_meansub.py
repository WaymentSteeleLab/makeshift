"""
Figure of merit: makeshift's HYDRONMR port (`makeshift.hydronmr.run`) vs. the
original Fast-HYDRONMR Fortran binary, run on 7 ground-truth proteins bundled
in `GROUND_TRUTH_DONT_OVERWRITE/` (AQADK, BLAC, BLVRB, CHI19, CYPA, KRAS, VHR).
YJBJ (23001 atoms) is excluded -- the current dense O(N^2) mobility-matrix
approach OOMs on it (see makeshift/hydronmr/README.md "Known limitations").

For each protein:
  - `GROUND_TRUTH_DONT_OVERWRITE/<protein>/in.pdb` is fed to
    `makeshift.hydronmr.run()`, using the *same* AER/temperature/viscosity/
    field-strength parameters the original Fortran run used (parsed from that
    protein's own `hydronmr.dat` -- BLAC in particular used AER=2.0 A and
    B0=14.1 T instead of the usual 3.0 A / 11.74 T, so this must be read
    per-protein rather than assumed from the bundled default config.yml).
  - `tmp.t12` is the original Fortran output: 3 header lines (molecule name,
    AER, field), then one row per residue (`resseq, T1/T2, <unused column>`),
    with blank rows for residues the original run skipped (N-terminus, Pro,
    missing density).

Per-residue T1/T2 depends on the absolute size/shape of the hydrodynamic bead
model, which the AER-sphere approximation used here reproduces only
approximately (see "Known limitations" in makeshift/hydronmr/README.md) --
hence comparing *shapes*: both series are mean-subtracted per protein before
plotting/correlating, which cancels the systematic scale offset and isolates
whether the per-residue *pattern* (which residues are more/less mobile
relative to the whole molecule) is reproduced.

Produces validate_t1t2_meansub.png: one row per protein, "T1/T2 - mean(T1/T2)"
vs. residue index for both the original and makeshift series, annotated with
N (matched residues) and Pearson r.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import pearsonr

from makeshift import hydronmr

DATA_DIR = Path(__file__).parent / "GROUND_TRUTH_DONT_OVERWRITE"
OUTPUT_FILE = Path(__file__).parent / "validate_t1t2_meansub.png"

# YJBJ excluded: OOMs (23001 atoms, dense O(N^2) mobility matrix).
PROTEINS = ["AQADK", "BLAC", "BLVRB", "CHI19", "CYPA", "KRAS", "VHR"]


def parse_hydronmr_dat(path):
    """Parse a Fast-HYDRONMR `hydronmr.dat` run-parameter file into a dict
    matching `makeshift.hydronmr`'s `config.yml` schema. Values are
    positional (comment after `!` is stripped); see any `hydronmr.dat` for
    the field order this assumes."""
    tokens = []
    for line in Path(path).read_text().splitlines():
        token = line.split("!", 1)[0].strip().rstrip(",")
        if token and token != "*":
            tokens.append(token)

    n_fields = int(float(tokens[11]))
    return dict(
        aer_angstrom=float(tokens[3]),
        nsig=int(float(tokens[4])),
        temperature_k=float(tokens[5]),
        viscosity_poise=float(tokens[6]),
        iflag_dipoles=int(float(tokens[7])),
        gamma_x_e7=float(tokens[8]),
        r_nh_angstrom=float(tokens[9]),
        csa_ppm=float(tokens[10]),
        fields_tesla=[float(tokens[12 + i]) for i in range(n_fields)],
    )


def parse_tmp_t12(path):
    """Original Fortran per-residue T1/T2, keyed by resseq. Skips the 3
    header lines and any row with a blank value (residues the original run
    excluded: N-terminus, Pro, missing density)."""
    values = {}
    for line in Path(path).read_text().splitlines()[3:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        values[int(parts[0])] = float(parts[1])
    return values


def run_makeshift(pdb_path, dat_path):
    """Run makeshift.hydronmr.run() with this protein's own ground-truth
    run parameters (not the bundled default config.yml -- see module
    docstring re: BLAC's non-default AER/field)."""
    cfg = parse_hydronmr_dat(dat_path)
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        config_path = f.name
    result = hydronmr.run(pdb_path, config_path=config_path)
    df = result.to_dataframe()
    return dict(zip(df["seqpos"], df["T1_over_T2"]))


def main():
    panels = []
    for protein in PROTEINS:
        protein_dir = DATA_DIR / protein
        original = parse_tmp_t12(protein_dir / "tmp.t12")
        predicted = run_makeshift(protein_dir / "in.pdb", protein_dir / "hydronmr.dat")

        common = sorted(set(original) & set(predicted))
        orig_vals = np.array([original[i] for i in common])
        pred_vals = np.array([predicted[i] for i in common])
        r, _ = pearsonr(orig_vals, pred_vals)

        panels.append(dict(
            protein=protein, resseq=common,
            orig_ms=orig_vals - orig_vals.mean(),
            pred_ms=pred_vals - pred_vals.mean(),
            r=r, n=len(common),
        ))
        print(f"{protein}: N={len(common)} r={r:.4f}")

    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.6 * len(panels)))
    for ax, p in zip(axes, panels):
        ax.plot(p["resseq"], p["orig_ms"], "o-", ms=3, lw=1, color="steelblue",
                 label="Original (tmp.t12) - mean")
        ax.plot(p["resseq"], p["pred_ms"], "s-", ms=3, lw=1, color="darkorange",
                 label="makeshift.hydronmr - mean")
        ax.axhline(0, color="0.7", lw=0.8, zorder=0)
        ax.set_title(f'{p["protein"]}  (N={p["n"]}, r={p["r"]:.4f})')
        ax.set_xlabel("Residue index")
        ax.set_ylabel("T1/T2 - mean(T1/T2)")
        ax.legend(frameon=False, fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150)
    print(f"saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
