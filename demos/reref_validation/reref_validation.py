"""
Compare makeshift's LACS / PANAV re-referencing offsets to the reference
implementations, using the BMRB entries bundled in this folder.

Inputs:
  inputs/bmr<id>_3.str           — local NMR-STAR 3 deposits
  LACS/<id>_LACS.str             — BMRB-hosted LACS validation files
  PANAV/bmr<id>_3.str_output.txt — PANAV v2.1 (NMRbox) stdout dumps

Sign convention: makeshift matches the paper / reference dumps
(`offset ≈ d_ave − d_obs`, `corrected = Val + offset`), so the plot is
reference vs. makeshift directly.

Produces reref_validation.png with a LACS panel and a PANAV panel sharing
identical axis limits and tick spacing.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

import makeshift as ms

HERE = Path(__file__).parent
INPUT_DIR = HERE / "inputs"
LACS_DIR = HERE / "LACS"
PANAV_DIR = HERE / "PANAV"
OUTPUT_FILE = HERE / "reref_validation.png"

BMRB_IDS = [11080, 15451, 15490, 15521, 15581, 15763, 15959, 52018, 5991]

# LACS fits CA and CB independently (they can differ, e.g. deuterated samples).
LACS_ATOMS = ("CA", "CB", "C")
PANAV_ATOMS = ("N", "CA", "CB", "C")

ATOM_COLORS = {"N": "#4C72B0", "CA": "#55A868", "CB": "#C44E52", "C": "#8172B3"}
LACS_YNAME = {"CO": "C", "CA": "CA", "CB": "CB", "HA": "HA", "N": "N", "HN": "H", "H": "H"}
PANAV_OFFSET_RE = re.compile(r"(CO|CA|CB|N):\s*([-+]?\d*\.?\d+)\s*ppm")

# rows collected per method: (bmrb_id, atom, reference, makeshift)
rows = {"lacs": [], "panav": []}

print("Comparing makeshift reref offsets to LACS / PANAV references")
print("(Δ(ref−ms) near 0 ⇒ same-sign paper convention)\n")

for method, atoms in (("lacs", LACS_ATOMS), ("panav", PANAV_ATOMS)):
    for bmrb_id in BMRB_IDS:

        # ---- reference offsets -------------------------------------------
        reference = {}

        if method == "lacs":
            ref_path = LACS_DIR / f"{bmrb_id}_LACS.str"
            if not ref_path.exists():
                print(f"{bmrb_id} lacs: no reference file (skip)")
                continue
            yname = None
            for line in ref_path.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "_LACS_plot.Y_coord_name":
                    yname = parts[1]
                elif parts[0] == "_LACS_plot.Y_axis_chem_shift_offset" and yname is not None:
                    reference[LACS_YNAME.get(yname, yname)] = float(parts[1])
                    yname = None
        else:
            ref_path = PANAV_DIR / f"bmr{bmrb_id}_3.str_output.txt"
            lines = ref_path.read_text(encoding="utf-8", errors="replace").splitlines()
            done = False
            for i, line in enumerate(lines):
                if done or "Detected reference offsets" not in line:
                    continue
                for nxt in lines[i + 1:]:
                    if not nxt.strip():
                        continue
                    for match in PANAV_OFFSET_RE.finditer(nxt):
                        atom = "C" if match.group(1) == "CO" else match.group(1)
                        reference[atom] = float(match.group(2))
                    done = True
                    break

        # ---- makeshift offsets -------------------------------------------
        try:
            entry = ms.NMRStarEntry.from_file(INPUT_DIR / f"bmr{bmrb_id}_3.str")
            cs = ms.ChemicalShifts.from_entry(entry)
        except Exception as exc:
            print(f"{bmrb_id} {method}: makeshift failed ({exc})")
            continue

        n_obs = cs.data.Atom_ID.value_counts().to_dict()
        n_ha = sum(int(n_obs.get(a, 0)) for a in ("HA", "HA2", "HA3"))
        # PANAV needs HA to assign SS; without it the whole fit is empty.
        if method == "panav" and n_ha == 0:
            print(f"{bmrb_id} panav: no HA in deposit (skip)")
            continue

        try:
            cs.reref(method=method)
            predicted = dict(cs.reref_offsets or {})
        except Exception as exc:
            print(f"{bmrb_id} {method}: makeshift failed ({exc})")
            continue

        # ---- pair them up ------------------------------------------------
        # Atoms with no observations in the deposit can't be fit; PANAV still
        # often prints CO: 0.00 in that case — skip those, don't call them missing.
        missing = []
        skipped_absent = []
        for atom in atoms:
            ref = reference.get(atom)
            pred = predicted.get(atom)
            if ref is None:
                continue
            if pred is None:
                if int(n_obs.get(atom, 0)) == 0:
                    skipped_absent.append(atom)
                else:
                    missing.append(atom)
                continue
            rows[method].append((bmrb_id, atom, ref, pred))
            print(
                f"{bmrb_id:5d}  {method:5s}  {atom:2s}  "
                f"ref={ref:+7.3f}  ms={pred:+7.3f}  "
                f"Δ(ref−ms)={ref - pred:+7.3f}"
            )
        if skipped_absent:
            print(f"{bmrb_id} {method}: no {skipped_absent} in deposit (skip)")
        if missing:
            print(f"{bmrb_id} {method}: makeshift missing {missing}")
    print()

values = []
for method in ("lacs", "panav"):
    for _, _, ref, pred in rows[method]:
        values.append(ref)
        values.append(pred)

if values:
    lo, hi = min(values), max(values)
else:
    lo, hi = -1.0, 1.0

pad = 0.05 * (hi - lo) if hi > lo else 0.5
lo, hi = lo - pad, hi + pad

step = 10.0
for candidate in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0):
    if (hi - lo) / candidate <= 6:
        step = candidate
        break

ticks = np.arange(np.floor(lo / step) * step, np.ceil(hi / step) * step + 0.5 * step, step)
lims = [ticks[0], ticks[-1]]

# ---- plot ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for ax, method, title in (
    (axes[0], "lacs", "LACS"),
    (axes[1], "panav", "PANAV v2.1 (NMRbox)"),
):
    data = rows[method]

    for atom in ("N", "CA", "CB", "C"):
        xs = [ref for _, a, ref, _ in data if a == atom]
        ys = [pred for _, a, _, pred in data if a == atom]
        if not xs:
            continue
        color = ATOM_COLORS.get(atom, "0.4")
        # Open CA markers so coincident CA/CB points (common for protonated
        # entries) stay visible instead of CB covering CA.
        if atom == "CA":
            ax.scatter(
                xs, ys, s=30,  color=color, linewidth=1.4, label=atom, zorder=3, alpha=0.5
            )
        else:
            ax.scatter(
                xs, ys, s=30, color=color, linewidth=0.5, label=atom, zorder=2, alpha=0.5,
            )

    ax.plot(lims, lims, color="0.7", lw=1, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect("equal")
    ax.set_xlabel("reference offset (ppm)")
    ax.set_ylabel("makeshift offset (ppm)")

    if data:
        x = np.array([ref for _, _, ref, _ in data])
        y = np.array([pred for _, _, _, pred in data])
        r = pearsonr(x, y)[0]
        rmsd = float(np.sqrt(np.mean((x - y) ** 2)))
        ax.set_title(f"{title}\nr = {r:.4f}   RMSD = {rmsd:.3f} ppm   n = {len(data)}")
        ax.legend(frameon=False, fontsize=8, loc="best")
        print(f"{method.upper():5s}: n={len(data)}  pearson_r={r:.6f}  rmsd={rmsd:.4f} ppm")
    else:
        ax.set_title(f"{title}\n(no overlapping offsets)")

fig.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=200)
print(f"\nsaved {OUTPUT_FILE}")