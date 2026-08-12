"""
Compare makeshift LACS / PANAV offsets to BMRB reference values.

Fetches everything live (nothing is kept on disk except the output figure):
  - chemical shifts via ``ChemicalShifts.from_bmrb(..., keep_download=False)``
  - LACS offsets from ``…/bmr{id}/validation/LACS.str``
  - PANAV offsets from ``https://api.bmrb.io/current/entry/{id}/validate``

Entry selection:
  - default: the 9-protein RCI validation set
  - ``--ids-file``: one BMRB id per line
  - ``--all``: every macromolecule entry from the BMRB list API
    (~18k). Use ``--methods lacs`` for a faster LACS-only corpus run;
    PANAV validate is slow (~tens of seconds per entry).

Produces ``reref_validation.png``: LACS row (CA, CB, CO) and PANAV row
(N, CA, CB, CO), Makeshift on x, BMRB on y.

BMRB ``LACS.str`` does not deposit N/HN offsets, so LACS N is not compared.
PANAV N comes from the validate API.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, pearsonr

import makeshift as ms

HERE = Path(__file__).parent
OUTPUT_FILE = HERE / "reref_validation.png"

DEFAULT_IDS = [11080, 15451, 15490, 15521, 15581, 15763, 15959, 52018, 5991]

LIST_URL = "https://api.bmrb.io/v2/list_entries?database=macromolecules"
LACS_URL = (
    "https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr{eid}/validation/LACS.str"
)
PANAV_URL = "https://api.bmrb.io/current/entry/{eid}/validate"

LACS_ATOMS = ("CA", "CB", "C")
PANAV_ATOMS = ("N", "CA", "CB", "C")
LACS_YNAME = {
    "CO": "C", "CA": "CA", "CB": "CB", "HA": "HA",
    "N": "N", "HN": "H", "H": "H",
}
ATOM_LABEL = {"CA": "CA", "CB": "CB", "C": "CO", "N": "N"}


def _urlopen(url: str, timeout: float, method: str = "GET"):
    req = urllib.request.Request(
        url, method=method, headers={"User-Agent": "makeshift-reref-validation"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _urlopen_text(url: str, timeout: float) -> str:
    with _urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def list_macromolecule_ids(timeout: float = 120.0) -> list[int]:
    """All BMRB macromolecule entry ids (live list API)."""
    raw = json.loads(_urlopen_text(LIST_URL, timeout=timeout))
    return sorted({int(x) for x in raw})


def lacs_exists(eid: int, timeout: float = 15.0) -> bool:
    """True if BMRB hosts a LACS.str for this entry (HEAD)."""
    try:
        with _urlopen(LACS_URL.format(eid=eid), timeout=timeout, method="HEAD"):
            return True
    except Exception:
        return False


def fetch_lacs_offsets(eid: int, timeout: float = 25.0) -> dict[str, float] | None:
    """Parse BMRB-deposited LACS.str offsets. None if missing or empty."""
    try:
        text = _urlopen_text(LACS_URL.format(eid=eid), timeout=timeout)
    except urllib.error.HTTPError as exc:
        if getattr(exc, "code", None) == 404:
            return None
        return None
    except Exception:
        return None

    out: dict[str, float] = {}
    yname = None
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "_LACS_plot.Y_coord_name":
            yname = parts[1]
        elif parts[0] == "_LACS_plot.Y_axis_chem_shift_offset" and yname is not None:
            atom = LACS_YNAME.get(yname, yname)
            try:
                out[atom] = float(parts[1])
            except ValueError:
                pass
            yname = None
    return out or None


def fetch_panav_offsets(eid: int, timeout: float = 90.0) -> dict[str, float] | None:
    """Parse PANAV offsets from the BMRB validate API. None on miss/failure."""
    try:
        payload = json.loads(_urlopen_text(PANAV_URL.format(eid=eid), timeout=timeout))
    except Exception:
        return None
    if not payload:
        return None
    entry = next(iter(payload.values()))
    panav = entry.get("panav") if isinstance(entry, dict) else None
    if not isinstance(panav, dict) or not panav:
        return None
    block = next(iter(panav.values()))
    raw = block.get("offsets") if isinstance(block, dict) else None
    if not isinstance(raw, dict):
        return None
    out = {}
    for key, val in raw.items():
        atom = "C" if key == "CO" else key
        try:
            out[atom] = float(val)
        except (TypeError, ValueError):
            continue
    return out or None


def _pair_atoms(atoms, reference, predicted, n_obs):
    """Yield (atom, ref, pred) for atoms present on both sides."""
    for atom in atoms:
        ref = reference.get(atom)
        pred = predicted.get(atom)
        if ref is None or pred is None:
            continue
        if int(n_obs.get(atom, 0)) == 0 and abs(ref) < 1e-12:
            continue
        yield atom, float(ref), float(pred)


def process_entry(eid: int, do_lacs: bool = True, do_panav: bool = True) -> dict:
    """Fetch references + makeshift for one entry. No files written."""
    from makeshift.reref import compute_offsets

    result = {"id": eid, "lacs": [], "panav": [], "notes": []}

    lacs_ref = fetch_lacs_offsets(eid) if do_lacs else None
    panav_ref = fetch_panav_offsets(eid) if do_panav else None

    if lacs_ref is None and panav_ref is None:
        result["notes"].append("no reference offsets")
        return result

    try:
        cs = ms.ChemicalShifts.from_bmrb(eid, keep_download=False)
    except Exception as exc:
        result["notes"].append(f"from_bmrb failed: {exc}")
        return result

    n_obs = cs.data.Atom_ID.value_counts().to_dict()
    n_ha = sum(int(n_obs.get(a, 0)) for a in ("HA", "HA2", "HA3"))

    if lacs_ref is not None:
        try:
            pred, _, _ = compute_offsets(cs.data, method="lacs")
            pred = pred or {}
            for atom, ref, ms_v in _pair_atoms(LACS_ATOMS, lacs_ref, pred, n_obs):
                result["lacs"].append((eid, atom, ref, ms_v))
        except Exception as exc:
            result["notes"].append(f"lacs failed: {exc}")

    if panav_ref is not None:
        if n_ha == 0:
            result["notes"].append("no HA for PANAV")
        else:
            try:
                pred, _, _ = compute_offsets(cs.data, method="panav")
                pred = pred or {}
                for atom, ref, ms_v in _pair_atoms(PANAV_ATOMS, panav_ref, pred, n_obs):
                    result["panav"].append((eid, atom, ref, ms_v))
            except Exception as exc:
                result["notes"].append(f"panav failed: {exc}")

    return result


def prefilter_lacs_ids(
    ids: list[int],
    workers: int,
    max_keep: int | None = None,
) -> list[int]:
    """Keep entries that host a LACS.str (concurrent HEAD).

    If ``max_keep`` is set, scan in batches and stop once that many hits are
    found (avoids HEADing all ~18k ids for a small smoke test).
    """
    target = f" (stop at {max_keep})" if max_keep else ""
    print(f"Prefiltering {len(ids)} ids for LACS.str{target} …")
    kept: list[int] = []
    done = 0
    batch = max(workers * 4, 64)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for start in range(0, len(ids), batch):
            chunk = ids[start : start + batch]
            futures = {pool.submit(lacs_exists, eid): eid for eid in chunk}
            for fut in as_completed(futures):
                eid = futures[fut]
                done += 1
                try:
                    if fut.result():
                        kept.append(eid)
                except Exception:
                    pass
                if done % 500 == 0:
                    print(
                        f"  HEAD {done}/{len(ids)}  with_lacs={len(kept)}",
                        flush=True,
                    )
            if max_keep is not None and len(kept) >= max_keep:
                break
    kept.sort()
    if max_keep is not None:
        kept = kept[:max_keep]
    print(f"  HEAD done={done}  kept={len(kept)} with LACS.str\n")
    return kept


def _panel_stats(xs, ys):
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    if len(x) < 2:
        return None
    r = float(pearsonr(x, y)[0])
    slope = float(linregress(x, y).slope)
    rmsd = float(np.sqrt(np.mean((x - y) ** 2)))
    return r, slope, rmsd, len(x)


def _draw_atom_panels(axes, rows, atoms, method_title):
    """rows: list of (eid, atom, ref_bmrb, makeshift)."""
    for ax, atom in zip(axes, atoms):
        xs = [ms_v for _, a, _, ms_v in rows if a == atom]
        ys = [ref for _, a, ref, _ in rows if a == atom]
        label = ATOM_LABEL.get(atom, atom)
        if not xs:
            ax.set_title(f"{method_title} {label}\n(no pairs)")
            ax.set_aspect("equal")
            continue
        ax.scatter(xs, ys, s=12, alpha=0.35, edgecolors="none", color="0.25")
        stats = _panel_stats(xs, ys)
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        pad = 0.05 * (hi - lo) if hi > lo else 0.5
        lims = [lo - pad, hi + pad]
        ax.plot(lims, lims, ls="--", color="0.6", lw=1, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Makeshift {method_title} {label} offset (ppm)")
        ax.set_ylabel(f"BMRB {method_title} {label} offset (ppm)")
        if stats:
            r, slope, rmsd, n = stats
            ax.set_title(
                f"{method_title} {label}: Makeshift vs BMRB\n"
                f"r = {r:.3f}   slope = {slope:.3f}   "
                f"RMSD = {rmsd:.3f}   n = {n}"
            )
            print(
                f"{method_title:5s} {label:2s}: n={n}  r={r:.4f}  "
                f"slope={slope:.4f}  rmsd={rmsd:.4f}"
            )


def parse_ids(args) -> list[int]:
    """Return the raw id list before LACS prefilter / final limit."""
    if args.all:
        print("Fetching full BMRB macromolecule entry list …")
        ids = list_macromolecule_ids()
        print(f"  {len(ids)} macromolecule entries\n")
    elif args.ids_file:
        ids = []
        for line in Path(args.ids_file).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.search(r"(\d+)", line)
            if m:
                ids.append(int(m.group(1)))
        ids = sorted(set(ids))
    else:
        ids = list(DEFAULT_IDS)
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--all",
        action="store_true",
        help="Use every BMRB macromolecule entry (~18k)",
    )
    src.add_argument(
        "--ids-file",
        type=Path,
        help="File of BMRB ids (one per line)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only the first N usable ids (after LACS prefilter when that runs)",
    )
    parser.add_argument(
        "--methods",
        choices=("both", "lacs", "panav"),
        default="both",
        help="Which references to fetch (default both). "
        "'lacs' is much faster for --all.",
    )
    parser.add_argument(
        "--prefilter-lacs",
        action="store_true",
        help="HEAD-scan for LACS.str first and only process those entries "
        "(recommended with --all when LACS is included)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent workers (default 6; 16 with --all)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output figure path (default {OUTPUT_FILE.name})",
    )
    args = parser.parse_args(argv)

    do_lacs = args.methods in ("both", "lacs")
    do_panav = args.methods in ("both", "panav")
    workers = args.workers if args.workers is not None else (16 if args.all else 6)

    ids = parse_ids(args)
    # HEAD-filter to entries that host LACS.str when requested, or automatically
    # for LACS-only --all. With --limit, stop once enough LACS hits are found.
    will_prefilter = args.prefilter_lacs or (args.all and do_lacs and not do_panav)
    if will_prefilter:
        ids = prefilter_lacs_ids(
            ids,
            workers=max(workers, 24),
            max_keep=args.limit,
        )
    elif args.limit is not None:
        ids = ids[: max(0, args.limit)]

    if args.all and do_panav and not args.prefilter_lacs:
        print(
            "Note: --all with PANAV hits the validate API per entry and can take "
            "many hours. Prefer '--methods lacs' or add '--prefilter-lacs'.\n"
        )

    print(
        f"Comparing makeshift reref to live BMRB "
        f"({args.methods}; n_entries={len(ids)}; workers={workers})"
    )
    print("Nothing is written to disk except the output figure.\n")

    lacs_rows: list[tuple] = []
    panav_rows: list[tuple] = []
    n_with_pairs = 0
    log_every = 1 if len(ids) <= 50 else 50

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(process_entry, eid, do_lacs, do_panav): eid for eid in ids
        }
        done = 0
        for fut in as_completed(futures):
            eid = futures[fut]
            done += 1
            try:
                rec = fut.result()
            except Exception as exc:
                if len(ids) <= 50:
                    print(f"{eid}: worker error ({exc})")
                continue
            lacs_rows.extend(rec["lacs"])
            panav_rows.extend(rec["panav"])
            if rec["lacs"] or rec["panav"]:
                n_with_pairs += 1
            if len(ids) <= 50:
                note = f"  ({'; '.join(rec['notes'])})" if rec["notes"] else ""
                print(
                    f"[{done}/{len(ids)}] {eid}: "
                    f"lacs_pairs={len(rec['lacs'])}  "
                    f"panav_pairs={len(rec['panav'])}{note}"
                )
            elif done % log_every == 0 or done == len(ids):
                print(
                    f"[{done}/{len(ids)}] paired_entries={n_with_pairs}  "
                    f"lacs_pts={len(lacs_rows)}  panav_pts={len(panav_rows)}",
                    flush=True,
                )

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2))
    _draw_atom_panels(axes[0, :3], lacs_rows, LACS_ATOMS, "LACS")
    axes[0, 3].axis("off")
    _draw_atom_panels(axes[1, :], panav_rows, PANAV_ATOMS, "PANAV")

    fig.tight_layout()
    out = args.output
    fig.savefig(out, dpi=200)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
