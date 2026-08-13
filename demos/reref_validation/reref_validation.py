"""
Compare makeshift LACS / PANAV offsets to live BMRB references.

Fetches shifts via ``ChemicalShifts.from_bmrb``, LACS from ``LACS.str``,
PANAV from the validate API. ``--all`` caches ids in ``bmrb_all_ids.txt``;
results append to ``reref_validation_results.jsonl`` (resumable).

Output: one paper figure (LACS CA/CB/CO centered over PANAV N/CA/CB/CO)
plus a rows TSV. BMRB LACS.str has no N, so LACS N is not compared.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator
from scipy.stats import linregress, pearsonr

import makeshift as ms
from makeshift.reref import compute_offsets

HERE = Path(__file__).parent
OUTPUT_FILE = HERE / "reref_validation.png"
ALL_IDS_CACHE = HERE / "bmrb_all_ids.txt"
RESULTS_CACHE = HERE / "reref_validation_results.jsonl"
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


def _get(url: str, timeout: float, method: str = "GET") -> str:
    req = urllib.request.Request(
        url, method=method, headers={"User-Agent": "makeshift-reref-validation"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_lacs_offsets(eid: int) -> dict[str, float] | None:
    try:
        text = _get(LACS_URL.format(eid=eid), timeout=25)
    except Exception:
        return None
    out, yname = {}, None
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "_LACS_plot.Y_coord_name":
            yname = parts[1]
        elif parts[0] == "_LACS_plot.Y_axis_chem_shift_offset" and yname is not None:
            try:
                out[LACS_YNAME.get(yname, yname)] = float(parts[1])
            except ValueError:
                pass
            yname = None
    return out or None


def fetch_panav_offsets(eid: int) -> dict[str, float] | None:
    try:
        payload = json.loads(_get(PANAV_URL.format(eid=eid), timeout=90))
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
        try:
            out["C" if key == "CO" else key] = float(val)
        except (TypeError, ValueError):
            continue
    return out or None


def process_entry(eid: int, do_lacs: bool, do_panav: bool) -> dict:
    result = {"id": eid, "lacs": [], "panav": [], "notes": []}
    lacs_ref = panav_ref = cs = None
    cs_exc = None
    with ThreadPoolExecutor(max_workers=3) as net:
        f_lacs = net.submit(fetch_lacs_offsets, eid) if do_lacs else None
        f_panav = net.submit(fetch_panav_offsets, eid) if do_panav else None
        f_cs = net.submit(ms.ChemicalShifts.from_bmrb, eid, keep_download=False)
        if f_lacs is not None:
            try:
                lacs_ref = f_lacs.result()
            except Exception:
                lacs_ref = None
        if f_panav is not None:
            try:
                panav_ref = f_panav.result()
            except Exception:
                panav_ref = None
        try:
            cs = f_cs.result()
        except Exception as exc:
            cs_exc = exc

    if lacs_ref is None and panav_ref is None:
        result["notes"].append("no reference offsets")
        return result
    if cs is None:
        result["notes"].append(f"from_bmrb failed: {cs_exc}")
        return result

    n_obs = cs.data.Atom_ID.value_counts().to_dict()
    n_ha = sum(int(n_obs.get(a, 0)) for a in ("HA", "HA2", "HA3"))

    def _pairs(atoms, ref, pred):
        for atom in atoms:
            r, p = ref.get(atom), pred.get(atom)
            if r is None or p is None:
                continue
            if int(n_obs.get(atom, 0)) == 0 and abs(r) < 1e-12:
                continue
            yield atom, float(r), float(p)

    if lacs_ref is not None:
        try:
            pred, _, _ = compute_offsets(cs.data, method="lacs")
            for atom, ref, ms_v in _pairs(LACS_ATOMS, lacs_ref, pred or {}):
                result["lacs"].append((eid, atom, ref, ms_v))
        except Exception as exc:
            result["notes"].append(f"lacs failed: {exc}")

    if panav_ref is not None:
        if n_ha == 0:
            result["notes"].append("no HA for PANAV")
        else:
            try:
                pred, _, _ = compute_offsets(cs.data, method="panav", cona=False)
                for atom, ref, ms_v in _pairs(PANAV_ATOMS, panav_ref, pred or {}):
                    result["panav"].append((eid, atom, ref, ms_v))
            except Exception as exc:
                result["notes"].append(f"panav failed: {exc}")
    return result


def prefilter_lacs_ids(ids, workers, max_keep=None):
    print(
        f"Prefiltering {len(ids)} ids for LACS.str"
        + (f" (stop at {max_keep})" if max_keep else "")
        + " …",
        flush=True,
    )
    kept, done, t0 = [], 0, time.time()
    batch = max(workers * 4, 64)

    def exists(eid):
        try:
            _get(LACS_URL.format(eid=eid), timeout=15, method="HEAD")
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for start in range(0, len(ids), batch):
            chunk = ids[start : start + batch]
            futs = {pool.submit(exists, eid): eid for eid in chunk}
            for fut in as_completed(futs):
                done += 1
                try:
                    if fut.result():
                        kept.append(futs[fut])
                except Exception:
                    pass
                if done % 200 == 0:
                    print(f"  HEAD [{done}/{len(ids)}] kept={len(kept)}", flush=True)
            if max_keep is not None and len(kept) >= max_keep:
                break
    kept = sorted(kept)[:max_keep] if max_keep else sorted(kept)
    print(f"  HEAD done={done} kept={len(kept)} ({time.time() - t0:.0f}s)\n", flush=True)
    return kept


def load_results(path: Path, do_lacs: bool, do_panav: bool):
    done_ids, by_id = set(), {}
    if not path.is_file():
        return done_ids, [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            by_id[int(rec["id"])] = rec
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    lacs_rows, panav_rows = [], []
    for eid, rec in by_id.items():
        if do_lacs and not rec.get("did_lacs"):
            continue
        if do_panav and not rec.get("did_panav"):
            continue
        done_ids.add(eid)
        for key, dest in (("lacs", lacs_rows), ("panav", panav_rows)):
            for p in rec.get(key) or []:
                try:
                    dest.append((eid, str(p["atom"]), float(p["ref"]), float(p["ms"])))
                except (KeyError, TypeError, ValueError):
                    pass
    return done_ids, lacs_rows, panav_rows


def append_result(path: Path, rec: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        fh.flush()


def save_rows_tsv(lacs_rows, panav_rows, path: Path) -> None:
    path = Path(path)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    lines = ["method\tbmrb_id\tatom\tref_ppm\tmakeshift_ppm"]
    for method, rows in (("lacs", lacs_rows), ("panav", panav_rows)):
        for eid, atom, ref, ms_v in rows:
            lines.append(f"{method}\t{eid}\t{atom}\t{ref:.6f}\t{ms_v:.6f}")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_figure(lacs_rows, panav_rows, path: Path, *, announce: bool = True) -> None:
    """LACS (3, centered) over PANAV (4); equal panel sizes."""
    path = Path(path)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    tag = "writing figure" if announce else "  checkpoint: writing"
    print(
        f"{tag} → {path.name} (lacs={len(lacs_rows)} panav={len(panav_rows)}) …",
        flush=True,
    )

    fig = plt.figure(figsize=(7.2, 5.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.38)
    top = outer[0].subgridspec(1, 5, width_ratios=[0.5, 1, 1, 1, 0.5], wspace=0.28)
    bot = outer[1].subgridspec(1, 4, wspace=0.28)
    rows_axes = [
        (lacs_rows, LACS_ATOMS, "LACS", [fig.add_subplot(top[0, j]) for j in (1, 2, 3)]),
        (panav_rows, PANAV_ATOMS, "PANAV", [fig.add_subplot(bot[0, j]) for j in range(4)]),
    ]
    try:
        for rows, atoms, method, axes in rows_axes:
            for i, (ax, atom) in enumerate(zip(axes, atoms)):
                xs = [ms_v for _, a, _, ms_v in rows if a == atom]
                ys = [ref for _, a, ref, _ in rows if a == atom]
                label = ATOM_LABEL.get(atom, atom)
                vals = xs + ys
                if vals:
                    lo, hi = int(np.floor(min(vals))), int(np.ceil(max(vals)))
                    if lo == hi:
                        lo, hi = lo - 1, hi + 1
                    span = hi - lo
                    n_int = min(3, span)
                    step = int(np.ceil(span / n_int))
                    n_int = min(int(np.ceil(span / step)), 3)
                    hi = lo + step * n_int
                    ticks = list(range(lo, hi + 1, step))
                    lims = (lo, hi)
                else:
                    lims, ticks = (-1, 1), [-1, 0, 1]

                if len(xs) >= 2:
                    x, y = np.asarray(xs, float), np.asarray(ys, float)
                    r = float(pearsonr(x, y)[0])
                    slope = float(linregress(x, y).slope)
                    rmsd = float(np.sqrt(np.mean((x - y) ** 2)))
                    n = len(x)
                    ax.set_title(
                        f"{method} {label}\n$r$={r:.3f}  $n$={n}", fontsize=10, pad=6
                    )
                    if announce:
                        print(
                            f"{method:5s} {label:2s}: n={n}  r={r:.4f}  "
                            f"slope={slope:.4f}  rmsd={rmsd:.4f}",
                            flush=True,
                        )
                else:
                    ax.set_title(f"{method} {label}", fontsize=10, pad=6)
                    if announce:
                        print(f"{method:5s} {label:2s}: no pairs", flush=True)

                ax.set_xlabel("makeshift (ppm)", fontsize=9)
                ax.set_ylabel("BMRB (ppm)" if i == 0 else "", fontsize=9)
                ax.tick_params(labelsize=8, width=0.6, length=3)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.8)
                ax.set_xlim(lims)
                ax.set_ylim(lims)
                ax.set_aspect("equal", adjustable="box")
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.yaxis.set_major_locator(FixedLocator(ticks))
                ax.plot(lims, lims, ls="--", color="0.55", lw=0.8, zorder=0)
                if xs:
                    ax.scatter(
                        xs, ys, s=8, alpha=0.4, edgecolors="none",
                        color="0.15", rasterized=True,
                    )
        fig.savefig(tmp, dpi=300)
    finally:
        plt.close(fig)
    tmp.replace(path)
    print(f"  wrote {path} ({path.stat().st_size} bytes)", flush=True)


def load_ids_file(path: Path) -> list[int]:
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.search(r"(\d+)", line)
        if m:
            ids.append(int(m.group(1)))
    return sorted(set(ids))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--all", action="store_true", help=f"Use/build {ALL_IDS_CACHE.name}")
    src.add_argument("--ids-file", type=Path, help="File of BMRB ids")
    parser.add_argument("--refresh-ids", action="store_true", help="Rebuild id cache")
    parser.add_argument("--no-resume", action="store_true", help="Ignore results JSONL")
    parser.add_argument("--results", type=Path, default=RESULTS_CACHE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--methods", choices=("both", "lacs", "panav"), default="both")
    parser.add_argument("--prefilter-lacs", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("-o", "--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args(argv)

    do_lacs = args.methods in ("both", "lacs")
    do_panav = args.methods in ("both", "panav")
    workers = args.workers if args.workers is not None else (32 if args.all else 6)
    will_prefilter = args.prefilter_lacs or (args.all and do_lacs and not do_panav)
    used_cache = False

    if args.all and ALL_IDS_CACHE.is_file() and not args.refresh_ids:
        ids = load_ids_file(ALL_IDS_CACHE)
        used_cache = True
        print(
            f"Loaded {len(ids)} ids from {ALL_IDS_CACHE.name} "
            f"(--refresh-ids to rebuild)",
            flush=True,
        )
        if args.limit is not None:
            ids = ids[: max(0, args.limit)]
    else:
        if args.all:
            print("Fetching BMRB macromolecule list …", flush=True)
            ids = sorted({int(x) for x in json.loads(_get(LIST_URL, timeout=120))})
            print(f"  {len(ids)} entries\n", flush=True)
        elif args.ids_file:
            ids = load_ids_file(args.ids_file)
        else:
            ids = list(DEFAULT_IDS)

        if will_prefilter:
            ids = prefilter_lacs_ids(ids, workers=max(workers, 24), max_keep=args.limit)
        elif args.limit is not None:
            ids = ids[: max(0, args.limit)]

        if args.all and args.limit is None and ids:
            note = "LACS.str HEAD-prefiltered" if will_prefilter else "all macromolecules"
            ALL_IDS_CACHE.write_text(
                f"# n={len(ids)}\n# {note}\n"
                + "\n".join(str(i) for i in ids)
                + "\n",
                encoding="utf-8",
            )
            print(f"Saved {len(ids)} ids → {ALL_IDS_CACHE}", flush=True)

    if args.all and do_panav and not args.prefilter_lacs and not used_cache:
        print(
            "Note: --all + PANAV hits validate per entry (slow). "
            "Prefer --methods lacs or --prefilter-lacs.\n",
            flush=True,
        )

    if args.checkpoint_every is None:
        checkpoint_every = 100 if len(ids) > 50 else 0
    else:
        checkpoint_every = max(0, args.checkpoint_every)
    results_path = Path(args.results)
    rows_path = args.output.with_name(args.output.stem + "_rows.tsv")

    if args.no_resume:
        done_ids, lacs_rows, panav_rows = set(), [], []
        print("Resume off (--no-resume).\n", flush=True)
    else:
        done_ids, lacs_rows, panav_rows = load_results(results_path, do_lacs, do_panav)
        if done_ids:
            print(
                f"Resuming {results_path.name}: {len(done_ids)} done, "
                f"lacs_pts={len(lacs_rows)} panav_pts={len(panav_rows)}",
                flush=True,
            )

    todo = [eid for eid in ids if eid not in done_ids]
    n_total, n_done_prior = len(ids), len(ids) - len(todo)
    if n_done_prior:
        print(
            f"Skipping {n_done_prior} finished; {len(todo)} left "
            f"({n_done_prior}/{n_total} plotted).\n",
            flush=True,
        )
    print(
        f"Comparing makeshift vs BMRB ({args.methods}; n={n_total}; "
        f"n_todo={len(todo)}; workers={workers})",
        flush=True,
    )
    if checkpoint_every:
        print(
            f"Checkpoint every {checkpoint_every} → "
            f"{args.output.name} / {rows_path.name}",
            flush=True,
        )
    print(f"Results → {results_path.name}\n", flush=True)

    def checkpoint(announce: bool) -> None:
        if not (lacs_rows or panav_rows):
            return
        try:
            save_rows_tsv(lacs_rows, panav_rows, rows_path)
            print(f"  checkpoint rows → {rows_path}", flush=True)
        except Exception as exc:
            print(f"  checkpoint rows failed: {exc}", flush=True)
        try:
            save_figure(lacs_rows, panav_rows, args.output, announce=announce)
        except Exception as exc:
            print(f"  checkpoint figure failed: {exc}", flush=True)

    if n_done_prior and (lacs_rows or panav_rows):
        print(
            f"Writing figure from {n_done_prior} resumed entries "
            f"(lacs_pts={len(lacs_rows)} panav_pts={len(panav_rows)}) …",
            flush=True,
        )
        checkpoint(announce=True)

    if not todo:
        print("Nothing left to do.", flush=True)
        if not n_done_prior:
            checkpoint(announce=True)
        return

    n_with_pairs = len({r[0] for r in lacs_rows} | {r[0] for r in panav_rows})
    log_every = 1 if len(todo) <= 50 else 25
    t0, done_new, last_ckpt = time.time(), 0, n_done_prior

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {
            pool.submit(process_entry, eid, do_lacs, do_panav): eid for eid in todo
        }
        for fut in as_completed(futs):
            eid = futs[fut]
            done_new += 1
            completed = n_done_prior + done_new
            try:
                rec = fut.result()
            except Exception as exc:
                append_result(
                    results_path,
                    {
                        "id": eid,
                        "did_lacs": do_lacs,
                        "did_panav": do_panav,
                        "lacs": [],
                        "panav": [],
                        "notes": [f"worker error: {exc}"],
                    },
                )
                if len(todo) <= 50:
                    print(f"{eid}: worker error ({exc})", flush=True)
                continue

            append_result(
                results_path,
                {
                    "id": int(rec["id"]),
                    "did_lacs": do_lacs,
                    "did_panav": do_panav,
                    "lacs": [
                        {"atom": a, "ref": float(r), "ms": float(m)}
                        for _, a, r, m in rec["lacs"]
                    ],
                    "panav": [
                        {"atom": a, "ref": float(r), "ms": float(m)}
                        for _, a, r, m in rec["panav"]
                    ],
                    "notes": list(rec.get("notes") or []),
                },
            )
            lacs_rows.extend(rec["lacs"])
            panav_rows.extend(rec["panav"])
            if rec["lacs"] or rec["panav"]:
                n_with_pairs += 1

            if len(todo) <= 50:
                note = f"  ({'; '.join(rec['notes'])})" if rec["notes"] else ""
                print(
                    f"[{completed}/{n_total}] {eid}: "
                    f"lacs={len(rec['lacs'])} panav={len(rec['panav'])}{note}",
                    flush=True,
                )
            elif done_new % log_every == 0 or done_new == len(todo):
                elapsed = max(time.time() - t0, 1e-6)
                rate = done_new / elapsed
                eta = (n_total - completed) / rate if rate else float("inf")
                print(
                    f"[{completed}/{n_total}]  {100 * completed / max(n_total, 1):5.1f}%  "
                    f"{rate:.2f}/s  eta={eta / 60:.0f}m  paired={n_with_pairs}  "
                    f"lacs_pts={len(lacs_rows)} panav_pts={len(panav_rows)}",
                    flush=True,
                )

            if (
                checkpoint_every
                and completed % checkpoint_every == 0
                and completed != last_ckpt
            ):
                last_ckpt = completed
                checkpoint(announce=False)

    print(flush=True)
    try:
        save_rows_tsv(lacs_rows, panav_rows, rows_path)
        print(f"saved rows {rows_path}", flush=True)
    except Exception as exc:
        print(f"rows save failed: {exc}", flush=True)
    try:
        save_figure(lacs_rows, panav_rows, args.output, announce=True)
    except Exception as exc:
        print(f"figure save failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
