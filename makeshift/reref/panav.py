"""
PANAV re-referencing (Wang & Wishart 2005; Wang et al. 2010).

HA→SS (PSSI joint P_s + 5-residue B/C density smooth), then iterative
N/CA/CB/C offsets vs panav_distns.csv. Offset = mean(d_ave - d_obs);
corrected = Val + offset. Optional CONA fragment scan afterward (does not
change offsets).
"""

import numpy as np
import pandas as pd

from ..data.tables import get_panav_distns
from ..utils.constants import _AA_3TO1, _SS

_REF = get_panav_distns()
_NO_REF = {("PRO", "N"), ("PRO", "H"), ("GLY", "CB")}
_OFFSET_ATOMS = ("N", "CA", "CB", "C")
_ALL_ATOMS = ("N", "CA", "CB", "C", "H", "HA")
_STD = {"C": 175.7, "CA": 56.6, "CB": 34.4, "N": 119.3, "H": 7.93, "HA": 4.41}
_MIN_N = 25
_CONA_TOL = 0.1
_DENSITY_CUT = 0.35  # PSSI: promote to majority type if P > this


def _ok(aa, atom):
    if (aa, atom) in _NO_REF:
        return False
    try:
        for ss in _SS:
            mu, sig = _REF[aa][ss][atom]
            if pd.isna(mu) or pd.isna(sig) or sig == 0:
                return False
        return True
    except KeyError:
        return False


def _gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))


def _ss_prob(aa, shifts, atoms):
    """
    Joint P_s over ``atoms`` (normalize-then-product; missing atoms skipped).
    Returns (best_ss, probs) or (None, None).
    """
    raw = {ss: 1.0 for ss in _SS}
    n = 0
    for atom in atoms:
        if atom not in shifts or not _ok(aa, atom):
            continue
        g = {ss: _gauss(shifts[atom], *_REF[aa][ss][atom]) for ss in _SS}
        gsum = sum(g.values())
        if gsum <= 0:
            continue
        n += 1
        for ss in _SS:
            raw[ss] *= g[ss] / gsum
    if n == 0:
        return None, None
    tot = sum(raw.values())
    probs = {ss: raw[ss] / tot for ss in _SS}
    return max(_SS, key=probs.get), probs


def _smooth(seq_ids, ss, probs):
    """5-residue B/C density filter: majority type wins if P > 0.35."""
    for i in range(len(seq_ids) - 4):
        win = seq_ids[i:i + 5]
        if win[-1] - win[0] != 4:
            continue
        labels = [ss.get(s) for s in win]
        if any(lab is None for lab in labels):
            continue
        for maj in ("E", "C"):  # strand / coil (paper B / C)
            if labels.count(maj) / 5.0 <= 0.5:
                continue
            for s, lab in zip(win, labels):
                if lab != maj and probs.get(s, {}).get(maj, 0.0) > _DENSITY_CUT:
                    ss[s] = maj


def _residue_prob(aa, shifts):
    """Best SS joint Gauss product (CONA)."""
    best = 0.0
    for ss in _SS:
        p, n = 1.0, 0
        for atom, val in shifts.items():
            if atom not in _ALL_ATOMS or not _ok(aa, atom):
                continue
            p *= _gauss(val, *_REF[aa][ss][atom])
            n += 1
        if n:
            best = max(best, p)
    return best


def _mean_3sigma(vals):
    a = np.asarray(vals, float)
    if len(a) < _MIN_N:
        return None
    m, s = a.mean(), a.std()
    if s > 0:
        a = a[np.abs(a - m) <= 3 * s]
    if len(a) < _MIN_N:
        return None
    return float(a.mean())


def reref_panav(df, cona=True):
    """Return ``(offsets, check, cona_meta)``.

    CONA runs after offsets are fit and does not change them. Pass
    ``cona=False`` to skip the fragment scan (faster when only offsets matter).
    """
    df = df.copy()
    df["Atom_ID"] = df["Atom_ID"].replace({"HA2": "HA", "HA3": "HA"})
    df["Comp_ID"] = df["Comp_ID"].str.upper()

    orig = {}
    for _, r in df.iterrows():
        atom = r["Atom_ID"]
        if atom not in _ALL_ATOMS or pd.isna(r["Val"]):
            continue
        sid = int(r["Seq_ID"])
        rec = orig.setdefault(sid, {"aa": r["Comp_ID"], "shifts": {}})
        rec["aa"] = r["Comp_ID"]
        v = float(r["Val"])
        prev = rec["shifts"].get(atom)
        rec["shifts"][atom] = v if prev is None else 0.5 * (prev + v)

    seq_ids = sorted(orig)
    none = {a: None for a in _OFFSET_ATOMS}
    if not seq_ids:
        return none, {a: False for a in _OFFSET_ATOMS}, None

    # crude offsets → deviant (6σ from every SS)
    crude = {}
    for atom in _ALL_ATOMS:
        vals = [
            orig[s]["shifts"][atom]
            for s in seq_ids
            if atom in orig[s]["shifts"]
            and not (atom == "CA" and orig[s]["aa"] == "GLY")
            and not (atom == "CB" and orig[s]["aa"] in ("ALA", "SER", "THR"))
        ]
        crude[atom] = (float(np.mean(vals)) - _STD[atom]) if vals else 0.0

    deviant = {s: set() for s in seq_ids}
    for s in seq_ids:
        aa = orig[s]["aa"]
        for atom, val in orig[s]["shifts"].items():
            if not _ok(aa, atom):
                continue
            if all(
                abs(val + crude[atom] - _REF[aa][ss][atom][0])
                > 6 * _REF[aa][ss][atom][1]
                for ss in _SS
            ):
                deviant[s].add(atom)

    ss = {}

    def assign_ss(cn_off):
        atoms = ("HA", "N", "CA", "CB", "C") if cn_off is not None else ("HA",)
        probs = {}
        for s in seq_ids:
            sh = dict(orig[s]["shifts"])
            if cn_off is not None:
                for atom in _OFFSET_ATOMS:
                    if cn_off.get(atom) is not None and atom in sh:
                        sh[atom] = sh[atom] + cn_off[atom]
            label, p = _ss_prob(orig[s]["aa"], sh, atoms)
            ss[s], probs[s] = label, (p or {})
        _smooth(seq_ids, ss, probs)

    def compute_offsets():
        out, ok = {}, {}
        for atom in _OFFSET_ATOMS:
            deltas = []
            for s in seq_ids:
                aa, sh = orig[s]["aa"], orig[s]["shifts"]
                if atom in deviant[s] or atom not in sh or ss.get(s) is None:
                    continue
                if not _ok(aa, atom):
                    continue
                mu, _ = _REF[aa][ss[s]][atom]
                deltas.append(mu - sh[atom])
            off = _mean_3sigma(deltas)
            out[atom], ok[atom] = (off, True) if off is not None else (None, False)
        return out, ok

    # HA→SS → offsets; then two rounds with trial-adjusted C/N
    assign_ss(None)
    offsets, check = compute_offsets()
    for _ in range(2):
        assign_ss(offsets)
        offsets, check = compute_offsets()

    cal = {
        s: {
            "aa": orig[s]["aa"],
            "shifts": {
                a: (v + offsets[a] if a in _OFFSET_ATOMS and offsets.get(a) is not None else v)
                for a, v in orig[s]["shifts"].items()
            },
            "ss": ss[s],
        }
        for s in seq_ids
    }

    # CONA: score each 3–6-mer under every same-length window in the protein.
    # Runs after offsets are final — skipping it does not change offsets.
    if not cona:
        return offsets, check, None

    cona_meta, suggestions = {}, []
    tot_sel = tot_conf = 0
    for k in (3, 4, 5, 6):
        windows = []
        for i in range(len(seq_ids) - k + 1):
            ids = seq_ids[i:i + k]
            if ids[-1] - ids[0] != k - 1:
                continue
            n_cacb = sum(
                ("CA" in cal[s]["shifts"]) + ("CB" in cal[s]["shifts"]) for s in ids
            )
            if n_cacb < k:
                continue
            aas = [cal[s]["aa"] for s in ids]
            shs = [cal[s]["shifts"] for s in ids]
            seq1 = "".join(_AA_3TO1.get(a, "X") for a in aas)
            windows.append((ids[0], ids[-1], aas, seq1, shs))

        selected = confirmed = 0
        for oi, (start, end, _, seq_o, shs_o) in enumerate(windows):
            selected += 1
            scores = np.array([
                float(np.prod([_residue_prob(a, sh) for a, sh in zip(w[2], shs_o)]))
                for w in windows
            ])
            p_max = float(scores.max()) if len(scores) else 0.0
            j_best = int(np.argmax(scores)) if len(scores) else oi
            norm = scores / p_max if p_max > 0 else scores
            if float(norm[oi]) >= 1.0 - _CONA_TOL:
                confirmed += 1
            elif j_best != oi:
                suggestions.append({
                    "start": int(start),
                    "end": int(end),
                    "original": seq_o,
                    "original_pct": round(100.0 * float(norm[oi]), 2),
                    "suggested": windows[j_best][3],
                    "suggested_pct": 100.0,
                    "suggested_start": int(windows[j_best][0]),
                    "suggested_end": int(windows[j_best][1]),
                })

        cona_meta[f"{k}-residue"] = {
            "selected": selected,
            "confirmed": confirmed,
            "score": (100.0 * confirmed / selected) if selected else None,
        }
        tot_sel += selected
        tot_conf += confirmed

    suspicious = []
    for s in seq_ids:
        aa, sh, ssi = cal[s]["aa"], cal[s]["shifts"], cal[s]["ss"]
        if ssi is None:
            continue
        for atom, val in sh.items():
            if not _ok(aa, atom):
                continue
            mu, sig = _REF[aa][ssi][atom]
            if abs(val - mu) > 4 * sig:
                suspicious.append(
                    {"seq_id": int(s), "aa": aa, "atom": atom, "val": val}
                )

    cona_meta["overall"] = {
        "selected": tot_sel,
        "confirmed": tot_conf,
        "score": (100.0 * tot_conf / tot_sel) if tot_sel else None,
    }
    if suggestions:
        cona_meta["suggestions"] = suggestions
    if suspicious:
        cona_meta["suspicious"] = suspicious
    deviant_list = [
        {"seq_id": int(s), "aa": orig[s]["aa"], "atom": a}
        for s, atoms in deviant.items() for a in atoms
    ]
    if deviant_list:
        cona_meta["deviant"] = deviant_list

    return offsets, check, cona_meta
