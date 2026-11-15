"""
Read CPMG acquisition parameters from a Bruker ``acqus`` file.

``acqus`` is JCAMP-DX: ``##$NAME= value`` lines, where arrays such as the
delay list ``D`` and pulse list ``P`` start with a ``(0..n)`` token and may
continue over several lines.

Which delay/pulse index holds which CPMG quantity depends on the pulse
program. :data:`CPMG_ACQUS_MAP` follows the N15CPMG.wtg pulse program
(time_equil = d20, 15N CPMG 90° = p30 / 2); check it against your pulse
program before relying on it.

The CPMG constant-time period T is *not* read from acqus: how it relates to
the delays depends on the pulse program (in N15CPMG.wtg.NEW_tst1, d17 is not
T), so it always comes from the config's ``time_T2``. ``d17`` is returned
only for comparison.
"""

import re
from pathlib import Path

#: acqus source of each CPMG parameter (N15CPMG.wtg pulse program).
CPMG_ACQUS_MAP = {
    "time_equil": "D[20]",     # equilibration delay, s
    "pw90": "P[30] / 2",       # p30 is the 15N CPMG 180 (us); pw90 = p30 / 2
    "carrier": "O3 / BF3",     # 15N offset (Hz) / 15N base frequency (MHz) = ppm
    "h_larmor_frq": "BF1",     # 1H base frequency, MHz
    "temperature": "TE - 273.15",  # K -> °C
}

_PARAM = re.compile(r"##\$(\w+)=\s*(.*?)(?=\n##|\Z)", re.S)


def read_acqus(path):
    """
    Parse a Bruker ``acqus`` file into ``{NAME: value}``.

    Arrays (``D``, ``P``, ``PLW``, ...) become lists, numeric scalars become
    floats, and ``<text>`` strings lose their angle brackets.
    """
    text = Path(path).read_text(errors="replace")
    par = {}
    for m in _PARAM.finditer(text):
        name, raw = m.group(1), m.group(2).strip()
        tokens = raw.split()
        if tokens and re.fullmatch(r"\(\d+\.\.\d+\)", tokens[0]):
            par[name] = [_number(t) for t in tokens[1:]]
        elif raw.startswith("<") and raw.endswith(">"):
            par[name] = raw[1:-1]
        else:
            par[name] = _number(raw)
    return par


def _number(token):
    try:
        return float(token)
    except ValueError:
        return token


def cpmg_params_from_acqus(path):
    """
    CPMG experiment parameters from an ``acqus`` file, using
    :data:`CPMG_ACQUS_MAP`.

    Returns a dict with ``time_equil`` (s), ``pw90`` (s), ``carrier``
    (15N, ppm), ``h_larmor_frq`` (MHz), ``temperature`` (°C), plus ``d17``
    (s) and ``pulse_program`` (``PULPROG``) for reference. The constant-time
    period T is not included; see the module docstring.
    """
    par = read_acqus(path)
    try:
        D, P = par["D"], par["P"]
        out = {
            "time_equil": float(D[20]),
            "pw90": float(P[30]) / 2 * 1e-6,
            "carrier": float(par["O3"]) / float(par["BF3"]),
            "h_larmor_frq": float(par["BF1"]),
            "temperature": float(par["TE"]) - 273.15,
        }
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as e:
        raise ValueError(f"{path}: could not read CPMG parameters from acqus "
                         f"({type(e).__name__}: {e})") from None
    if out["pw90"] <= 0:
        raise ValueError(f"{path}: acqus gives pw90={out['pw90']} s; expected > 0. "
                         f"Check that the pulse program matches CPMG_ACQUS_MAP "
                         f"(pulse program: {par.get('PULPROG')})")
    out["d17"] = float(D[17]) if len(D) > 17 else None
    out["pulse_program"] = par.get("PULPROG")
    return out
