"""
Helpers for the TALOS-N subpackage
"""

import warnings
from pathlib import Path

import pandas as pd

from ..utils.constants import _AA_3TO1, _BACKBONE

BACKBONE_ATOMS = tuple(_BACKBONE) + ("HN",)


def aa_code_to_one_letter(resname):
    resname = str(resname).strip()
    if len(resname) == 1:
        return resname.upper()
    return _AA_3TO1.get(resname.upper(), (resname.upper()[:1] or "X"))


def filter_backbone(df, atoms=BACKBONE_ATOMS):
    return df[df["Atom_ID"].isin(atoms)].copy()


def infer_sequence(df):
    seq = df[["Seq_ID", "Comp_ID"]].drop_duplicates().sort_values("Seq_ID")
    return "".join(aa_code_to_one_letter(aa) for aa in seq["Comp_ID"])


def shifts_to_tab(df, output_path, sequence=None, first_resid=None):
    output_path = Path(output_path)

    if sequence is None:
        sequence = infer_sequence(df)
        warnings.warn(
            "No sequence supplied; inferring from chemical shift table. "
            "For BMRB entries, pass the full polymer sequence explicitly.",
            stacklevel=2,
        )
    if first_resid is None:
        first_resid = int(df["Seq_ID"].min())

    df_sorted = df.sort_values(["Seq_ID", "Atom_ID"])

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"DATA FIRST_RESID {first_resid}\n")
        for offset in range(0, len(sequence), 60):
            handle.write(f"DATA SEQUENCE {sequence[offset:offset + 60]}\n")
        handle.write("\n")
        handle.write("VARS RESID RESNAME ATOMNAME SHIFT\n")
        handle.write("FORMAT %4d %1s %4s %8.3f\n")
        handle.write("\n")

        for _, row in df_sorted.iterrows():
            handle.write(
                f"{int(row['Seq_ID']):4d} "
                f"{aa_code_to_one_letter(row['Comp_ID']):1s} "
                f"{str(row['Atom_ID'])[:4]:4s} "
                f"{float(row['Val']):8.3f}\n"
            )

    return output_path


def parse_tab(tab_path):
    tab_path = Path(tab_path)
    if not tab_path.is_file():
        raise FileNotFoundError(f"TALOS-N tab file not found: {tab_path}")

    with open(tab_path, encoding="utf-8") as handle:
        lines = handle.readlines()

    columns = None
    data_start = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("VARS"):
            columns = stripped.split()[1:]
            data_start = index + 1
            if index + 1 < len(lines) and lines[index + 1].strip().startswith("FORMAT"):
                data_start = index + 2
            break

    if columns is None or data_start is None:
        raise ValueError(f"Could not find VARS section in {tab_path}")

    rows = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("REMARK"):
            continue
        parts = stripped.split()
        if len(parts) >= len(columns):
            rows.append(parts[: len(columns)])

    if not rows:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(rows, columns=columns)
    for col in result.columns:
        converted = pd.to_numeric(result[col], errors="coerce")
        if converted.notna().any():
            result[col] = converted
    return result