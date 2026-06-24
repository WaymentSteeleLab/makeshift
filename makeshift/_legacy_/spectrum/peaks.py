import numpy as np
import pandas as pd
import nmrglue as ng

from ..io.io import estimate_background, _annotate_ppm

# ---------------------------------------------------------------------------
# Peak picking
# ---------------------------------------------------------------------------

def pick_peaks(ref_data, baseline=10, algorithm="downward",
               est_params=True, h_ppm_min=6.0, h_ppm_max=11.0):
    """
    Pick peaks in the amide region of a 2D ¹H-¹⁵N spectrum.

    Parameters
    ----------
    ref_data : Spectrum
        Output of read_ucsf (data + per-axis unit-conversion objects).
    baseline : float
        Peaks above baseline × noise floor are picked.
    algorithm : str
        nmrglue peak-picking algorithm ('downward' or 'connected').
    h_ppm_min, h_ppm_max : float
        ¹H window to search (ppm).

    Returns
    -------
    peaks_df : DataFrame
        Columns: N_axis, H_axis, N_lw, H_lw, est_vol, cid, N_ppm, H_ppm.
        est_vol is the picker's volume estimate — treat as approximate.
    """
    uc = ref_data.uc
    data = np.array(ref_data.data, dtype=float)
    ppm_w2 = uc[1].ppm_scale()
    col_mask = (ppm_w2 >= h_ppm_min) & (ppm_w2 <= h_ppm_max)
    col_start = int(np.where(col_mask)[0][0])
    col_stop = int(np.where(col_mask)[0][-1]) + 1
    data_slice = data[:, col_start:col_stop]

    background = estimate_background(data_slice)
    threshold = baseline * background

    peaks_table = ng.analysis.peakpick.pick(
        data_slice, pthres=threshold, nthres=None,
        algorithm=algorithm, est_params=est_params,
        diag=False, edge=(0, col_start), table=True,
    )
    peaks_df = _annotate_ppm(peaks_table, uc)
    return peaks_df


# ---------------------------------------------------------------------------
# BMRB / peaklist loading
# ---------------------------------------------------------------------------

def load_peaklist(peaklist, seq_offset=0):
    """
    Load backbone amide ¹H/¹⁵N chemical shift assignments, from either a
    BMRB entry or a local peaklist CSV.

    Returns a DataFrame with one row per assigned residue:
        Seq_ID, Auth_seq_ID, Comp_ID, H_ppm, N_ppm, assn_label

    assn_label is one-letter code + Seq_ID (e.g. 'K28'). Seq_ID is the
    1-indexed sequence position, so it lines up directly with a 1-indexed
    construct `sequence` string.

    Parameters
    ----------
    peaklist : int, str, or Path
        Either a BMRB accession number (int, or a string of digits), or
        the path to a local peaklist CSV.

        If a BMRB id: downloads and parses the corresponding NMR-STAR
        file. Seq_ID/Auth_seq_ID come from BMRB directly. If multiple
        chemical shift saveframes are present the first is used and a
        note is printed.

        If a CSV path: expected columns are res, shift, atom — one row
        per (residue, atom) pair, e.g.::

            res     shift    atom
            R4      8.004    1H
            R4      123.022  15N

        `res` is a one-letter amino-acid code followed by the author
        residue number (e.g. 'R4'). Auth_seq_ID is taken from this
        number, and Seq_ID = Auth_seq_ID + seq_offset.
    seq_offset : int
        Only used for local CSV peaklists. Added to the author residue
        number to get Seq_ID. Default 0 (i.e. Seq_ID == Auth_seq_ID).
        Ignored for BMRB entries.
    """
    is_bmrb = isinstance(peaklist, int) or (isinstance(peaklist, str) and peaklist.strip().isdigit())

    if is_bmrb:
        from ..io.parsing import fetch_nmrstar_file, parse_nmr_star
        from ..io.ChemShifts import get_chem_shifts

        bmrb_id = peaklist
        str_file, _ = fetch_nmrstar_file(bmrb_id)
        parsed  = parse_nmr_star(str_file)
        cs_all  = get_chem_shifts(parsed)

        saveframes = cs_all['cs_saveframe_id'].unique()
        if len(saveframes) > 1:
            print(f'  Note: {len(saveframes)} chemical shift saveframes — '
                  f'using first ({saveframes[0]}). Others: {list(saveframes[1:])}')
        cs = cs_all[cs_all['cs_saveframe_id'] == saveframes[0]]

        n_df = (cs[cs['Atom_ID'] == 'N']
                [['Seq_ID', 'Auth_seq_ID', 'Comp_ID', 'Val']]
                .rename(columns={'Val': 'N_ppm'}))
        h_df = (cs[cs['Atom_ID'].isin(['H', 'HN'])]
                [['Seq_ID', 'Val']]
                .rename(columns={'Val': 'H_ppm'}))
        out = (n_df.merge(h_df, on='Seq_ID')
                   .dropna(subset=['H_ppm', 'N_ppm'])
                   .reset_index(drop=True))
        out['assn_label'] = (
            out['Comp_ID'].map(_AA_3TO1).fillna('?') +
            out['Seq_ID'].astype(str)
        )
        print(f'  {len(out)} backbone amide assignments loaded from BMRB entry {bmrb_id}')
        return out

    _AA_1TO3 = {v: k for k, v in _AA_3TO1.items()}

    df = pd.read_csv(peaklist)
    df = df.copy()
    df['aa_1'] = df['res'].str.extract(r'^([A-Za-z])')
    df['Auth_seq_ID'] = df['res'].str.extract(r'(\d+)').astype(int)
    df['Comp_ID'] = df['aa_1'].map(_AA_1TO3)

    n_df = (df[df['atom'] == '15N']
            [['Auth_seq_ID', 'Comp_ID', 'shift']]
            .rename(columns={'shift': 'N_ppm'}))
    h_df = (df[df['atom'] == '1H']
            [['Auth_seq_ID', 'shift']]
            .rename(columns={'shift': 'H_ppm'}))

    out = (n_df.merge(h_df, on='Auth_seq_ID')
               .dropna(subset=['H_ppm', 'N_ppm'])
               .reset_index(drop=True))
    out['Seq_ID'] = out['Auth_seq_ID'] + seq_offset
    out['assn_label'] = (
        out['Comp_ID'].map(_AA_3TO1).fillna('?') +
        out['Seq_ID'].astype(str)
    )
    print(f'  {len(out)} backbone amide assignments loaded from {peaklist}')
    return out[['Seq_ID', 'Auth_seq_ID', 'Comp_ID', 'H_ppm', 'N_ppm', 'assn_label']]
