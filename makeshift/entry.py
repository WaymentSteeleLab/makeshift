import os
import re
import warnings
import tempfile
import urllib.request
from collections import defaultdict

import pandas as pd
import numpy as np

from .utils.constants import _UNIPROT_DBCODES, _UNIPROT_RE
from .utils.constants import _DEUTER_KEYWORDS, _METHYL_KEYWORDS, _DENATURANT_KEYWORDS

_BMRB_URL = "https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr{id}/bmr{id}_3.str"
_VALUE_RE = re.compile(r'(?:"[^"]*"|\'[^\']*\'|[^\s]+)')

class _CategoryView(dict):
    """A {category: {framecode: saveframe}} mapping with attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"no saveframe category {name!r}; available: {', '.join(self)}"
            ) from None

    def __dir__(self):
        return list(self.keys())

    def __repr__(self):
        return f"<categories: {', '.join(self)}>"


class NMRStarEntry:
    """A parsed NMR-STAR file, indexed by saveframe category and framecode.

    Construct with one of the classmethods rather than the initializer:

        entry = NMRStarEntry.from_bmrb(15000)
        entry = NMRStarEntry.from_file("bmr15000_3.str")

    The parsed structure is available as ``entry.data``:
        data[Sf_category][Sf_framecode] -> {tag: value, _Loop_name: [rows]}
    """

    def __init__(self, data=None, entry_id=None, source_file=None):
        self.data = data or {}
        self.entry_id = entry_id
        self.source_file = source_file

    def get_entry(self):
        return self.entry_id

    @classmethod
    def from_file(cls, file_path, entry_id=None):
        return cls(data=cls._parse(file_path), entry_id=entry_id, source_file=file_path)

    @classmethod
    def from_bmrb(cls, bmrb_id, output_dir="", keep_download=False):
        """Download a BMRB entry and parse it.

        By default the .str file is fetched to a temp file and removed once
        parsed; pass ``keep_download=True`` to write it to ``output_dir``.
        """
        url = _BMRB_URL.format(id=bmrb_id)

        if keep_download:
            path = os.path.join(output_dir, f"bmr{bmrb_id}_3.str")
            urllib.request.urlretrieve(url, path)
            return cls.from_file(path, entry_id=bmrb_id)

        fd, path = tempfile.mkstemp(suffix=f"_bmr{bmrb_id}_3.str")
        os.close(fd)
        try:
            urllib.request.urlretrieve(url, path)
            data = cls._parse(path)
        finally:
            os.remove(path)
        return cls(data=data, entry_id=bmrb_id)

    # parser
    @staticmethod
    def _parse(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()

        data = {}
        saveframe_name = None
        in_saveframe = False
        current_tags = {}
        current_loops = defaultdict(list)

        def flush():
            if in_saveframe and saveframe_name:
                data[saveframe_name] = {**current_tags, **current_loops}

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("save_"):
                flush()
                saveframe_name = line[5:].strip() or None
                current_tags = {}
                current_loops = defaultdict(list)
                in_saveframe = bool(saveframe_name)
                i += 1
                continue

            if not in_saveframe:
                i += 1
                continue

            if line == "loop_":
                loop_tags = []
                i += 1
                while i < len(lines) and lines[i].strip().startswith("_"):
                    loop_tags.append(lines[i].strip())
                    i += 1
                loop_category = loop_tags[0].split(".")[0] if loop_tags else ""
                base_prefix = loop_category + "."
                while i < len(lines):
                    data_line = lines[i].strip()
                    if data_line == "stop_":
                        break
                    if data_line:
                        values = _VALUE_RE.findall(data_line)
                        if len(values) == len(loop_tags):
                            row = {tag[len(base_prefix):] if tag.startswith(base_prefix) else tag:
                                   val.strip('"')
                                   for tag, val in zip(loop_tags, values)}
                            current_loops[loop_category].append(row)
                    i += 1
                i += 1  # past stop_
                continue

            if line == "stop_":
                i += 1
                continue

            if line.startswith("_"):
                key = line.split()[0].split(".")[-1]
                # multiline (semicolon-delimited) value
                if len(line.split()) == 1 and i + 1 < len(lines) and lines[i + 1].strip() == ";":
                    i += 2
                    value_lines = []
                    while i < len(lines) and lines[i].strip() != ";":
                        value_lines.append(lines[i].rstrip("\n"))
                        i += 1
                    current_tags[key] = "".join(value_lines)
                    i += 1  # past closing ';'
                else:
                    parts = line.split(None, 1)
                    current_tags[key] = parts[1].strip("\"'") if len(parts) > 1 else ""
                    i += 1
                continue

            i += 1

        flush()
        return NMRStarEntry._restructure(data)

    @property
    def categories(self):
        """Saveframe categories as an attribute-accessible mapping.

            entry.categories                           -> <categories: entity, ...>
            list(entry.categories)                     -> ['entity', ...]
            entry.categories.assigned_chemical_shifts  -> {framecode: saveframe}
        """
        return _CategoryView(self.data)

    def saveframe(self, category, framecode=None):
        """Return one saveframe dict, or all framecodes for a category."""
        cat = self.data.get(category, {})
        return cat if framecode is None else cat[framecode]

    @staticmethod
    def loop_to_dataframe(loop):
        """Turn a loop (list of row dicts) into a DataFrame."""
        if not loop:
            return pd.DataFrame()
        cols = {k: [] for k in loop[0].keys()}
        for row in loop:
            for k, v in row.items():
                cols[k].append(v)
        return pd.DataFrame.from_records(cols)

    def _loop_records(self, category, loop_name, tags, id_key):
        frames = []
        for framecode, sf in self.saveframe(category).items():
            if loop_name in sf:
                df = self.loop_to_dataframe(sf[loop_name]).reindex(columns=tags)
                df.insert(0, id_key, framecode)
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _clean(val):
        val = (val or "").strip().strip("'\"")
        return val if val and val not in (".", "?", "N/A") else None
   
    @staticmethod
    def _num(val):
        val = (val or "").strip().strip("'\"")
        if not val or val in (".", "?", "N/A"):
            return np.nan
        try:
            return float(val)
        except ValueError:
            return val

    # access 
    
    def sequences(self, entity_id=None):
        """One row per entity: ID, polymer type, one-letter sequence."""
        tags = ["ID", "Polymer_type", "Polymer_seq_one_letter_code"]
        rows = []
        for framecode, sf in self.saveframe("entity").items():
            rows.append({"entity": framecode, **{t: sf.get(t) for t in tags}})
        seq_df = pd.DataFrame.from_records(rows, columns=["entity"] + tags)
        seq_df = seq_df.astype({
            "ID": "Int64",  # nullable integer dtype
            "Polymer_type": "string",
            "Polymer_seq_one_letter_code": "string",
        })

        if entity_id is not None:
            try:
                return seq_df.loc[seq_df["ID"] == int(entity_id), "Polymer_seq_one_letter_code"].item()
            except Exception:
                warnings.warn(f"Could not find entity: {entity_id}. Returning all sequences", UserWarning)
        return seq_df

    def polymer_type(self, entity_id=None):
        """One row per entity: ID, polymer type, one-letter sequence."""
        
        seq_df = self.sequences()
        if entity_id is not None:
            try:
                return seq_df.loc[seq_df["ID"] == int(entity_id), "Polymer_type"].item()
            except:
                warnings.warn(f"Could not find entity: {entity_id}. Returning all information", UserWarning)
        return seq_df

    def sample_info(self):
        """One row per sample component (flattens the _Sample_component loop)."""
        tags = ["ID", "Sample_ID", "Mol_common_name", "Entity_ID",
                "Isotopic_labeling", "Concentration_val", "Concentration_val_units"]
        return self._loop_records("sample", "_Sample_component", tags, id_key="sample")

    def _sample_rows(self, entity_id=None, sample_id=None):
        """sample_info() filtered (as strings) by sample_id and/or entity_id."""
        df = self.sample_info()
        if df.empty:
            return df
        if sample_id is not None:
            df = df[df["Sample_ID"].astype("string") == str(sample_id)]
        if entity_id is not None:
            df = df[df["Entity_ID"].astype("string") == str(entity_id)]
        return df

    @staticmethod
    def _any_keyword(series, keywords):
        vals = series.dropna().astype("string").str.lower()
        return bool(vals.apply(lambda s: any(k in s for k in keywords)).any())

    def is_deuterated(self, entity_id=None, sample_id=None):
        """
        True if the entity's isotopic labeling indicates 2H/deuteration.
        """
        rows = self._sample_rows(entity_id, sample_id)
        return self._any_keyword(rows["Isotopic_labeling"], _DEUTER_KEYWORDS)

    def is_methyl_labeled(self, entity_id=None, sample_id=None):
        """
        True if the isotopic labeling indicates methyl / selective labeling
        """
        rows = self._sample_rows(entity_id, sample_id)
        return self._any_keyword(rows["Isotopic_labeling"], _METHYL_KEYWORDS)

    def is_denatured(self, sample_id=None):
        """
        True if the sample contains a chemical denaturant (urea or GdnHCl).
        """
        rows = self._sample_rows(sample_id=sample_id)
        return self._any_keyword(rows["Mol_common_name"], _DENATURANT_KEYWORDS)

    def assembly_info(self):
        """One row per entity assembly (flattens the _Entity_assembly loop)."""
        return self._loop_records("assembly", "_Entity_assembly",
                                  ["Entity_assembly_name"], id_key="assembly")

    def spectrometers(self):
        """One row per spectrometer: name, manufacturer, model, field strength."""
        tags = ["ID", "Name", "Manufacturer", "Model", "Field_strength"]
        return self._loop_records("NMR_spectrometer_list", "_NMR_spectrometer_view",
                                  tags, id_key="spectrometer_list")
 
    def shift_reference(self):
        """One row per referenced nucleus (how the depositor referenced shifts)."""
        tags = ["Atom_type", "Atom_isotope_number", "Mol_common_name",
                "Chem_shift_val", "Indirect_shift_ratio", "Ref_method", "Ref_type"]
        return self._loop_records("chem_shift_reference", "_Chem_shift_ref",
                                  tags, id_key="reference")
 
    def sample_conditions(self):
        """One row per sample-condition set, with each condition Type as a
        column (e.g. pH, temperature, pressure, ionic_strength) plus its units."""
        rows = []
        for framecode, sf in self.saveframe("sample_conditions").items():
            row = {"sample_conditions": framecode}
            for var in sf.get("_Sample_condition_variable", []):
                ctype = var.get("Type")
                if not ctype or ctype in (".", "?"):
                    continue
                row[ctype] = self._num(var.get("Val"))
                units = self._clean(var.get("Val_units"))
                if units and units.lower() != "na":
                    row[f"{ctype}_units"] = units
            rows.append(row)
        return pd.DataFrame.from_records(rows)

    # dynamics: relaxation, order parameters, H/D exchange

    _RELAX = {
        "T1":    ("heteronucl_T1_relaxation",    "_T1",             "T1"),
        "T2":    ("heteronucl_T2_relaxation",    "_T2",             "T2"),
        "T1RHO": ("heteronucl_T1rho_relaxation", "_T1rho",          "T1rho"),
        "NOE":   ("heteronucl_NOEs",             "_Heteronucl_NOE", "Heteronucl_NOE"),
    }
    _RELAX_ALIASES = {"R1": "T1", "R2": "T2", "R1RHO": "T1RHO",
                      "HETNOE": "NOE", "HETERONUCLNOE": "NOE"}

    def relaxation(self, kind):
        """
        Heteronuclear relaxation data as a tidy DataFrame, one row per residue.

        kind : 'T1'/'R1', 'T2'/'R2', 'T1rho'/'R1rho', or 'NOE' (case-insensitive).
        Columns: Seq_ID, Comp_ID, Atom_ID, Val, Val_err (T2 lists also carry
        Rex_val/Rex_err when present; NOE adds the second atom Seq_ID_2 etc.).
        The source list framecode is the first column, so multiple field
        strengths stay distinct.

        BMRB is inconsistent about the value tag — some entries use the generic
        `Val`/`Val_err`, others the type-prefixed `T1_val`/`T1_val_err` — so
        whichever is present is coalesced into a single `Val`/`Val_err`.
        """
        key = kind.upper().replace("-", "").replace("_", "")
        key = self._RELAX_ALIASES.get(key, key)
        if key not in self._RELAX:
            raise ValueError(
                f"unknown relaxation kind {kind!r}; "
                "choose from T1/R1, T2/R2, T1rho/R1rho, NOE"
            )
        category, loop, prefix = self._RELAX[key]
        df = self._coalesce_value(self.data_loop(category, loop), prefix)

        if key == "NOE":
            keep = ["list", "Seq_ID_1", "Comp_ID_1", "Atom_ID_1",
                    "Seq_ID_2", "Comp_ID_2", "Atom_ID_2", "Val", "Val_err"]
            df = df.reindex(columns=keep).rename(
                columns={"Seq_ID_1": "Seq_ID", "Comp_ID_1": "Comp_ID",
                         "Atom_ID_1": "Atom_ID"})
            return self._coerce(df, ["Seq_ID", "Seq_ID_2", "Val", "Val_err"])

        keep = ["list", "Seq_ID", "Comp_ID", "Atom_ID", "Atom_type", "Val", "Val_err"]
        if "Rex_val" in df.columns and df["Rex_val"].notna().any():
            keep += ["Rex_val", "Rex_err"]
        df = df.reindex(columns=keep)
        return self._coerce(df, ["Seq_ID", "Val", "Val_err", "Rex_val", "Rex_err"])

    _ORDER = {
        "Order":    ("order_parameters",    "_Order",             "Order"),
    }

    def order_parameters(self):
        """
        Model-free order parameters (S2) as a tidy DataFrame, one row per
        residue: Seq_ID, Comp_ID, Atom_ID, S2, S2_err, Tau_e, Rex, Model_fit.
        """

        category, loop, prefix = self._ORDER['Order']
        df = self._coalesce_value(self.data_loop(category, loop), prefix)

        tags = ["Seq_ID", "Comp_ID", "Atom_ID",
                "Order_param_val", "Order_param_val_fit_err",
                "Tau_e_val", "Rex_val", "Model_fit"]
        df = self._loop_records("order_parameters", "_Order_param", tags, id_key="list")
        df = self._coerce(df, ["Seq_ID", "Order_param_val",
                               "Order_param_val_fit_err", "Tau_e_val", "Rex_val"])
        return df.rename(columns={"Order_param_val": "S2",
                                  "Order_param_val_fit_err": "S2_err",
                                  "Tau_e_val": "Tau_e", "Rex_val": "Rex"})
    def datasets(self):
        """What data the entry contains: one row per data type with its count
        (from the entry_information _Data_set loop). Use this to discover which
        of the methods above will return anything."""
        return self._loop_records("entry_information", "_Data_set",
                                  ["Type", "Count"], id_key="entry")

    def data_loop(self, category, loop_name, tags=None):
        """
        Generic escape hatch: flatten the `loop_name` loop from every saveframe
        of `category` into one DataFrame (framecode in the first column). Keeps
        all columns when `tags` is None. Use for data types without a dedicated
        method (coupling constants, RDCs, spectral density, CSA, ...); inspect a
        saveframe's available loops with `entry.saveframe(category)`.
        """
        frames = []
        for framecode, sf in self.saveframe(category).items():
            if loop_name in sf:
                df = self.loop_to_dataframe(sf[loop_name])
                if tags is not None:
                    df = df.reindex(columns=tags)
                df.insert(0, "list", framecode)
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _coalesce_value(df, prefix):
        """
        Normalise a relaxation loop's value columns to canonical `Val`/`Val_err`.

        BMRB entries use either the generic tag (`Val`) or the type-prefixed one
        (`T1_val`, `T2_val_err`, ...); take whichever is populated. Null markers
        are converted to NaN first so the fallback actually fills.
        """
        if df.empty:
            return df
        df = df.replace({".": np.nan, "?": np.nan, "N/A": np.nan})
        for canon, suffix in [("Val", "val"), ("Val_err", "val_err")]:
            typed = f"{prefix}_{suffix}"           # e.g. T2_val, T2_val_err
            if canon not in df.columns:
                df[canon] = np.nan
            if typed in df.columns:
                df[canon] = df[canon].fillna(df[typed])
                df = df.drop(columns=[typed])
        return df

    @staticmethod
    def _coerce(df, num_cols):
        """Replace BMRB null markers with NaN and coerce `num_cols` to numeric."""
        if df.empty:
            return df
        df = df.replace({".": np.nan, "?": np.nan, "N/A": np.nan})
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    @staticmethod
    def _restructure(data):
        """Index flat {framecode: saveframe} by category then framecode."""
        out = {}
        for v in data.values():
            out.setdefault(v["Sf_category"], {})[v["Sf_framecode"]] = v
        return out

    def __repr__(self):
        return (f"NMRStarEntry(entry_id={self.entry_id!r}, "
                f"categories=[{', '.join(self.data)}])")

    def get_pdb_ids(self):
        pdbs = []
        for sf in self.saveframe("entry_information").values():
            for row in sf.get("_Related_entries", []):
                if row.get("Database_name") == "PDB":
                    code = (row.get("Database_accession_code") or "").strip()
                    if code and code not in (".", "?"):
                        pdbs.append(code)
        # PDB codes also appear cross-referenced on entities
        links = self.db_links()
        if not links.empty:
            for _, row in links.iterrows():
                if (row.get("Database_code") or "").strip().upper() == "PDB":
                    code = (row.get("Accession_code") or "").strip()
                    if code and code not in (".", "?"):
                        pdbs.append(code)
        return list(dict.fromkeys(pdbs))

    def db_links(self, entity_id=None):
        df = self._loop_records(
            "entity", "_Entity_db_link",
            ["Database_code", "Accession_code", "Entity_ID"], id_key="entity")
        if entity_id is not None and not df.empty:
            df = df[df["Entity_ID"].astype("string") == str(entity_id)]
        return df

    def get_alphafold_ids(self, entity_id=None):
        df = self.db_links(entity_id=entity_id)
        if df.empty:
            return []
        alphafold, uniprot = [], []
        for _, row in df.iterrows():
            code = (row.get("Database_code") or "").strip()
            acc = (row.get("Accession_code") or "").strip()
            if not acc or acc in (".", "?"):
                continue
            if code.upper() in ("ALPHAFOLD", "ALPHAFOLDDB", "AFDB"):
                alphafold.append(acc)
            elif code.upper() in _UNIPROT_DBCODES or _UNIPROT_RE.match(acc.upper()):
                uniprot.append(acc)
        return list(dict.fromkeys(alphafold or uniprot))

    # alias: AlphaFold lookup uses UniProt accessions
    get_uniprot_ids = get_alphafold_ids

    def _entry_citation(self):
            """The saveframe marked as the entry citation, else the first citation."""
            cites = self.saveframe("citations")
            for sf in cites.values():
                if sf.get("Class") == "entry citation":
                    return sf
            return next(iter(cites.values()), None)

    def get_entry_title(self):
        """Entry title: from entry_information, falling back to the entry citation."""
        for sf in self.saveframe("entry_information").values():
            try:
                title = self._clean(sf.get("Title"))
                return title
            except:
                continue
        return None

    def get_citation_title(self):
        """Citation title: entry_information first, then the entry citation."""
        for sf in self.saveframe("entry_information").values():
            title = self._clean(sf.get("Title"))
            if title:
                return title
        cite = self._entry_citation()
        return self._clean(cite.get("Title")) if cite else None

    def citation_info(self):
        """Title, journal, year, DOI, PubMed ID, and authors of the entry citation."""
        cite = self._entry_citation() or {}
        authors = [
            f"{self._clean(a.get('Given_name')) or ''} {self._clean(a.get('Family_name')) or ''}".strip()
            for a in cite.get("_Citation_author", [])
        ]
        return {
            "citation_title": self.get_citation_title(),
            "journal": self._clean(cite.get("Journal_name_full"))
                       or self._clean(cite.get("Journal_abbrev")),
            "year": self._clean(cite.get("Year")),
            "pubmed_id": self._clean(cite.get("PubMed_ID")),
            "authors": authors,
        }