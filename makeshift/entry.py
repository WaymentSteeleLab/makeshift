import os
import re
import warnings
import tempfile
import urllib.request
from collections import defaultdict

import pandas as pd

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

    # access 
    
    def sequences(self, entity_id=None):
        """One row per entity: ID, polymer type, one-letter sequence."""
        tags = ["ID", "Polymer_type", "Polymer_seq_one_letter_code"]
        rows = []
        for framecode, sf in self.saveframe("entity").items():
            rows.append({"entity": framecode, **{t: sf.get(t) for t in tags}})
        seq_df = pd.DataFrame.from_records(rows)
        seq_df = seq_df.astype({
            "ID": "Int64",  # nullable integer dtype
            "Polymer_type": "string",
            "Polymer_seq_one_letter_code": "string",
        })
        if entity_id is not None:
            try:
                return seq_df.loc[seq_df["ID"] == int(entity_id), "Polymer_seq_one_letter_code"][0]
            except:
                warnings.warn(f"Could not find entity: {entity_id}. Returning all sequences", UserWarning)
        return pd.DataFrame.from_records(rows)

    def sample_info(self):
        """One row per sample component (flattens the _Sample_component loop)."""
        tags = ["ID", "Sample_ID", "Mol_common_name", "Entity_ID",
                "Isotopic_labeling", "Concentration_val", "Concentration_val_units"]
        return self._loop_records("sample", "_Sample_component", tags, id_key="sample")

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