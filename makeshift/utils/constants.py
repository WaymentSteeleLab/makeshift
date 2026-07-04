_AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

_AA_1TO3 = {v: k for k, v in _AA_3TO1.items()}

_BACKBONE = ["H", "HA", "N", "CA", "CB", "C"]

_SS = ("C", "H", "E") 

_DEUTER_KEYWORDS = ("2h", "deuter")
_METHYL_KEYWORDS = ("ilv", "ail", "ile", "leu", "val", "ala", "met", "thr",
                        "sail", "methyl", "ch3", "chd2", "ch2d")

_DENATURANT_KEYWORDS = ("urea", "carbamide", "gdnhcl", "gdmhcl",
                            "guanidinium chloride", "guanidine hydrochloride",
                            "guanidinium hydrochloride", "guanidine hcl", "guanidin", 
                            "gdncl", "gdnscn", "guhcl", "gu-hcl")

_UNIPROT_DBCODES = {"SP", "TR", "UNP", "TREMBL", "UNIPROT", "UNIPROTKB", "SWISS-PROT"}

_NULL_MARKERS = (".", "?", "", "n/a", "na", "<na>", "nan", "none")

import re

_UNIPROT_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$"
    r"|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")

_PDB_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")