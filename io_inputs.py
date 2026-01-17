from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class InputZZp:
    p: int
    e: int
    a: int
    b: int
    Px: int
    Py: int
    Pz: int
    Qx: int
    Qy: int
    Qz: int
    ordP: int
    ordQ: int


def load_dlp_input(path: Path) -> List[InputZZp]:
    """
    Parse prime-field input file format used by dlp_input (p e a b Px Py Qx Qy ordP #).
    """
    entries: List[InputZZp] = []
    with path.open() as f:
        tokens = f.read().strip().split()
    i = 0
    while i < len(tokens):
        p = int(tokens[i]); i += 1
        e = int(tokens[i]); i += 1
        a = int(tokens[i]); i += 1
        b = int(tokens[i]); i += 1
        Px = int(tokens[i]); i += 1
        Py = int(tokens[i]); i += 1
        Pz = 1
        Qx = int(tokens[i]); i += 1
        Qy = int(tokens[i]); i += 1
        Qz = 1
        ordP = int(tokens[i]); i += 1
        ordQ = ordP
        # skip sentinel "#"
        if i < len(tokens) and tokens[i] == "#":
            i += 1
        entries.append(InputZZp(p, e, a, b, Px, Py, Pz, Qx, Qy, Qz, ordP, ordQ))
    return entries
