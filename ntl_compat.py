from contextlib import contextmanager
from typing import Optional

import numpy as np

from linear_algebra import determinant_mod, gauss

_ZZP_MODULUS: Optional[int] = None


def set_modulus(p: int) -> None:
    global _ZZP_MODULUS
    _ZZP_MODULUS = int(p)


def get_modulus() -> int:
    if _ZZP_MODULUS is None:
        raise ValueError("ZZ_p modulus is not set")
    return _ZZP_MODULUS


@contextmanager
def zzp_push(p: int):
    global _ZZP_MODULUS
    old = _ZZP_MODULUS
    set_modulus(p)
    try:
        yield
    finally:
        if old is None:
            _ZZP_MODULUS = None
        else:
            _ZZP_MODULUS = old


def is_zero(x: int, p: Optional[int] = None) -> bool:
    mod = p if p is not None else get_modulus()
    return int(x) % mod == 0


def is_one(x: int, p: Optional[int] = None) -> bool:
    mod = p if p is not None else get_modulus()
    return int(x) % mod == 1


def determinant(mat: np.ndarray, p: Optional[int] = None) -> int:
    mod = p if p is not None else get_modulus()
    return determinant_mod(mat, mod)


def gauss_mod(mat: np.ndarray, start_col: int, p: Optional[int] = None) -> np.ndarray:
    mod = p if p is not None else get_modulus()
    return gauss(mat, start_col, mod)
