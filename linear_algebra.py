import numpy as np


def gauss(mat: np.ndarray, start_col: int, p: int) -> np.ndarray:
    """
    Gaussian elimination over GF(p), starting pivots at (start_col,start_col), akin to NTL gauss(mat,w).
    """
    m = mat.copy() % p
    n_rows, n_cols = m.shape
    r = start_col
    for c in range(start_col, n_cols):
        if r >= n_rows:
            break
        pivot = None
        for i in range(r, n_rows):
            if m[i, c] % p != 0:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            m[[r, pivot]] = m[[pivot, r]]
        inv = pow(int(m[r, c] % p), -1, p)
        m[r] = (m[r] * inv) % p
        for i in range(n_rows):
            if i == r:
                continue
            factor = m[i, c] % p
            if factor != 0:
                m[i] = (m[i] - factor * m[r]) % p
        r += 1
    return m % p


def determinant_mod(sub: np.ndarray, p: int) -> int:
    m = sub.copy() % p
    n = m.shape[0]
    if n == 0:
        return 1
    det = 1
    for i in range(n):
        pivot = None
        for r in range(i, n):
            if m[r, i] % p != 0:
                pivot = r
                break
        if pivot is None:
            return 0
        if pivot != i:
            m[[i, pivot]] = m[[pivot, i]]
            det = (-det) % p
        pivot_val = int(m[i, i] % p)
        det = (det * pivot_val) % p
        inv = pow(pivot_val, -1, p)
        for r in range(i + 1, n):
            factor = (m[r, i] * inv) % p
            if factor != 0:
                m[r, i:] = (m[r, i:] - factor * m[i, i:]) % p
    return int(det % p)
