import numpy as np
from mpi4py import MPI

from apm_port import load_kernel_from_file
from ntl_compat import get_modulus


def r_oe(mat: np.ndarray, p: int) -> np.ndarray:
    rows, cols = mat.shape
    max_i = min(rows - 1, cols) - 1
    if max_i < 0:
        return mat
    for i in range(max_i + 1):
        for j in range(i, -1, -1):
            inv = pow(int(mat[j + 1][i - j] % p), -1, p) if mat[j + 1][i - j] % p != 0 else 0
            if inv == 0:
                continue
            e = (mat[j][i - j] * inv) % p
            mat[j] = (mat[j] - e * mat[j + 1]) % p
    return mat


def reverse_oblique_elimination(ord_p: int) -> bool:
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()
    mod = get_modulus()
    ker = load_kernel_from_file(processor_id, total, ord_p)
    r_oe(ker, mod)
    return False
