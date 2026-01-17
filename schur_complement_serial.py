import numpy as np
from mpi4py import MPI

from apm_port import load_kernel_from_file
from ntl_compat import gauss_mod, get_modulus
from schur_complement import get_sub_matrix_extended
from search_all_two_minors import PartitionData2x2, ResultData2x2, is_2by2_determinant_zero_2


def make_kernel_from_matrix(ker: np.ndarray, p: int) -> np.ndarray:
    rows, cols = ker.shape
    new_ker = np.zeros((rows, cols * 2), dtype=object)
    for i in range(rows):
        if i < cols:
            for j in range(cols):
                new_ker[i][j] = ker[i][j] % p
        new_ker[i][new_ker.shape[1] - i - 1] = 1
    return new_ker % p


def schur_complement_serial(ord_p: int) -> bool:
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    ker = load_kernel_from_file(processor_id, total, ord_p)
    ker_col_cnt = ker.shape[1]
    column_reduce_constant = 1
    columns_to_be_reduced = ker_col_cnt // column_reduce_constant
    column_reduce_count = 0

    mod = get_modulus()
    ker_full = make_kernel_from_matrix(ker, mod)
    org_mat = ker_full.copy()
    while column_reduce_count < columns_to_be_reduced:
        ker_full = org_mat.copy()
        ker_full = gauss_mod(ker_full, column_reduce_count, mod)

        h_prime = get_sub_matrix_extended(ker_full, column_reduce_count)

        pD = PartitionData2x2(0, 1, -1)
        found, _ = is_2by2_determinant_zero_2(h_prime, pD, ker_full, ord_p)
        if found:
            return True

        column_reduce_count += 1
    return False
