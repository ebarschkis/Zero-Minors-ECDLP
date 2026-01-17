import numpy as np
from mpi4py import MPI

from apm_port import get_dlp, is_kernel_having_r_zeros, load_kernel_from_file
from ntl_compat import gauss_mod, get_modulus
from schur_complement_serial import make_kernel_from_matrix


def second_ge(mat: np.ndarray, col: int, p: int) -> None:
    n = mat.shape[0]
    mat_col = mat.shape[1]
    cur_row = 1
    for k in range(n, n + col):
        tmp = mat[cur_row - 1][k]
        for i in range(cur_row, n):
            factor = mat[i][k]
            for j in range(mat_col):
                if tmp % p != 0 and factor % p != 0:
                    if mat[cur_row - 1][j] % p != 0:
                        if tmp % p == 1:
                            mat[i][j] = (factor * mat[cur_row - 1][j] + mat[i][j]) % p
                        else:
                            mat[i][j] = ((factor * pow(int(tmp % p), -1, p)) * mat[cur_row - 1][j] + mat[i][j]) % p
        cur_row += 1


def gaussian_elimination_multiple(ord_p: int) -> bool:
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()
    mod = get_modulus()

    new_ker = load_kernel_from_file(processor_id, total, ord_p)
    ker = make_kernel_from_matrix(new_ker, mod)
    r = ker.shape[1] // 2

    for i in range(1, ker.shape[0]):
        ker = gauss_mod(ker, i, mod)
        found, row_index = is_kernel_having_r_zeros(ker, r)
        if found:
            random_numbers = []
            path = f"randomNumbers/p_{processor_id}_{total}_{ord_p.bit_length()}.txt"
            with open(path) as f:
                count = int(f.readline().strip())
                parts = f.read().strip().split()
            random_numbers = [int(x) for x in parts[:count]]
            k_random_nums = (ker.shape[1] // 2) - 1
            t_random_nums = (ker.shape[1] // 2) + 1
            dlp = get_dlp(ker, row_index, k_random_nums, t_random_nums, random_numbers, ord_p)
            print(f"\nFIRST - G.E. - DLP :: {dlp}\t row-reduce cnt :: {i}\t rowIndex :: {row_index}\t pId :: {processor_id}")
            return True
    return False
