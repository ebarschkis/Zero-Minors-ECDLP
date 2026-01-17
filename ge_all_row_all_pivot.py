import numpy as np
from mpi4py import MPI

from apm_port import get_dlp, load_kernel_from_file
from ntl_compat import gauss_mod, get_modulus, is_zero
from schur_complement_serial import make_kernel_from_matrix


def test_row(mat: np.ndarray, row1: int, col: int, row2: int, ord_p: int, mod: int) -> None:
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    if is_zero(mat[row1][col], mod):
        print("\n Divisior Zero.... Noooooo...\n")
        return

    c = (mat[row2][col] * pow(int(mat[row1][col] % mod), -1, mod)) % mod
    vec = [(mat[row2][i] - (c * mat[row1][i])) % mod for i in range(mat.shape[1])]

    r = mat.shape[1] // 2
    zero_cnt = sum(1 for v in vec if v % mod == 0)
    if zero_cnt == r:
        col2 = None
        for i in range(r):
            if i != col and vec[i] % mod == 0:
                col2 = i
                break
        print(f"\n zeroCnt :: {zero_cnt}\t r :: {r}\t row1 :: {row1}\t col1 :: {col}\t row2 :: {row2}\t col2 :: {col2}")

        new_mat = mat.copy()
        new_mat[row2] = vec

        path = f"randomNumbers/p_{processor_id}_{total}_{ord_p.bit_length()}.txt"
        with open(path) as f:
            count = int(f.readline().strip())
            parts = f.read().strip().split()
        random_numbers = [int(x) for x in parts[:count]]

        row_index = row2
        k_random_nums = (new_mat.shape[1] // 2) - 1
        t_random_nums = (new_mat.shape[1] // 2) + 1
        dlp = get_dlp(new_mat, row_index, k_random_nums, t_random_nums, random_numbers, ord_p)
        print(f"\n DLP :: {dlp} @ pID :: {processor_id}")


def test_element(mat: np.ndarray, row: int, col: int, ord_p: int, mod: int) -> None:
    for i in range(row + 1, mat.shape[0]):
        if i != row:
            test_row(mat, row, col, i, ord_p, mod)


def ge_all_row_all_pivot(ord_p: int) -> None:
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    mod = get_modulus()
    new_ker = load_kernel_from_file(processor_id, total, ord_p)
    ker = make_kernel_from_matrix(new_ker, mod)
    r = ker.shape[1] // 2

    row_cnt = 0
    while row_cnt < r:
        ker = gauss_mod(ker, row_cnt, mod)
        for i in range(row_cnt, ker.shape[0]):
            for j in range(ker.shape[1] // 2):
                test_element(ker, i, j, ord_p, mod)
        row_cnt += 1
