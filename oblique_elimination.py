import numpy as np
from mpi4py import MPI
from typing import Tuple

from apm_port import is_kernel_having_r_zeros, load_kernel_from_file
from containment import makeMatrixFromRowColCombination
from elimination_test import print_combination
from ntl_compat import determinant, get_modulus, is_zero
from schur_complement_serial import make_kernel_from_matrix
from search_all_two_minors import ResultData2x2, solve_dlp_row_index


def get_ljk_minor(mat: np.ndarray, k: int, j: int) -> np.ndarray:
    minor = np.zeros((k + 1, k + 1), dtype=object)
    row_1 = 0
    for row in range(j, j + k + 1):
        col_1 = 0
        for col in range(0, k + 1):
            minor[row_1][col_1] = mat[row][col]
            col_1 += 1
        row_1 += 1
    return minor


def is_oblique_elimination_successful(new_ker: np.ndarray, L: np.ndarray, U: np.ndarray) -> bool:
    n = new_ker.shape[0]
    end = n - 1
    for j in range(1, end + 1):
        for k in range(1, n - j + 1):
            minor = get_ljk_minor(L, k - 1, j - 1)
            if is_zero(determinant(minor)):
                print(f"\n L => Procssing - k :: {k - 1}\t j :: {j - 1}")
                print(f"\t det :: {determinant(minor)}")
                return False

    U_transpose = U.T
    for j in range(1, end + 1):
        for k in range(1, n - j + 1):
            minor = get_ljk_minor(U_transpose, k - 1, j - 1).T
            if is_zero(determinant(minor)):
                print(f"\n U => Procssing - k :: {k - 1}\t j :: {j - 1}")
                print(f"\t det :: {determinant(minor)}")
                return False
    return True


def lu_decomposition(a: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    n = a.shape[0]
    l = np.zeros((n, n), dtype=object)
    u = np.zeros((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if j < i:
                l[j][i] = 0
            else:
                l[j][i] = a[j][i]
                for k in range(i):
                    l[j][i] = (l[j][i] - l[j][k] * u[k][i]) % p
        for j in range(n):
            if j < i:
                u[i][j] = 0
            elif j == i:
                u[i][j] = 1
            else:
                inv = pow(int(l[i][i] % p), -1, p) if l[i][i] % p != 0 else 0
                u[i][j] = (a[i][j] * inv) % p
                for k in range(i):
                    u[i][j] = (u[i][j] - (l[i][k] * u[k][j] * inv)) % p
    return l % p, u % p


def oblique_elimination_old(ord_p: int) -> bool:
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()
    mod = get_modulus()
    new_ker = load_kernel_from_file(processor_id, total, ord_p)
    ker = make_kernel_from_matrix(new_ker, mod)
    L, U = lu_decomposition(new_ker, mod)
    if is_oblique_elimination_successful(new_ker, L, U):
        print("\n O.E. is successful...\n")
    else:
        print("\n O.E. is NOT possible...\n")
    return False


def get_number_of_zeros_in_row(mat: np.ndarray, row: int, p: int) -> int:
    cnt = 0
    for i in range(mat.shape[1]):
        if mat[row][i] % p == 0:
            cnt += 1
    return cnt


def reduce_oblique(oblique_cnt: int, M: np.ndarray, p: int) -> np.ndarray:
    n = M.shape[0]
    L = np.zeros((n, n), dtype=object)
    if oblique_cnt < 1 or oblique_cnt >= n:
        print(f"\n in-valid value of obliqueCnt :: {oblique_cnt}")
        return L

    for i in range(n):
        L[i][i] = 1

    number_of_terms = ((oblique_cnt + 1) * oblique_cnt) // 2
    X_i = [0] * (n - 1)
    X_i_cnt = 0
    for _ in range(n - oblique_cnt - 1):
        X_i[X_i_cnt] = 0
        X_i_cnt += 1

    start = n - oblique_cnt
    end = n - 1
    for i, j in zip(range(start, end + 1), range(0, oblique_cnt)):
        numerator = (-M[i][j]) % p
        a_i = 0
        number_of_terms_in_denominator = j + 1
        for k in range(number_of_terms_in_denominator):
            x_i_sum = 1
            l_end = number_of_terms_in_denominator - 1 - k
            for l in range(l_end):
                x_i_sum = (x_i_sum * X_i[(n - 1) - oblique_cnt + k + l]) % p
            mat_row = n - oblique_cnt + k - 1
            a_i = (a_i + M[mat_row][j] * x_i_sum) % p
        if a_i % p == 0:
            print(f" This is going to crash now ... denominator :: {a_i}\t Calulating X_{i}\n")
        X_i[X_i_cnt] = (numerator * pow(int(a_i % p), -1, p)) % p
        X_i_cnt += 1

    row = n - oblique_cnt
    col = row - 1
    for i in range(oblique_cnt):
        L[row][col] = X_i[(n - 1) - oblique_cnt + i] % p
        row += 1
        col += 1

    if number_of_terms > 1:
        start_row = (n + 1) - oblique_cnt
        start_col = start_row - 2
        for row in range(start_row, n):
            mul_element = L[row][row - 1]
            col_end = row - 1
            for col in range(start_col, col_end):
                L[row][col] = (mul_element * L[row - 1][col]) % p

    return L % p


def reduce_oblique_2(oblique_cnt: int, M: np.ndarray, rD: ResultData2x2, p: int) -> bool:
    mat_row = M.shape[0]
    start_row = mat_row - oblique_cnt
    col = 0
    for i in range(start_row, mat_row):
        if M[i - 1][col] % p != 0:
            ele = (-M[i][col] * pow(int(M[i - 1][col] % p), -1, p)) % p
            M[i] = (M[i] + ele * M[i - 1]) % p
            col += 1
        else:
            rD.row1 = i - 1
            rD.col1 = col
            rD.row2 = i
            rD.col2 = col
            return True
    return False


def schur_complement_oe(mat: np.ndarray, oblique_cnt: int, ord_p: int, org_mat: np.ndarray) -> bool:
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    mod = get_modulus()

    row_ = mat.shape[0]
    if oblique_cnt > 1:
        row1_start = mat.shape[0] - oblique_cnt
        col1_start = oblique_cnt
        for row1 in range(row1_start, row_):
            for row2 in range(row1 + 1, row_):
                for col1 in range(col1_start, row_):
                    for col2 in range(col1 + 1, row_):
                        det = (mat[row1][col1] * mat[row2][col2] - mat[row1][col2] * mat[row2][col1]) % mod
                        if det % mod == 0:
                            start_row = mat.shape[0] - oblique_cnt - 1
                            I = [i for i in range(start_row, row1)] + [row1, row2]
                            J = [i for i in range(len(I) - 2)] + [col1, col2]
                            minor = makeMatrixFromRowColCombination(I, J, org_mat)
                            minor2 = makeMatrixFromRowColCombination(I, J, mat)
                            print(f"\n OESC - processorId (ker) :: {processor_id}\t obliqueCnt :: {oblique_cnt}\t orgMat => det :: {determinant(minor)}\t hPrime => det :: {determinant(minor2)}")
                            print(org_mat)
                            print(mat)
                            print("\n minor-originalMat :: \n", minor)
                            print("\n minor-reducedMat :: \n", minor2)
                            print_combination(I, len(I))
                            print_combination(J, len(I))
                            return True
    return False


def oblique_elimination(ord_p: int) -> bool:
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()
    mod = get_modulus()

    new_ker = load_kernel_from_file(processor_id, total, ord_p)
    ker = make_kernel_from_matrix(new_ker, mod)
    mat_dim = new_ker.shape[0]
    mat1 = ker.copy()

    print(f" mat1.r :: {mat1.shape[0]}\t mat1.c :: {mat1.shape[1]}\t Looking for r :: {mat1.shape[1] // 2} zeros in there...")

    for i in range(1, mat_dim // 2):
        rD = ResultData2x2()
        if reduce_oblique_2(i, mat1, rD, mod):
            r = mat1.shape[1] // 2
            found, row_index = is_kernel_having_r_zeros(mat1, r)
            if found:
                print(f"\n DLP solved r :: {r}\t rowIndex :: {row_index}")
                print("\n mat1[rowIndex] :: ", mat1[row_index])
                dlp = solve_dlp_row_index(mat1, ord_p, row_index)
                print(f"\n DLP :: {dlp}")
                return True
        if schur_complement_oe(mat1, i, ord_p, ker):
            return True
    return False
