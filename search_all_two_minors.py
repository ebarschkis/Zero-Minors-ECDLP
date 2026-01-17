import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from mpi4py import MPI

from combinatorics import nCr
from containment import (
    _getNextCombination,
    getNextCombination_partWise,
    initCombinations,
    isLastCombination,
    makeMatrixFromRowColCombination,
    printCombination2,
)
from ntl_compat import determinant, get_modulus, is_zero

RANDOM_NUM_DIR = Path("randomNumbers")


@dataclass
class PartitionData2x2:
    i_start: int
    j_start: int
    quota: int


@dataclass
class ResultData2x2:
    row1: int = 0
    row2: int = 0
    col1: int = 0
    col2: int = 0


def get_dlp_2(ker: np.ndarray, row_index: int, k_random: int, t_random: int, random_numbers: List[int], ord_p: int) -> int:
    dlp_a = 0
    dlp_b = 0
    for k in range(k_random):
        if ker[row_index][k] % ord_p != 0:
            dlp_a += random_numbers[k]
    for k in range(k_random, k_random + t_random):
        if ker[row_index][k] % ord_p != 0:
            dlp_b += random_numbers[k]

    dlp_a %= ord_p
    dlp_b %= ord_p
    if dlp_b == 0:
        return 0
    inv = pow(int(dlp_b), -1, ord_p)
    return (dlp_a * inv) % ord_p


def compute_partition_data_2x2(r: int) -> List[PartitionData2x2]:
    comm = MPI.COMM_WORLD
    total = comm.Get_size()
    combos = nCr(r, 2)
    per = combos // total
    extra = combos - (per * total)

    pD = [PartitionData2x2(0, 1, int(per))]
    cnt = 0
    p_cnt = 1
    quota = int(per)
    for i in range(r):
        for j in range(i + 1, r):
            cnt += 1
            if cnt == quota:
                if p_cnt < total:
                    pD.append(PartitionData2x2(i, j, int(per)))
                    p_cnt += 1
                    cnt = 0
    pD[total - 1].quota += int(extra)
    return pD


def send_partition_data_to_processors_2x2(r: int) -> None:
    comm = MPI.COMM_WORLD
    pD = compute_partition_data_2x2(r)
    for i, pd in enumerate(pD):
        comm.send([pd.i_start, pd.j_start, pd.quota], dest=i, tag=0)


def receive_partition_data_2x2() -> PartitionData2x2:
    comm = MPI.COMM_WORLD
    arr = comm.recv(source=0, tag=0)
    return PartitionData2x2(arr[0], arr[1], arr[2])


def is_dependence_found(arr: List[int]) -> Tuple[bool, int, int]:
    count = len(arr)
    for i in range(count):
        for j in range(i + 1, count):
            if arr[i] == arr[j]:
                return True, i, j
    return False, -1, -1


def is_2by2_determinant_zero(mat: np.ndarray, pD: PartitionData2x2) -> Tuple[bool, ResultData2x2]:
    n, mat_col = mat.shape
    A = pD.i_start
    B = pD.j_start
    cnt = 0
    for row1 in range(A, n):
        for row2 in range(B, n):
            result = []
            for col in range(mat_col):
                if mat[row2][col] % get_modulus() == 0 or mat[row1][col] % get_modulus() == 0:
                    return True, ResultData2x2(row1=row1, row2=row2, col1=col, col2=col)
                result.append((mat[row1][col] * pow(int(mat[row2][col] % get_modulus()), -1, get_modulus())) % get_modulus())

            dep, c1, c2 = is_dependence_found(result)
            if dep:
                return True, ResultData2x2(row1=row1, row2=row2, col1=c1, col2=c2)

            cnt += 1
            if cnt == pD.quota:
                return False, ResultData2x2()
        A = row1 + 1
        B = A + 1
    return False, ResultData2x2()


def get_north_west_element_location(minor: np.ndarray, mat: np.ndarray) -> Tuple[int, int]:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j] == minor[0][0]:
                return i, j
    return 0, 0


def get_index_for_minor(minor: np.ndarray, mat: np.ndarray) -> Tuple[List[int], List[int]]:
    minor_row = minor.shape[0]
    i1, j1 = get_north_west_element_location(minor, mat)
    J = []
    for i in range(j1, mat.shape[1]):
        for j in range(minor.shape[1]):
            if mat[i1][i] == minor[0][j]:
                J.append(i)
                break
    I = []
    for i in range(i1, mat.shape[0]):
        for j in range(minor.shape[0]):
            if mat[i][J[minor_row - 1]] == minor[j][minor.shape[1] - 1]:
                I.append(i)
                break
    return I, J


def load_random_numbers_file(processor_id: int, total: int, ord_p: int) -> List[int]:
    bits = ord_p.bit_length()
    path = RANDOM_NUM_DIR / f"p_{processor_id}_{total}_{bits}.txt"
    data = path.read_text().strip().split()
    count = int(data[0]) if data else 0
    nums = [int(x) for x in data[1 : 1 + count]]
    return nums


def solve_dlp(mat: np.ndarray, ord_p: int) -> int:
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()
    row_index = mat.shape[0] - 1
    org_col = mat.shape[1]
    k_rn = (org_col // 2) - 1
    t_rn = k_rn + 2
    random_numbers = load_random_numbers_file(processor_id, total, ord_p)
    return get_dlp_2(mat, row_index, k_rn, t_rn, random_numbers, ord_p)


def solve_dlp_row_index(mat: np.ndarray, ord_p: int, row_index: int) -> int:
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()
    org_col = mat.shape[1]
    k_rn = (org_col // 2) - 1
    t_rn = k_rn + 2
    random_numbers = load_random_numbers_file(processor_id, total, ord_p)
    return get_dlp_2(mat, row_index, k_rn, t_rn, random_numbers, ord_p)


def gaussian_extended(org_mat: np.ndarray, J: List[int], length: int) -> np.ndarray:
    mat = org_mat.copy()
    for i in range(mat.shape[0]):
        for j in range(i + 1, mat.shape[0]):
            if mat[i][J[i]] % get_modulus() == 0:
                continue
            ele = (mat[j][J[i]] * pow(int(mat[i][J[i]] % get_modulus()), -1, get_modulus())) % get_modulus()
            for k in range(j, mat.shape[0]):
                mat[k] = (mat[k] - (ele * mat[i])) % get_modulus()
    return mat


def get_extended_minor(org_mat: np.ndarray, I: List[int], length: int) -> np.ndarray:
    extended = np.zeros((length, org_mat.shape[1]), dtype=object)
    for i in range(length):
        extended[i] = org_mat[I[i]]
    return extended


def elimination_minor(minor: np.ndarray, org_mat: np.ndarray, ord_p: int) -> bool:
    I, J = get_index_for_minor(minor, org_mat)
    extended = get_extended_minor(org_mat, I, minor.shape[0])
    ge_mat = gaussian_extended(extended, J, minor.shape[0])
    dlp = solve_dlp(ge_mat, ord_p)
    return dlp == 9343


def init_combinations_bigger_minor(vec: List[int], prev_dim: int, dimension: int, h_prime_rows: int) -> None:
    index = 0
    number_of_elements = prev_dim
    for i in range(number_of_elements, dimension):
        while True:
            flag = True
            for j in range(number_of_elements):
                if index == vec[j]:
                    flag = False
                    break
            if flag:
                vec[i] = index
                index += 1
                number_of_elements += 1
                break
            index += 1


def get_next_combination_bigger_minor(vec: List[int], prev_dim: int, dimension: int, h_prime_rows: int) -> None:
    s_s = prev_dim
    d_s = dimension - s_s
    d_arr = [vec[i] for i in range(s_s, dimension)]
    while True:
        _getNextCombination(d_arr, h_prime_rows, d_s)
        flag = True
        for i in range(d_s):
            for j in range(s_s):
                if d_arr[i] == vec[j]:
                    flag = False
                    break
            if not flag:
                break
        if flag:
            for i in range(d_s):
                vec[s_s + i] = d_arr[i]
            break


def is_last_combination_bigger_minor(vec: List[int], prev_dim: int, dimension: int, h_prime_rows: int) -> bool:
    s_s = prev_dim
    d_s = dimension - s_s
    d_arr = [vec[i] for i in range(s_s, dimension)]
    if isLastCombination(d_arr, d_s, h_prime_rows):
        return True
    return False


def get_binary_string(vec: List[int], dimension: int, mat_row: int) -> List[bool]:
    binary = [False] * mat_row
    for i in range(dimension):
        binary[vec[i]] = True
    return binary


def hamming_distance(bin_str1: List[bool], bin_str2: List[bool], mat_row: int) -> int:
    cnt = 0
    for i in range(mat_row):
        if bin_str1[i] != bin_str2[i]:
            cnt += 1
    return cnt


def get_hamming_distance_vec(I: List[int], J: List[int], dimension: int, mat_rows: int) -> int:
    bin1 = get_binary_string(I, dimension, mat_rows)
    bin2 = get_binary_string(J, dimension, mat_rows)
    return hamming_distance(bin1, bin2, mat_rows)


def write_combination_to_log(I_ext: List[int], J_ext: List[int], dimension: int) -> None:
    with open("output/log.txt", "a") as fout:
        fout.write(f"\n dim :: {dimension}\n")
        fout.write(" ".join(str(x) for x in I_ext) + "\n")
        fout.write(" ".join(str(x) for x in J_ext) + "\n")


def process_bigger_minor_of_dimension(I: List[int], J: List[int], prev_dim: int, dimension: int, h_prime: np.ndarray) -> bool:
    if prev_dim == 0:
        return False
    h_prime_rows = h_prime.shape[0]

    I_ext = I[:]
    J_ext = J[:]
    init_combinations_bigger_minor(I_ext, prev_dim, dimension, h_prime_rows)
    init_combinations_bigger_minor(J_ext, prev_dim, dimension, h_prime_rows)

    while True:
        minor = makeMatrixFromRowColCombination(I_ext, J_ext, h_prime)
        if is_zero(determinant(minor)):
            print("\n Hamming Distance :: ", get_hamming_distance_vec(I_ext, J_ext, dimension, h_prime_rows))
            write_combination_to_log(I_ext, J_ext, dimension)
            return True

        if not is_last_combination_bigger_minor(J_ext, prev_dim, dimension, h_prime_rows):
            get_next_combination_bigger_minor(J_ext, prev_dim, dimension, h_prime_rows)
        else:
            if is_last_combination_bigger_minor(I_ext, prev_dim, dimension, h_prime_rows):
                return False
            init_combinations_bigger_minor(J_ext, prev_dim, dimension, h_prime_rows)
            get_next_combination_bigger_minor(I_ext, prev_dim, dimension, h_prime_rows)


def process_bigger_minors(h_prime: np.ndarray, org_mat: np.ndarray, org_row1: int, org_row2: int, org_col1: int, org_col2: int) -> None:
    h_prime_rows = h_prime.shape[0]
    dim = 2
    row = [org_row1, org_row2]
    col = [org_col1, org_col2]
    for dimension in range(3, h_prime_rows + 1):
        print(f"\n Processing minors of dim :: {dimension}\t mat_row :: {h_prime_rows}")
        time_s = time.time()
        flag = process_bigger_minor_of_dimension(row, col, dim, dimension, h_prime)
        if flag:
            break
        dim += 1
        print(f"\n Block-Size :: {dimension}\t Time :: {time.time() - time_s} Sec. \n")


def is_2by2_determinant_zero_2(mat: np.ndarray, pD: PartitionData2x2, org_mat: np.ndarray, ord_p: int) -> Tuple[bool, ResultData2x2]:
    n = mat.shape[0]
    start_row1 = pD.i_start
    start_row2 = pD.j_start
    mod = get_modulus()
    for row1 in range(start_row1, n):
        for row2 in range(start_row2, n):
            ratios = []
            for col in range(mat.shape[1]):
                if mat[row1][col] % mod == 0 or mat[row2][col] % mod == 0:
                    return True, ResultData2x2(row1=row1, row2=row2, col1=col, col2=col)
                ratios.append((mat[row1][col] * pow(int(mat[row2][col] % mod), -1, mod)) % mod)
            dep, c1, c2 = is_dependence_found(ratios)
            if dep:
                return True, ResultData2x2(row1=row1, row2=row2, col1=c1, col2=c2)
        start_row2 = start_row1 + 2
    return False, ResultData2x2()


def search_all_two_minors(ord_p: int) -> None:
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    from apm_port import load_kernel_from_file

    ker = load_kernel_from_file(processor_id, total, ord_p)
    pD = compute_partition_data_2x2(ker.shape[0])[processor_id]
    rD = ResultData2x2()
    found, rD = is_2by2_determinant_zero(ker, pD)
    if found:
        print(f"\n ZM @pId :: {processor_id}\t row1 :: {rD.row1}\t row2 :: {rD.row2}\t col1 :: {rD.col1}\t col2 :: {rD.col2}")
