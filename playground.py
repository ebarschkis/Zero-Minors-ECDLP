from pathlib import Path

from mpi4py import MPI

from apm_port import parse_ntl_matrix
from containment import initCombinations, isLastCombination, _getNextCombination, isMinorPresent
from schur_complement import get_sub_matrix_extended
from schur_complement_serial import make_kernel_from_matrix
from ntl_compat import gauss_mod, set_modulus


def test_determinant():
    p = 33554393
    set_modulus(p)

    file_name = Path("kernel_DB/25_29/25/kernel_c1_25_1.txt")
    if not file_name.exists():
        print(f"\n Error opening file :: {file_name}")
        return

    org_mat = parse_ntl_matrix(file_name.read_text())
    ext_org_mat = make_kernel_from_matrix(org_mat, p)

    columns_reduce_cnt = 0
    while columns_reduce_cnt < 10:
        print(f" column-reduce-cnt :: {columns_reduce_cnt}")
        ker = org_mat.copy()
        org_mat = gauss_mod(org_mat, columns_reduce_cnt, p)
        h_prime = get_sub_matrix_extended(org_mat, columns_reduce_cnt)
        _ = h_prime
        isMinorPresent(org_mat, 3, 3)
        columns_reduce_cnt += 1


def process_small_algo_test(n, dimension, number_of_random_numbers, random_numbers):
    left_side = 0
    right_side = 0

    for i in range(n):
        left_side += random_numbers[i]

    for i in range(n, number_of_random_numbers):
        right_side += random_numbers[i]

    vector = [0] * dimension
    initCombinations(vector, dimension)

    row_number = 6
    row_vec = [0] * row_number
    for i in range(row_number):
        row_vec[i] = (n - 1) - i
    _ = row_vec

    while_cnt = 0
    while not isLastCombination(vector, dimension, n):
        _getNextCombination(vector, n, dimension)
        while_cnt += 1
    print(f"\n while cnt :: {while_cnt}")


def short_algo_test():
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    _ = processor_id

    p = 33554393
    set_modulus(p)

    file_id = 1
    while file_id <= 10:
        file_name = Path(f"kernel_DB/25_29/25/kernel_c1_25_{file_id}_RN.txt")
        print(f"\n fileName :: {file_name}")
        if not file_name.exists():
            print(f"\n Error opening file :: {file_name}")
            return

        tokens = file_name.read_text().split()
        if not tokens:
            return
        number_of_random_numbers = int(tokens[0])
        random_numbers = [int(tokens[i + 1]) for i in range(number_of_random_numbers)]
        print(f"\n number of random numbers :: {number_of_random_numbers}")

        dimension = 3
        n = number_of_random_numbers // 2
        for i in range(dimension, 6):
            print(f"\n Processing dimension :: {i}")
            process_small_algo_test(n, i, number_of_random_numbers, random_numbers)
        break


testDeterminant = test_determinant
shortAlgoTest = short_algo_test
