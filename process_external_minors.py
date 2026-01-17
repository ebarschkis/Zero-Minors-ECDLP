from pathlib import Path

from mpi4py import MPI

from apm_port import parse_ntl_matrix
from containment import (
    makeMatrixFromRowColCombination,
    printCombination2,
    processAllSubMatricesOfDimension,
)
from ntl_compat import set_modulus


def process_all_sub_matrices_of_dimension_modified(I_prime, J_prime, dimension_prime, mat, dimension):
    num_rows = dimension_prime
    if num_rows < dimension:
        print(f"\n Invalid dimension in processAllSubMatricesOfDimension() numRows :: {num_rows}\t dimension :: {dimension}")
        return 0

    row_combination = list(range(dimension))
    col_combination = list(range(dimension))
    while_cnt = 1
    zero_minor_cnt = 0

    row_combination_modified = [0] * dimension
    col_combination_modified = [0] * dimension

    while True:
        for i in range(dimension):
            row_combination_modified[i] = I_prime[row_combination[i]]
            col_combination_modified[i] = J_prime[row_combination[i]]

        minor = makeMatrixFromRowColCombination(row_combination_modified, col_combination_modified, mat)
        if minor is not None and minor.size > 0:
            from ntl_compat import determinant, is_zero

            if is_zero(determinant(minor)):
                print("\n Minor found...\n")
                print("\n minor :: \n ", minor)
                print("\n row/col :: \n")
                printCombination2(row_combination_modified, dimension)
                printCombination2(col_combination_modified, dimension)
                zero_minor_cnt += 1

        if row_combination and col_combination:
            if not (row_combination[-1] == num_rows - 1):
                from containment import _getNextCombination

                _getNextCombination(col_combination, num_rows, dimension)
            else:
                if row_combination[0] == num_rows - dimension:
                    print(f"\n whileCnt :: {while_cnt}\t zeroMinorCnt :: {zero_minor_cnt}")
                    break
                from containment import initCombinations, _getNextCombination

                initCombinations(col_combination, dimension)
                _getNextCombination(row_combination, num_rows, dimension)
        while_cnt += 1

    return zero_minor_cnt


def _fun(I_prime, J_prime, dimension_prime, org_mat, minor):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    start = (minor.shape[0] - 1) - processor_id
    end = 2
    i = start
    while i >= end:
        print(f"\n Processing minors of dim :: {i}\t mat_row :: {org_mat.shape[0]} @ processor :: {processor_id}")
        file_name = Path(f"output/elimination_dim_{i}.txt")
        file_name.parent.mkdir(parents=True, exist_ok=True)
        with file_name.open("w") as fout:
            processAllSubMatricesOfDimension(i, minor, fout)
        i = i - total


def process_external_minors() -> None:
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    p = 33554393
    set_modulus(p)

    file_name = Path("kernel_DB/25_29/25/kernel_c1_25_4.txt")
    print(f"\n fileName :: {file_name}")
    if not file_name.exists():
        print(f"\n Error opening file :: {file_name}")
        return

    org_mat = parse_ntl_matrix(file_name.read_text())

    fin1 = Path("output/3/dimension_72_3_0.txt")
    if not fin1.exists():
        print("\n Unable to open file minor dimension file....\n")
        return

    tokens = fin1.read_text().split()
    idx = 0
    while idx < len(tokens):
        dimension = int(tokens[idx])
        idx += 1
        if idx >= len(tokens):
            break
        print(f"\n Input large minor dimension :: {dimension}")
        I = [int(tokens[idx + i]) for i in range(dimension)]
        idx += dimension
        J = [int(tokens[idx + i]) for i in range(dimension)]
        idx += dimension

        minor = makeMatrixFromRowColCombination(I, J, org_mat)

        dimension_prime = org_mat.shape[0] - dimension
        I_prime = []
        J_prime = []
        for i in range(org_mat.shape[0]):
            if i not in I:
                I_prime.append(i)
            if i not in J:
                J_prime.append(i)

        _fun(I_prime, J_prime, dimension_prime, org_mat, minor)
        print("\n I/J_ prime :: \n")
        printCombination2(I_prime, dimension_prime)
        printCombination2(J_prime, dimension_prime)

        if idx < len(tokens):
            hash_hash = tokens[idx]
            idx += 1
        print("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n")
