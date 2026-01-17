from mpi4py import MPI
import numpy as np

from apm_port import load_kernel_from_file
from combinatorics import nCr
from containment import (
    initCombinations,
    isLastCombination,
    _getNextCombination,
    _getNextCombination_continous,
    makeMatrixFromRowColCombination,
    printCombination2,
    get_kth_combination,
)
from ntl_compat import determinant, gauss_mod, get_modulus, is_zero


def _mod_inv(x: int, p: int) -> int:
    return pow(int(x % p), -1, p)


def get_determinant(mat, new_mat, meta_mat, row_mat, col_mat, determinant_val):
    p = get_modulus()
    mat_row = mat.shape[0]
    new_mat = new_mat.copy() % p
    row_mat = row_mat.copy() % p
    col_mat = col_mat.copy() % p
    determinant_val = determinant_val % p

    new_mat[0][mat_row] = col_mat[0][0]
    inv = _mod_inv(new_mat[0][0], p)
    factor = (row_mat[0][0] * inv) % p
    row_mat[0] = (row_mat[0] - (factor * new_mat[0])) % p
    for i in range(1, mat_row):
        for j in range(i):
            new_mat[i][mat_row] = (col_mat[i][0] - (meta_mat[i][j] * col_mat[j][0])) % p
            col_mat[i][0] = new_mat[i][mat_row]
        inv = _mod_inv(new_mat[i][i], p)
        factor = (row_mat[0][i] * inv) % p
        row_mat[0] = (row_mat[0] - (factor * new_mat[i])) % p
    determinant_val = (determinant_val * row_mat[0][mat_row]) % p
    return determinant_val


def get_determinant_generic(new_mat, meta_mat, determinant_val):
    p = get_modulus()
    new_mat = new_mat.copy() % p
    mat_row = meta_mat.shape[0]
    mat_col = meta_mat.shape[1]
    new_mat_row = new_mat.shape[1]

    for i in range(1, mat_row):
        pivot_row = 0
        for j in range(i):
            for k in range(mat_row, new_mat_row):
                new_mat[i][k] = (new_mat[i][k] - (meta_mat[i][j] * new_mat[pivot_row][k])) % p
            pivot_row += 1

    new_mat = gauss_mod(new_mat, new_mat.shape[0] - 1, p)
    for i in range(mat_col, new_mat.shape[0]):
        determinant_val = (determinant_val * new_mat[i][i]) % p
    return determinant_val % p


def get_complement_block(block_size, block_I, mat):
    org_rows = mat.shape[0]
    complement = [0] * (org_rows - block_size)
    cnt = 0
    index = 0
    while cnt < org_rows:
        if cnt != block_I[0]:
            complement[index] = cnt
            index += 1
            cnt += 1
        else:
            cnt += block_size
    return complement


def is_combination_valid(row_combo, col_combo, deviation_of, block_I, block_size, complement_block_I):
    if (complement_block_I[row_combo[0]] < block_I[0]) and (
        complement_block_I[row_combo[deviation_of - 1]] > block_I[block_size - 1]
    ):
        return False

    if (complement_block_I[col_combo[0]] < block_I[0]) and (
        complement_block_I[col_combo[deviation_of - 1]] > block_I[block_size - 1]
    ):
        return False

    if (complement_block_I[row_combo[deviation_of - 1]] < block_I[0]) and (
        complement_block_I[col_combo[0]] > block_I[block_size - 1]
    ):
        return False

    if (complement_block_I[row_combo[0]] > block_I[block_size - 1]) and (
        complement_block_I[col_combo[deviation_of - 1]] < block_I[0]
    ):
        return False

    return True


def deviation_parallel_v2(block_size, block_I, mat, complement_size, complement_block_I, row_combination_org, col_combination_org, quota):
    processor_id = MPI.COMM_WORLD.Get_rank()
    deviation_of = 2

    row_combo = list(row_combination_org[:deviation_of])
    col_combo = list(col_combination_org[:deviation_of])

    apm_rows = block_size + deviation_of
    apm_row = [0] * apm_rows
    apm_col = [0] * apm_rows

    for i in range(block_size):
        apm_row[i] = block_I[i]
        apm_col[i] = block_I[i]

    iteration_cnt = 0
    while True:
        for i in range(deviation_of):
            apm_row[block_size + i] = complement_block_I[row_combo[i]]
            apm_col[block_size + i] = complement_block_I[col_combo[i]]

        apm = makeMatrixFromRowColCombination(apm_row, apm_col, mat)
        if is_zero(determinant(apm)):
            print("\n ######################################### \n")
            print(
                f"\n ### determinant is zero (2D) ...\t blockSize :: {block_size}\t @ pId :: {processor_id}"
            )
            printCombination2(apm_row, apm_rows)
            printCombination2(apm_col, apm_rows)
            print("\n apm :: \n", apm)
            print("\n ######################################### \n")
            MPI.COMM_WORLD.Abort(73)

        iteration_cnt += 1
        if iteration_cnt > quota:
            return False

        if not isLastCombination(col_combo, deviation_of, complement_size):
            _getNextCombination(col_combo, complement_size, deviation_of)
        else:
            if isLastCombination(row_combo, deviation_of, complement_size):
                break
            initCombinations(col_combo, deviation_of)
            _getNextCombination(row_combo, complement_size, deviation_of)
    return False


def deviation_parallel_v2_custom_det(
    block_size,
    block_I,
    mat,
    complement_size,
    complement_block_I,
    row_combination_org,
    col_combination_org,
    quota,
):
    processor_id = MPI.COMM_WORLD.Get_rank()
    deviation_of = 2

    row_combo = list(row_combination_org[:deviation_of])
    col_combo = list(col_combination_org[:deviation_of])

    apm_rows = block_size + deviation_of
    apm_row = [0] * apm_rows
    apm_col = [0] * apm_rows

    for i in range(block_size):
        apm_row[i] = block_I[i]
        apm_col[i] = block_I[i]

    principal_minor = makeMatrixFromRowColCombination(block_I, block_I, mat)
    meta_mat = principal_minor.copy()
    for j in range(1, block_size):
        meta_mat[j][0] = (principal_minor[j][0] * _mod_inv(principal_minor[0][0], get_modulus())) % get_modulus()

    for i in range(1, block_size):
        principal_minor = gauss_mod(principal_minor, i, get_modulus())
        for j in range(i + 1, block_size):
            meta_mat[j][i] = (principal_minor[j][i] * _mod_inv(principal_minor[i][i], get_modulus())) % get_modulus()

    determinant_val = 1
    for i in range(block_size):
        determinant_val = (determinant_val * principal_minor[i][i]) % get_modulus()

    new_mat_dim = apm_rows
    new_mat = np.zeros((new_mat_dim, new_mat_dim), dtype=object)
    for i in range(block_size):
        for j in range(block_size):
            new_mat[i][j] = principal_minor[i][j]

    iteration_cnt = 0
    while True:
        for i in range(deviation_of):
            apm_row[block_size + i] = complement_block_I[row_combo[i]]
            apm_col[block_size + i] = complement_block_I[col_combo[i]]

        # Copy deviation rows
        cnt = 0
        for i in range(block_size, new_mat_dim):
            for j in range(new_mat_dim):
                new_mat[i][j] = mat[complement_block_I[row_combo[cnt]]][apm_col[j]]
            cnt += 1

        # Copy deviation cols
        for i in range(block_size):
            cnt = 0
            for j in range(block_size, new_mat_dim):
                new_mat[i][j] = mat[block_I[i]][complement_block_I[col_combo[cnt]]]
                cnt += 1

        deter = get_determinant_generic(new_mat, meta_mat, determinant_val)
        if is_zero(deter):
            print("\n ######################################### \n")
            print(
                f"\n ### determinant is zero (2D) ...\t blockSize :: {block_size}\t @ pId :: {processor_id}"
            )
            printCombination2(apm_row, apm_rows)
            printCombination2(apm_col, apm_rows)
            apm = makeMatrixFromRowColCombination(apm_row, apm_col, mat)
            print("\n apm :: \n", apm)
            print("\n ######################################### \n")

        iteration_cnt += 1
        if iteration_cnt > quota:
            return False

        if not isLastCombination(col_combo, deviation_of, complement_size):
            _getNextCombination(col_combo, complement_size, deviation_of)
        else:
            if isLastCombination(row_combo, deviation_of, complement_size):
                break
            initCombinations(col_combo, deviation_of)
            _getNextCombination(row_combo, complement_size, deviation_of)
    return False


def principle_deviation_parallel(ord_p):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    for file_id in range(total):
        mat = load_kernel_from_file(file_id, total, ord_p)
        ker_col_cnt = mat.shape[1]
        if processor_id == 0:
            print(
                f"\n Processing File :: {file_id}\t mat.r :: {mat.shape[0]}\t mat.c :: {ker_col_cnt}"
            )

        mat_rows = mat.shape[0]
        max_block_size = mat_rows // 2

        for block_size in range(2, max_block_size):
            if processor_id == 0:
                print(f" Processing block size :: {block_size} of {max_block_size}", end="")
            total_row_combinations = nCr(mat_rows - block_size, 2)

            if total_row_combinations >= 18446744073709551615:
                print("\n nCr :: ", nCr(mat_rows - block_size, 2))
                print("\n totalRowCombinations :: ", total_row_combinations)
                print(
                    "\n totalRowCombinations > 64 bits.... => principleDeviation_parallel() in almostPrinciple_deviation.tcc "
                )
                print(" The program will now terminate...\n")
                raise SystemExit(0)

            total_combinations = total_row_combinations * total_row_combinations
            quota = total_combinations // total
            extra = total_combinations % total

            symbol_vec_dim = mat_rows - block_size
            symbol_vec = list(range(symbol_vec_dim))

            row_combination_index = (processor_id * quota) // total_row_combinations
            col_combination_index = (processor_id * quota) % total_row_combinations

            row_combination_vec = [0, 0]
            col_combination_vec = [0, 0]
            get_kth_combination(symbol_vec, symbol_vec_dim, 2, row_combination_index, row_combination_vec)
            get_kth_combination(symbol_vec, symbol_vec_dim, 2, col_combination_index, col_combination_vec)

            if processor_id == total - 1:
                quota += extra

            block_I = [0] * block_size
            initCombinations(block_I, block_size)

            complement_size = mat_rows - block_size
            time_s = MPI.Wtime()
            while not isLastCombination(block_I, block_size, mat_rows):
                complement_block_I = get_complement_block(block_size, block_I, mat)
                deviation_parallel_v2(
                    block_size,
                    block_I,
                    mat,
                    complement_size,
                    complement_block_I,
                    row_combination_vec,
                    col_combination_vec,
                    quota,
                )
                _getNextCombination_continous(block_I, mat_rows, block_size)
            MPI.COMM_WORLD.Barrier()
            time_e = MPI.Wtime()
            if processor_id == 0:
                print(f"\t Time :: {time_e - time_s} seconds.")
    return False


getDeterminant = get_determinant
getDeterminant_generic = get_determinant_generic
getComplementBlock = get_complement_block
isCombinationValid = is_combination_valid
deviation_parallel_v2_customDet = deviation_parallel_v2_custom_det
principleDeviation_parallel = principle_deviation_parallel
