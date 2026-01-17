from mpi4py import MPI
import numpy as np

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
from ntl_compat import determinant, is_zero
from apm_port import load_kernel_from_file
from almost_principle_deviation_v2 import (
    merger_vectors,
    solve_dlp_apm,
    is_valid_combination_pattern4d,
    init_combinations_seg,
    is_last_combination_seg,
    get_next_combination_seg,
    get_kth_combination_seg4,
)


def process_small(ord_p, ker, file_id):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    mat_rows = ker.shape[0]
    number_of_parts = 6
    max_block_size = mat_rows // number_of_parts
    block_start_dims = 2
    number_of_deviations_start = 2
    number_of_deviations_end = 3

    if processor_id == 0:
        print(f"\n ker.r :: {mat_rows}\t ker.c :: {ker.shape[1]}\t max_blockSize :: {max_block_size}")

    for block_size in range(block_start_dims, max_block_size + 1):
        for number_of_deviations in range(number_of_deviations_start, number_of_deviations_end + 1):
            if processor_id == 0:
                print(
                    f" Processing block size :: {block_size}\t of {max_block_size}\t numberOfDeviations :: {number_of_deviations}"
                )

            number_of_apm_half = nCr(mat_rows - block_size, number_of_deviations)
            number_of_apm = number_of_apm_half * number_of_apm_half
            quota = number_of_apm // total
            extra_combo = number_of_apm % total
            if processor_id == 0:
                quota += extra_combo

            symbol_vec_dim = mat_rows - block_size
            symbol_vec = list(range(symbol_vec_dim))
            row_combination_index = (processor_id * quota) // number_of_apm_half
            col_combination_index = (processor_id * quota) % number_of_apm_half

            row_combination_vec_org = [0] * number_of_deviations
            col_combination_vec_org = [0] * number_of_deviations
            get_kth_combination(symbol_vec, symbol_vec_dim, number_of_deviations, row_combination_index, row_combination_vec_org)
            get_kth_combination(symbol_vec, symbol_vec_dim, number_of_deviations, col_combination_index, col_combination_vec_org)
            comm.Barrier()

            block_I = [0] * block_size
            initCombinations(block_I, block_size)
            PM_cnt = 1
            number_of_PM = mat_rows - block_size
            while PM_cnt <= number_of_PM:
                s_time = MPI.Wtime()
                if processor_id == 0:
                    print(
                        f" Processing PM(PM-size = {block_size}) :: {PM_cnt} of {number_of_PM}",
                        end="",
                    )
                PM_cnt += 1

                complement_size = mat_rows - block_size
                complement_block_I = [0] * complement_size
                cnt = 0
                index = 0
                while cnt < mat_rows:
                    if cnt != block_I[0]:
                        complement_block_I[index] = cnt
                        index += 1
                        cnt += 1
                    else:
                        cnt += block_size

                row_combination_vec = list(row_combination_vec_org)
                col_combination_vec = list(col_combination_vec_org)

                apm_dims = block_size + number_of_deviations
                apm_cnt = 0
                while apm_cnt < quota:
                    row_combo = [0] * apm_dims
                    col_combo = [0] * apm_dims
                    merger_vectors(
                        apm_dims,
                        block_size,
                        number_of_deviations,
                        block_I,
                        complement_block_I,
                        row_combination_vec,
                        row_combo,
                    )
                    merger_vectors(
                        apm_dims,
                        block_size,
                        number_of_deviations,
                        block_I,
                        complement_block_I,
                        col_combination_vec,
                        col_combo,
                    )

                    minor = makeMatrixFromRowColCombination(row_combo, col_combo, ker)
                    if is_zero(determinant(minor)):
                        print(
                            f"\n [by6-APM] ZM found @ pId :: {processor_id}\t PM-size :: {block_size}\t APM-size :: {apm_dims}\t PM_cnt :: {PM_cnt - 1} of {number_of_PM}\t  nod_start :: {number_of_deviations_start}\t nod :: {number_of_deviations}"
                        )
                        printCombination2(row_combo, apm_dims)
                        printCombination2(col_combo, apm_dims)
                        comm.Abort(73)

                    if not isLastCombination(col_combination_vec, number_of_deviations, complement_size):
                        _getNextCombination(col_combination_vec, complement_size, number_of_deviations)
                    else:
                        if isLastCombination(row_combination_vec, number_of_deviations, complement_size):
                            break
                        initCombinations(col_combination_vec, number_of_deviations)
                        _getNextCombination(row_combination_vec, complement_size, number_of_deviations)
                    apm_cnt += 1

                if processor_id == 0:
                    print(f"\t Time :: {MPI.Wtime() - s_time} seconds ")

                if not isLastCombination(block_I, block_size, mat_rows):
                    _getNextCombination_continous(block_I, mat_rows, block_size)
                else:
                    break

            comm.Barrier()


def process_small_parallel_8(ord_p, ker, file_id):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    mat_rows = ker.shape[0]
    number_of_parts = 6
    max_block_size = mat_rows // number_of_parts
    block_start_dims = 2
    number_of_deviations_start = 4
    number_of_deviations_end = 5

    if processor_id == 0:
        print(f"\n ker.r :: {mat_rows}\t ker.c :: {ker.shape[1]}\t max_blockSize :: {max_block_size}")

    for block_size in range(block_start_dims, max_block_size + 1):
        for number_of_deviations in range(number_of_deviations_start, number_of_deviations_end + 1):
            if processor_id == 0:
                print(
                    f" Processing block size :: {block_size}\t of {max_block_size}\t numberOfDeviations :: {number_of_deviations}"
                )

            s = 4
            s_arr = [0] * s
            arr = [0] * (s - 1)
            t1 = mat_rows - block_size
            first_last_percentage = 22
            first_last_value = (first_last_percentage * t1) // 100
            middle_two = (t1 - (first_last_value * 2)) // 2
            s_arr[0] = first_last_value
            s_arr[1] = middle_two
            s_arr[2] = middle_two
            s_arr[3] = first_last_value
            if (s_arr[0] + s_arr[1] + s_arr[2] + s_arr[3]) < t1:
                s_arr[3] += t1 - (s_arr[0] + s_arr[1] + s_arr[2] + s_arr[3])

            number_of_apm_half = s_arr[0] * s_arr[1] * s_arr[2] * s_arr[3]
            number_of_apm = number_of_apm_half * number_of_apm_half
            quota = number_of_apm // total
            extra_combo = number_of_apm % total
            if processor_id == 0:
                quota += extra_combo

            symbol_vec_dim = mat_rows - block_size
            symbol_vec = list(range(symbol_vec_dim))
            row_combination_index = (processor_id * quota) // number_of_apm_half
            col_combination_index = (processor_id * quota) % number_of_apm_half
            row_combination_index += 1
            col_combination_index += 1

            row_combination_vec_org = [0] * number_of_deviations
            col_combination_vec_org = [0] * number_of_deviations

            arr[0] = s_arr[3]
            arr[1] = s_arr[2] * arr[0]
            arr[2] = s_arr[1] * arr[1]
            get_kth_combination_seg4(symbol_vec_dim, symbol_vec, 4, s_arr, arr, row_combination_index, row_combination_vec_org)
            get_kth_combination_seg4(symbol_vec_dim, symbol_vec, 4, s_arr, arr, col_combination_index, col_combination_vec_org)

            block_I = [0] * block_size
            initCombinations(block_I, block_size)
            PM_cnt = 1
            number_of_PM = mat_rows - block_size
            while PM_cnt <= number_of_PM:
                s_time = MPI.Wtime()
                if processor_id == 0:
                    print(
                        f" Processing PM(PM-size = {block_size}) :: {PM_cnt} of {number_of_PM}",
                        end="",
                    )
                PM_cnt += 1

                complement_size = mat_rows - block_size
                complement_block_I = [0] * complement_size
                cnt = 0
                index = 0
                while cnt < mat_rows:
                    if cnt != block_I[0]:
                        complement_block_I[index] = cnt
                        index += 1
                        cnt += 1
                    else:
                        cnt += block_size

                row_combination_vec = list(row_combination_vec_org)
                col_combination_vec = list(col_combination_vec_org)
                apm_dims = block_size + number_of_deviations
                apm_cnt = 0
                while apm_cnt < quota:
                    row_combo = [0] * apm_dims
                    col_combo = [0] * apm_dims
                    merger_vectors(
                        apm_dims,
                        block_size,
                        number_of_deviations,
                        block_I,
                        complement_block_I,
                        row_combination_vec,
                        row_combo,
                    )
                    merger_vectors(
                        apm_dims,
                        block_size,
                        number_of_deviations,
                        block_I,
                        complement_block_I,
                        col_combination_vec,
                        col_combo,
                    )

                    minor = makeMatrixFromRowColCombination(row_combo, col_combo, ker)
                    if is_zero(determinant(minor)):
                        print(
                            f"\n [by6-APM] ZM found @ pId :: {processor_id}\t PM-size :: {block_size}\t APM-size :: {apm_dims}\t PM_cnt :: {PM_cnt - 1} of {number_of_PM}\t  nod_start :: {number_of_deviations_start}\t nod :: {number_of_deviations}"
                        )
                        printCombination2(row_combo, apm_dims)
                        printCombination2(col_combo, apm_dims)
                        dlp = solve_dlp_apm(row_combo, col_combo, apm_dims, file_id, ker, ord_p)
                        print("\n DLP ::", dlp, "\t pId ::", processor_id, "\t", end="")
                        comm.Abort(73)

                    if not is_last_combination_seg(complement_block_I, complement_size, col_combination_vec, s, s_arr, s):
                        get_next_combination_seg(symbol_vec, symbol_vec_dim, col_combination_vec, s, s_arr, s)
                    else:
                        if is_last_combination_seg(complement_block_I, complement_size, row_combination_vec, s, s_arr, s):
                            break
                        init_combinations_seg(symbol_vec, s_arr, s, col_combination_vec)
                        get_next_combination_seg(symbol_vec, symbol_vec_dim, row_combination_vec, s, s_arr, s)
                    apm_cnt += 1

                if processor_id == 0:
                    print(f"\t Time :: {MPI.Wtime() - s_time} seconds ")

                if not isLastCombination(block_I, block_size, mat_rows):
                    _getNextCombination_continous(block_I, mat_rows, block_size)
                else:
                    break
            comm.Barrier()


def process_small_parallel_7(ord_p, ker, file_id):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    mat_rows = ker.shape[0]
    number_of_parts = 6
    max_block_size = mat_rows // number_of_parts
    block_start_dims = 2
    number_of_deviations_start = 4
    number_of_deviations_end = 5

    if processor_id == 0:
        print(f"\n ker.r :: {mat_rows}\t ker.c :: {ker.shape[1]}\t max_blockSize :: {max_block_size}")

    for block_size in range(block_start_dims, max_block_size + 1):
        for number_of_deviations in range(number_of_deviations_start, number_of_deviations_end + 1):
            if processor_id == 0:
                print(
                    f" Processing block size :: {block_size}\t of {max_block_size}\t numberOfDeviations :: {number_of_deviations}"
                )

            number_of_apm_half = nCr(mat_rows - block_size, number_of_deviations)
            number_of_apm = number_of_apm_half * number_of_apm_half
            quota = number_of_apm // total
            extra_combo = number_of_apm % total
            if processor_id == 0:
                quota += extra_combo

            symbol_vec_dim = mat_rows - block_size
            symbol_vec = list(range(symbol_vec_dim))
            row_combination_index = (processor_id * quota) // number_of_apm_half
            col_combination_index = (processor_id * quota) % number_of_apm_half

            row_combination_vec_org = [0] * number_of_deviations
            col_combination_vec_org = [0] * number_of_deviations
            get_kth_combination(symbol_vec, symbol_vec_dim, number_of_deviations, row_combination_index, row_combination_vec_org)
            get_kth_combination(symbol_vec, symbol_vec_dim, number_of_deviations, col_combination_index, col_combination_vec_org)
            comm.Barrier()

            block_I = [0] * block_size
            initCombinations(block_I, block_size)
            PM_cnt = 1
            number_of_PM = mat_rows - block_size
            while PM_cnt <= number_of_PM:
                s_time = MPI.Wtime()
                if processor_id == 0:
                    print(
                        f" Processing PM(PM-size = {block_size}) :: {PM_cnt} of {number_of_PM}",
                        end="",
                    )
                PM_cnt += 1

                complement_size = mat_rows - block_size
                complement_block_I = [0] * complement_size
                cnt = 0
                index = 0
                while cnt < mat_rows:
                    if cnt != block_I[0]:
                        complement_block_I[index] = cnt
                        index += 1
                        cnt += 1
                    else:
                        cnt += block_size

                row_combination_vec = list(row_combination_vec_org)
                col_combination_vec = list(col_combination_vec_org)
                apm_dims = block_size + number_of_deviations
                apm_cnt = 0
                while apm_cnt < quota:
                    if is_valid_combination_pattern4d(
                        row_combination_vec, col_combination_vec, number_of_deviations, complement_size
                    ):
                        row_combo = [0] * apm_dims
                        col_combo = [0] * apm_dims
                        merger_vectors(
                            apm_dims,
                            block_size,
                            number_of_deviations,
                            block_I,
                            complement_block_I,
                            row_combination_vec,
                            row_combo,
                        )
                        merger_vectors(
                            apm_dims,
                            block_size,
                            number_of_deviations,
                            block_I,
                            complement_block_I,
                            col_combination_vec,
                            col_combo,
                        )

                        minor = makeMatrixFromRowColCombination(row_combo, col_combo, ker)
                        if is_zero(determinant(minor)):
                            print(
                                f"\n [by6-APM] ZM found @ pId :: {processor_id}\t PM-size :: {block_size}\t APM-size :: {apm_dims}\t PM_cnt :: {PM_cnt - 1} of {number_of_PM}\t  nod_start :: {number_of_deviations_start}\t nod :: {number_of_deviations}"
                            )
                            printCombination2(row_combo, apm_dims)
                            printCombination2(col_combo, apm_dims)
                            dlp = solve_dlp_apm(row_combo, col_combo, apm_dims, file_id, ker, ord_p)
                            print("\n DLP ::", dlp, "\t pId ::", processor_id, "\t", end="")

                    if not isLastCombination(col_combination_vec, number_of_deviations, complement_size):
                        _getNextCombination(col_combination_vec, complement_size, number_of_deviations)
                    else:
                        if isLastCombination(row_combination_vec, number_of_deviations, complement_size):
                            break
                        initCombinations(col_combination_vec, number_of_deviations)
                        _getNextCombination(row_combination_vec, complement_size, number_of_deviations)
                    apm_cnt += 1

                if processor_id == 0:
                    print(f"\t Time :: {MPI.Wtime() - s_time} seconds ")

                if not isLastCombination(block_I, block_size, mat_rows):
                    _getNextCombination_continous(block_I, mat_rows, block_size)
                else:
                    break

            comm.Barrier()


def delete_remove_odd_index(ord_p, mat, file_id):
    org_row = mat.shape[0]
    org_col = mat.shape[1]
    new_mat = np.zeros((org_row // 2, org_col // 2), dtype=object)
    k = 0
    for i in range(1, org_row, 2):
        l = 0
        for j in range(1, org_col, 2):
            new_mat[k][l] = mat[i][j]
            l += 1
        k += 1
    return new_mat


def delete_remove_some(ord_p, mat, count):
    org_row = mat.shape[0]
    org_col = mat.shape[1]
    new_mat = np.zeros((org_row // 2, org_col // 2), dtype=object)
    flag = True
    k = 0
    for i in range(1, org_row):
        l = 0
        for j in range(1, org_col):
            if flag:
                new_mat[k][l] = mat[i][j]
            l += 1
        k += 1
    return new_mat


def principle_deviation_parallel_3_small(ord_p):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    for file_id in range(total):
        if processor_id == 0:
            print(f" Processing fileId :: {file_id + 1}\t of {total}")
        ker = load_kernel_from_file(file_id, total, ord_p)
        new_ker = delete_remove_odd_index(ord_p, ker, file_id)
        process_small_parallel_7(ord_p, new_ker, file_id)
    return True


process_small_parallel_7 = process_small_parallel_7
process_small_parallel_8 = process_small_parallel_8
delete_removeOddIndex = delete_remove_odd_index
delete_removeSome = delete_remove_some
principleDeviation_parallel_3_small = principle_deviation_parallel_3_small
