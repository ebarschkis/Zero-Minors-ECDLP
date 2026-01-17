import random

import numpy as np
from mpi4py import MPI

from apm_port import get_random_numbers_from_file, load_kernel_from_file
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
from almost_principle_deviation_serial import make_all_almost_principle_minors


def merger_vectors(apm_dims, block_size, number_of_deviations, block_I, complement_block_I, combination_vec, final_combo):
    block_ptr = 0
    vec_ptr = 0
    for i in range(apm_dims):
        if block_I[block_ptr] <= complement_block_I[combination_vec[vec_ptr]]:
            final_combo[i] = block_I[block_ptr]
            block_ptr += 1
        else:
            final_combo[i] = complement_block_I[combination_vec[vec_ptr]]
            vec_ptr += 1

        if vec_ptr == number_of_deviations:
            for j in range(i + 1, apm_dims):
                final_combo[j] = block_I[block_ptr]
                block_ptr += 1
            break
        if block_ptr == block_size:
            for j in range(i + 1, apm_dims):
                final_combo[j] = complement_block_I[combination_vec[vec_ptr]]
                vec_ptr += 1
            break


def solve_dlp_apm(row_combo, col_combo, dim, file_id, ker, ord_p):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    org_mat_col = ker.shape[1] * 2
    k_rn = (org_mat_col // 2) - 1

    vec = [True if i <= k_rn else False for i in range(org_mat_col)]
    for i in range(dim):
        vec[col_combo[i]] = False
    for i in range(dim):
        vec[(org_mat_col - 1) - row_combo[i]] = True

    random_numbers = get_random_numbers_from_file(file_id, total, ord_p)
    if len(random_numbers) < org_mat_col:
        random_numbers = list(random_numbers) + [0] * (org_mat_col - len(random_numbers))

    A = 0
    for i in range(k_rn):
        if vec[i]:
            A += random_numbers[i]

    B = 0
    for i in range(k_rn, org_mat_col):
        if vec[i]:
            B += random_numbers[i]

    A %= ord_p
    B %= ord_p
    if A == 0 or B == 0:
        print("\n Something is wrong... A or B is zero in getDLP() ... A ::", A, "\t B ::", B)
        return 0
    return (A * pow(int(B), -1, ord_p)) % ord_p


def num_runs(array):
    if not array:
        return 0
    runs = 1
    for i in range(1, len(array)):
        if array[i] != array[i - 1]:
            runs += 1
    return runs


def shuffle_array(array, size):
    rng = random.Random(size)
    for i in range(size):
        j = rng.randrange(i, size)
        array[i], array[j] = array[j], array[i]


def is_valid_combination_pattern4d(row_vec, col_vec, number_of_deviations, dims):
    first_last_percentage = 22
    first_last_value = (first_last_percentage * dims) // 100
    middle_two = (dims - (first_last_value * 2)) // 2

    seg = [0] * 4
    seg[0] = first_last_value
    seg[1] = seg[0] + middle_two
    seg[2] = seg[1] + middle_two
    seg[3] = seg[2] + first_last_value

    if row_vec[0] > seg[0]:
        return False
    if row_vec[3] < seg[3]:
        return False
    if col_vec[0] > seg[0]:
        return False
    if col_vec[3] < seg[3]:
        return False
    return True


def init_combinations_seg(vec_org, seg, seg_size, result_vec):
    start_index = 0
    for i in range(seg_size):
        result_vec[i] = vec_org[start_index]
        start_index += seg[i]


def is_last_combination_seg(vec_org, vec_org_size, v2, v2_size, seg, seg_size):
    index = 0
    for i in range(seg_size):
        index += seg[i]
        if v2[i] != vec_org[index - 1]:
            return False
    return True


def get_next_combination_seg(vec_org, vec_org_size, v2, v2_size, seg, seg_size):
    last_element_seg = [0] * seg_size
    a = 0
    for i in range(seg_size):
        a += seg[i]
        last_element_seg[i] = vec_org[a - 1]

    for i in range(v2_size - 1, -1, -1):
        if v2[i] != last_element_seg[i]:
            v2[i] += 1
            if i != v2_size - 1:
                for j in range(i + 1, v2_size):
                    v2[j] = last_element_seg[j] - seg[j] + 1
            return True
    return False


def get_kth_combination_seg4(n, v, s, s_arr, arr, ip, ans):
    if ip <= 0:
        return False
    ip -= 1

    if ip < arr[0]:
        ans[0] = 0
        ans[1] = s_arr[0]
        ans[2] = s_arr[0] + s_arr[1]
        ans[3] = s_arr[0] + s_arr[1] + s_arr[2] + ip
    elif ip < arr[1]:
        ans[0] = 0
        ans[1] = s_arr[0]
        ans[2] = s_arr[0] + s_arr[1] + (ip // arr[0])
        ans[3] = s_arr[0] + s_arr[1] + s_arr[2] + (ip % s_arr[3])
    elif ip < arr[2]:
        ans[0] = 0
        ans[1] = s_arr[0] + (ip // arr[1]) % s_arr[1]
        ans[2] = s_arr[0] + s_arr[1] + ((ip - arr[1]) // arr[0]) % s_arr[2]
        ans[3] = s_arr[0] + s_arr[1] + s_arr[2] + (ip % s_arr[3])
    else:
        ans[0] = ip // arr[2]
        ans[1] = s_arr[0] + (ip % arr[2]) // arr[1]
        ans[2] = s_arr[0] + s_arr[1] + (ip % arr[1]) // arr[0]
        ans[3] = s_arr[0] + s_arr[1] + s_arr[2] + (ip % s_arr[3])

    if (
        ans[0] < s_arr[0]
        and ans[1] < (s_arr[0] + s_arr[1])
        and ans[2] < (s_arr[0] + s_arr[1] + s_arr[2])
        and ans[3] < (s_arr[0] + s_arr[1] + s_arr[2] + s_arr[3])
    ):
        return True
    return False


def principle_deviation_parallel_8(ord_p):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    for file_id in range(total):
        if processor_id == 0:
            print(f" Processing fileId :: {file_id + 1}\t of {total}")
        ker = load_kernel_from_file(file_id, total, ord_p)

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
                            MPI.COMM_WORLD.Abort(73)

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
    return False


def principle_deviation_parallel_7(ord_p):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    for file_id in range(total):
        if processor_id == 0:
            print(f" Processing fileId :: {file_id + 1}\t of {total}")
        ker = load_kernel_from_file(file_id, total, ord_p)

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
    return False


def principle_deviation_parallel_shuffle_complement_6(ord_p):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    for file_id in range(total):
        if processor_id == 0:
            print(f" Processing fileId :: {file_id + 1}\t of {total}")
        ker = load_kernel_from_file(file_id, total, ord_p)

        mat_rows = ker.shape[0]
        number_of_parts = 6
        max_block_size = mat_rows // number_of_parts
        block_start_dims = 2
        number_of_deviations_start = 2
        number_of_deviations_end = 2

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
                    shuffle_array(complement_block_I, complement_size)

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
    return False


def principle_deviation_parallel_5(ord_p):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    for file_id in range(total):
        if processor_id == 0:
            print(f" Processing fileId :: {file_id + 1}\t of {total}")
        ker = load_kernel_from_file(file_id, total, ord_p)

        mat_rows = ker.shape[0]
        number_of_parts = 6
        max_block_size = mat_rows // number_of_parts
        block_start_dims = 2

        if processor_id == 0:
            print(f"\n ker.r :: {mat_rows}\t ker.c :: {ker.shape[1]}\t max_blockSize :: {max_block_size}")

        for block_size in range(block_start_dims, max_block_size):
            for number_of_deviations in range(2, 4):
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
                number_of_PM = mat_rows - (block_size + number_of_deviations)
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
                    row_combo = [0] * apm_dims
                    for i in range(block_size):
                        row_combo[i] = block_I[i]
                    for i in range(block_size, apm_dims):
                        row_combo[i] = row_combo[i - 1] + 1

                    apm_cnt = 0
                    while apm_cnt < quota:
                        col_combo = [0] * apm_dims
                        block_ptr = 0
                        vec_ptr = 0
                        for i in range(apm_dims):
                            if block_I[block_ptr] <= complement_block_I[col_combination_vec[vec_ptr]]:
                                col_combo[i] = block_I[block_ptr]
                                block_ptr += 1
                            else:
                                col_combo[i] = complement_block_I[col_combination_vec[vec_ptr]]
                                vec_ptr += 1

                            if vec_ptr == number_of_deviations:
                                for j in range(i + 1, apm_dims):
                                    col_combo[j] = block_I[block_ptr]
                                    block_ptr += 1
                                break
                            if block_ptr == block_size:
                                for j in range(i + 1, apm_dims):
                                    col_combo[j] = complement_block_I[col_combination_vec[vec_ptr]]
                                    vec_ptr += 1
                                break

                        minor = makeMatrixFromRowColCombination(row_combo, col_combo, ker)
                        if is_zero(determinant(minor)):
                            print(
                                f"\n by6-APM-3D ZM found @ pId :: {processor_id}\t PM-size :: {block_size}\t APM-size :: {apm_dims}\t PM_cnt :: {PM_cnt - 1} of {number_of_PM}"
                            )
                            printCombination2(row_combo, apm_dims)
                            printCombination2(col_combo, apm_dims)
                            printCombination2(complement_block_I, complement_size)
                            printCombination2(block_I, block_size)
                            printCombination2(row_combination_vec, number_of_deviations)
                            printCombination2(col_combination_vec, number_of_deviations)
                            MPI.COMM_WORLD.Abort(73)

                        if not isLastCombination(col_combination_vec, number_of_deviations, complement_size):
                            _getNextCombination(col_combination_vec, complement_size, number_of_deviations)
                        else:
                            break
                        apm_cnt += 1

                    if processor_id == 0:
                        print(f"\t Time :: {MPI.Wtime() - s_time} seconds ")

                    if not isLastCombination(block_I, block_size, mat_rows):
                        _getNextCombination_continous(block_I, mat_rows, block_size)
                    else:
                        break
                comm.Barrier()
    return False


def principle_deviation_parallel_4(ord_p):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()
    number_of_deviations = 2

    for file_id in range(total):
        if processor_id == 0:
            print(f" Processing fileId :: {file_id + 1}\t of {total}")
        ker = load_kernel_from_file(file_id, total, ord_p)

        mat_rows = ker.shape[0]
        max_block_size = mat_rows - 1
        block_start_dims = 2
        if processor_id == 0:
            print(f"\n ker.r :: {mat_rows}\t ker.c :: {ker.shape[1]}\t max_blockSize :: {max_block_size}")

        for block_size in range(block_start_dims, max_block_size):
            if processor_id == 0:
                print(f" Processing block size :: {block_size}\t of {max_block_size}")

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
            block_J = [0] * block_size
            initCombinations(block_I, block_size)
            initCombinations(block_J, block_size)
            PM_cnt = 1
            number_of_PM = mat_rows - block_size
            while PM_cnt < number_of_PM:
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

                complement_block_J = [0] * complement_size
                cnt = 0
                index = 0
                while cnt < mat_rows:
                    if cnt != block_J[0]:
                        complement_block_J[index] = cnt
                        index += 1
                        cnt += 1
                    else:
                        cnt += block_size

                row_combination_vec = list(row_combination_vec_org)
                col_combination_vec = list(col_combination_vec_org)

                apm_cnt = 0
                while apm_cnt < quota:
                    apm_dims = block_size + number_of_deviations
                    row_combo = [0] * apm_dims
                    block_ptr = 0
                    vec_ptr = 0
                    for i in range(apm_dims):
                        if block_I[block_ptr] <= complement_block_I[row_combination_vec[vec_ptr]]:
                            row_combo[i] = block_I[block_ptr]
                            block_ptr += 1
                        else:
                            row_combo[i] = complement_block_I[row_combination_vec[vec_ptr]]
                            vec_ptr += 1
                        if vec_ptr == number_of_deviations:
                            for j in range(i + 1, apm_dims):
                                row_combo[j] = block_I[block_ptr]
                                block_ptr += 1
                            break
                        if block_ptr == block_size:
                            for j in range(i + 1, apm_dims):
                                row_combo[j] = complement_block_I[row_combination_vec[vec_ptr]]
                                vec_ptr += 1
                            break

                    col_combo = [0] * apm_dims
                    block_ptr = 0
                    vec_ptr = 0
                    for i in range(apm_dims):
                        if block_J[block_ptr] <= complement_block_I[col_combination_vec[vec_ptr]]:
                            col_combo[i] = block_I[block_ptr]
                            block_ptr += 1
                        else:
                            col_combo[i] = complement_block_J[col_combination_vec[vec_ptr]]
                            vec_ptr += 1
                        if vec_ptr == number_of_deviations:
                            for j in range(i + 1, apm_dims):
                                col_combo[j] = block_J[block_ptr]
                                block_ptr += 1
                            break
                        if block_ptr == block_size:
                            for j in range(i + 1, apm_dims):
                                col_combo[j] = complement_block_J[col_combination_vec[vec_ptr]]
                                vec_ptr += 1
                            break

                    minor = makeMatrixFromRowColCombination(row_combo, col_combo, ker)
                    if is_zero(determinant(minor)):
                        print(
                            f"\n FULL-APM ZM found @ pId :: {processor_id}\t PM-size :: {block_size}\t APM-size :: {apm_dims}\t PM_cnt :: {PM_cnt - 1} of {number_of_PM}"
                        )
                        printCombination2(row_combo, apm_dims)
                        printCombination2(col_combo, apm_dims)
                        printCombination2(complement_block_I, complement_size)
                        printCombination2(block_I, block_size)
                        printCombination2(row_combination_vec, number_of_deviations)
                        printCombination2(col_combination_vec, number_of_deviations)
                        MPI.COMM_WORLD.Abort(73)

                    apm_cnt += 1
                    if not isLastCombination(col_combination_vec, number_of_deviations, complement_size):
                        _getNextCombination(col_combination_vec, complement_size, number_of_deviations)
                    else:
                        if isLastCombination(row_combination_vec, number_of_deviations, complement_size):
                            break
                        initCombinations(col_combination_vec, number_of_deviations)
                        _getNextCombination(row_combination_vec, complement_size, number_of_deviations)

                if processor_id == 0:
                    print(f"\t Time :: {MPI.Wtime() - s_time} seconds ")
                comm.Barrier()

                while True:
                    if not isLastCombination(block_J, block_size, mat_rows):
                        _getNextCombination_continous(block_J, mat_rows, block_size)
                    else:
                        if isLastCombination(block_I, block_size, mat_rows):
                            break
                        initCombinations(block_J, block_size)
                        _getNextCombination_continous(block_I, mat_rows, block_size)
                    if block_I[0] != 0 or block_J[0] != 0:
                        continue
                    break
            comm.Barrier()
    return False


def principle_deviation_parallel_3(ord_p):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    for file_id in range(total):
        if processor_id == 0:
            print(f" Processing fileId :: {file_id + 1}\t of {total}")
        ker = load_kernel_from_file(file_id, total, ord_p)

        mat_rows = ker.shape[0]
        number_of_parts = 6
        max_block_size = mat_rows // number_of_parts
        block_start_dims = 2
        number_of_deviations_start = 2
        number_of_deviations_end = 2

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
    return False


def principle_deviation_parallel_2(ord_p):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    for file_id in range(total):
        if processor_id == 0:
            print(f" Processing fileId :: {file_id + 1}\t of {total}")
        ker = load_kernel_from_file(file_id, total, ord_p)

        mat_rows = ker.shape[0]
        number_of_parts = 6
        max_block_size = mat_rows // number_of_parts
        block_start_dims = 2

        if processor_id == 0:
            print(f"\n ker.r :: {mat_rows}\t ker.c :: {ker.shape[1]}\t max_blockSize :: {max_block_size}")

        for block_size in range(block_start_dims, max_block_size):
            time_s = MPI.Wtime()
            if processor_id == 0:
                print(f"\n Processing blockSize :: {block_size} of {max_block_size}")

            number_of_blocks = mat_rows - block_size
            if number_of_blocks <= total:
                number_of_blocks_each = 1
                if processor_id == 0:
                    print(
                        f" Number of PM's :: {number_of_blocks_each}\t Total number of Processors :: {total}"
                    )
                    print(" [Warning] In-efficient CPU utilization...\n")
            else:
                number_of_blocks_each = number_of_blocks // total

            block_start_count = processor_id * number_of_blocks_each
            block_end_count = block_start_count + number_of_blocks_each

            extra_combo = number_of_blocks % total
            if extra_combo >= 1 and processor_id < extra_combo:
                if processor_id != 0:
                    block_start_count = block_start_count + processor_id
                block_end_count = block_end_count + processor_id + 1

            comm.Barrier()
            if processor_id == 0:
                print(
                    f" Total numberOf PM :: {number_of_blocks}\t numberOf PM each Processor gets :: {number_of_blocks_each}"
                )

            block_I = [0] * block_size
            initCombinations(block_I, block_size)
            iteration_cnt = 0
            flag = False
            while not flag:
                if block_start_count <= iteration_cnt < block_end_count:
                    principle_minor = makeMatrixFromRowColCombination(block_I, block_I, ker)
                    dim = 6
                    row = [0] * dim
                    col = [0] * dim
                    flag = make_all_almost_principle_minors(principle_minor, ker, block_size, block_I, dim, row, col)
                    if flag:
                        dlp = solve_dlp_apm(row, col, dim, file_id, ker, ord_p)
                        print("\n DLP ::", dlp, "\t pId ::", processor_id, "\t")
                        print(" Time ::", (MPI.Wtime() - time_s), " Sec. \n")
                        MPI.COMM_WORLD.Abort(73)

                if not isLastCombination(block_I, block_size, mat_rows):
                    _getNextCombination_continous(block_I, mat_rows, block_size)
                else:
                    break
                iteration_cnt += 1

            if processor_id == 0:
                print(f"\n Block-Size :: {block_size}\t Time :: {MPI.Wtime() - time_s} Sec. \n")
        comm.Barrier()
    return False


mergerVectors = merger_vectors
solveDLP_apm = solve_dlp_apm
numRuns = num_runs
shuffleArray = shuffle_array
isValidCombination_pattern4D = is_valid_combination_pattern4d
initCombinations_seg = init_combinations_seg
isLastCombination_seg = is_last_combination_seg
getNextCombination_seg = get_next_combination_seg
get_kth_combination_seg4 = get_kth_combination_seg4
principleDeviation_parallel_8 = principle_deviation_parallel_8
principleDeviation_parallel_7 = principle_deviation_parallel_7
principleDeviation_parallel_shuffleComplement_6 = principle_deviation_parallel_shuffle_complement_6
principleDeviation_parallel_5 = principle_deviation_parallel_5
principleDeviation_parallel_4 = principle_deviation_parallel_4
principleDeviation_parallel_3 = principle_deviation_parallel_3
principleDeviation_parallel_2 = principle_deviation_parallel_2
