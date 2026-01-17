import time
from typing import List, Optional, TextIO

import numpy as np

from ntl_compat import determinant, get_modulus, is_zero
from row_col import RowCol
from combinatorics import nCr


def write_combination_to_file(row_combination: List[int], col_combination: List[int], fout: TextIO, dimension: int) -> None:
    fout.write(f"{dimension}\n")
    fout.write("\t".join(str(x) for x in row_combination) + "\n")
    fout.write("\t".join(str(x) for x in col_combination) + "\n")
    fout.write("###\n")
    fout.flush()


def _get_next_combination(current_vec: List[int], n: int, r: int) -> None:
    max_vector = [0] * r
    next_vec = [0] * r
    max_vector[r - 1] = n - 1
    for i in range(r - 2, -1, -1):
        max_vector[i] = max_vector[i + 1] - 1
    if current_vec[r - 1] != max_vector[r - 1]:
        next_vec[r - 1] = current_vec[r - 1] + 1
        for i in range(r - 1):
            next_vec[i] = current_vec[i]
    else:
        index = -1
        for i in range(r - 1, -1, -1):
            if current_vec[i] != max_vector[i]:
                index = i
                break
        next_vec[index] = current_vec[index] + 1
        for i in range(r):
            if i < index:
                next_vec[i] = current_vec[i]
            elif i > index:
                next_vec[i] = next_vec[i - 1] + 1
    for i in range(r):
        current_vec[i] = next_vec[i]


def is_last_combination(combination: List[int], dimension: int, n: int) -> bool:
    start = n - dimension
    for i in range(dimension):
        if combination[i] != start + i:
            return False
    return True


def init_combinations(vec: List[int], dimension: int) -> None:
    for i in range(dimension):
        vec[i] = i


def print_combination(vec: List[int], dimension: int) -> None:
    print("\t".join(str(vec[i]) for i in range(dimension)))


def make_matrix_from_row_col_combination(row_combination: List[int], col_combination: List[int], mat: np.ndarray) -> np.ndarray:
    rows = [row_combination[i] for i in range(len(row_combination))]
    cols = [col_combination[i] for i in range(len(col_combination))]
    return mat[np.ix_(rows, cols)]


def process_all_sub_matrices_of_dimension(dimension: int, mat: np.ndarray, fout: TextIO) -> int:
    num_rows, num_cols = mat.shape
    if num_rows < dimension or num_cols < dimension:
        print("\n Invalid dimension in processAllSubMatricesOfDimension() \n")
        return 0

    row_comb = list(range(dimension))
    col_comb = list(range(dimension))
    while_cnt = 0
    zero_minor_cnt = 0

    while True:
        minor = make_matrix_from_row_col_combination(row_comb, col_comb, mat)
        if is_zero(determinant(minor)):
            zero_minor_cnt += 1
            write_combination_to_file(row_comb, col_comb, fout, dimension)

        if not is_last_combination(col_comb, dimension, num_cols):
            _get_next_combination(col_comb, num_cols, dimension)
        else:
            if is_last_combination(row_comb, dimension, num_rows):
                fout.write(f"\n whileCnt :: {while_cnt}\t zeroMinorCnt :: {zero_minor_cnt}\n")
                break
            init_combinations(col_comb, dimension)
            _get_next_combination(row_comb, num_rows, dimension)
        while_cnt += 1

    return zero_minor_cnt


def init_combinations_part_wise(vec: List[int], r: int, seg_len_vec: List[int]) -> None:
    vec[0] = 0
    vec[1] = seg_len_vec[0]
    for i in range(2, r):
        vec[i] = seg_len_vec[i - 2] + seg_len_vec[i - 1]


def make_vector_from_partwise_seg_combination(vec: List[int], dimension: int, main_vec: List[int], r: int, num_rows: int) -> None:
    main_ptr = 0
    vec_cnt = 0
    for i in range(num_rows):
        if main_ptr < r:
            if i != main_vec[main_ptr]:
                vec[vec_cnt] = i
                vec_cnt += 1
            else:
                main_ptr += 1
        else:
            vec[vec_cnt] = i
            vec_cnt += 1


def process_all_sub_matrices_of_dimension_partwise(dimension: int, mat: np.ndarray, fout: TextIO) -> int:
    num_rows, num_cols = mat.shape
    if num_rows <= dimension or num_cols <= dimension:
        print("\n Invalid dimension in processAllSubMatricesOfDimension() \n")
        return 0

    r = num_rows - dimension
    r = 3
    seg_len_vec = [0] * r
    seg_len_vec[0] = 5
    seg_len_vec[1] = int(num_rows * 0.6)
    seg_len_vec[2] = num_rows - seg_len_vec[1] - seg_len_vec[0]

    print(f"\n seg-1 :: {seg_len_vec[0]}\t seg-2 :: {seg_len_vec[1]}\t seg-3 :: {seg_len_vec[2]}")

    main_vec = [0] * r
    init_combinations_part_wise(main_vec, r, seg_len_vec)

    row_comb = [0] * dimension
    make_vector_from_partwise_seg_combination(row_comb, dimension, main_vec, r, num_rows)

    col_comb = list(range(dimension))
    while_cnt = 0
    zero_minor_cnt = 0

    while True:
        minor = make_matrix_from_row_col_combination(row_comb, col_comb, mat)
        if is_zero(determinant(minor)):
            zero_minor_cnt += 1
            write_combination_to_file(row_comb, col_comb, fout, dimension)

        if not is_last_combination(col_comb, dimension, num_cols):
            get_next_combination_part_wise(num_cols, r, main_vec, seg_len_vec)
            make_vector_from_partwise_seg_combination(row_comb, dimension, main_vec, r, num_rows)
        else:
            if is_last_combination_part_wise(main_vec, seg_len_vec, r, num_rows):
                fout.write(f"\n whileCnt :: {while_cnt}\t zeroMinorCnt :: {zero_minor_cnt}\n")
                break
            init_combinations(col_comb, dimension)
            get_next_combination_part_wise(num_cols, r, main_vec, seg_len_vec)
            make_vector_from_partwise_seg_combination(row_comb, dimension, main_vec, r, num_rows)

        while_cnt += 1

    return zero_minor_cnt


def process_all_sub_matrices_of_dimension_parallel(
    dimension: int,
    mat: np.ndarray,
    fout: TextIO,
    row_comb_vec: List[int],
    col_comb_vec: List[int],
    quota: int,
) -> int:
    num_rows, num_cols = mat.shape
    if num_rows <= dimension or num_cols <= dimension:
        print("\n Invalid dimension in processAllSubMatricesOfDimension() \n")
        return 0

    row_comb = row_comb_vec[:dimension]
    col_comb = col_comb_vec[:dimension]
    while_cnt = 0
    zero_minor_cnt = 0

    while while_cnt < quota:
        minor = make_matrix_from_row_col_combination(row_comb, col_comb, mat)
        if is_zero(determinant(minor)):
            zero_minor_cnt += 1
            write_combination_to_file(row_comb, col_comb, fout, dimension)

        if not is_last_combination(col_comb, dimension, num_cols):
            _get_next_combination(col_comb, num_cols, dimension)
        else:
            if is_last_combination(row_comb, dimension, num_rows):
                break
            init_combinations(col_comb, dimension)
            _get_next_combination(row_comb, num_rows, dimension)
        while_cnt += 1

    fout.write(f"\n Total minors processed :: {while_cnt}\t zeroMinorCnt :: {zero_minor_cnt}\n")
    return zero_minor_cnt


def process_all_sub_matrices_of_dimension_all_det(dimension: int, mat: np.ndarray, fout: TextIO) -> int:
    num_rows, num_cols = mat.shape
    if num_rows <= dimension or num_cols <= dimension:
        print("\n Invalid dimension in processAllSubMatricesOfDimension() \n")
        return 0

    row_comb = list(range(dimension))
    col_comb = list(range(dimension))
    while_cnt = 0
    zero_minor_cnt = 0

    modulus = get_modulus()
    det_arr = [0] * modulus
    while True:
        minor = make_matrix_from_row_col_combination(row_comb, col_comb, mat)
        det = int(determinant(minor)) % modulus
        det_arr[det] += 1
        if det == 0:
            zero_minor_cnt += 1

        if not is_last_combination(col_comb, dimension, num_cols):
            _get_next_combination(col_comb, num_cols, dimension)
        else:
            if is_last_combination(row_comb, dimension, num_rows):
                break
            init_combinations(col_comb, dimension)
            _get_next_combination(row_comb, num_rows, dimension)
        while_cnt += 1
    return zero_minor_cnt


def _get_next_combination_continous(current_vec: List[int], n: int, r: int) -> None:
    for i in range(r):
        current_vec[i] = current_vec[i] + 1


def process_all_sub_matrices_of_dimension_continous_row(dimension: int, mat: np.ndarray, fout: TextIO) -> int:
    num_rows, num_cols = mat.shape
    if num_rows <= dimension or num_cols <= dimension:
        print("\n Invalid dimension in processAllSubMatricesOfDimension() \n")
        return 0

    row_comb = list(range(dimension))
    col_comb = list(range(dimension))
    while_cnt = 0
    zero_minor_cnt = 0
    row_cnt = 0
    col_cnt = 0
    while True:
        minor = make_matrix_from_row_col_combination(row_comb, col_comb, mat)
        if is_zero(determinant(minor)):
            zero_minor_cnt += 1
            write_combination_to_file(row_comb, col_comb, fout, dimension)

        if not is_last_combination(col_comb, dimension, num_cols):
            _get_next_combination(col_comb, num_cols, dimension)
            col_cnt += 1
        else:
            if is_last_combination(row_comb, dimension, num_rows):
                fout.write(f"\n whileCnt :: {while_cnt}\t zeroMinorCnt :: {zero_minor_cnt}\n")
                break
            init_combinations(col_comb, dimension)
            col_cnt = 0
            _get_next_combination_continous(row_comb, num_rows, dimension)
            row_cnt += 1
        while_cnt += 1

    return zero_minor_cnt


def process_all_sub_matrices_of_dimension_first_continous_row(dimension: int, mat: np.ndarray) -> int:
    num_rows, num_cols = mat.shape
    if num_rows <= dimension or num_cols <= dimension:
        print("\n Invalid dimension in processAllSubMatricesOfDimension() \n")
        return 0

    row_comb = [i + 51 for i in range(dimension)]
    col_comb = list(range(dimension))

    print("\n Initial Row-Col :: \n ")
    print_combination(row_comb, dimension)
    print_combination(col_comb, dimension)
    print()

    zero_minor_cnt = 0
    while True:
        minor = make_matrix_from_row_col_combination(row_comb, col_comb, mat)
        if is_zero(determinant(minor)):
            zero_minor_cnt += 1
            print()
            print_combination(row_comb, dimension)
            print_combination(col_comb, dimension)
            print("\n Minor found... DLP should be solved...\n")
            return 1

        if not is_last_combination(col_comb, dimension, num_cols):
            _get_next_combination(col_comb, num_cols, dimension)
        else:
            break

    return 0


def process_all_sub_matrices_of_dimension_maximal_minor(dimension: int, mat: np.ndarray) -> int:
    num_rows, num_cols = mat.shape
    if num_rows <= dimension or num_cols <= dimension:
        print("\n Invalid dimension in processAllSubMatricesOfDimension() \n")
        return 0

    row_comb = list(range(dimension))
    col_comb = list(range(dimension))

    print("\n Initial Row-Col :: \n ")
    print_combination(row_comb, dimension)
    print_combination(col_comb, dimension)
    print()

    while_cnt = 0
    zero_minor_cnt = 0

    while True:
        minor = make_matrix_from_row_col_combination(row_comb, col_comb, mat)
        if is_zero(determinant(minor)):
            zero_minor_cnt += 1
            print("\n Zero minor - row:col \n")
            print_combination(row_comb, dimension)
            print_combination(col_comb, dimension)

        if not is_last_combination(col_comb, dimension, num_cols):
            _get_next_combination(col_comb, num_cols, dimension)
        else:
            if is_last_combination(row_comb, dimension, num_rows):
                print(f"\n whileCnt :: {while_cnt}\t zeroMinorCnt :: {zero_minor_cnt}")
                break
            init_combinations(col_comb, dimension)
            _get_next_combination(row_comb, num_rows, dimension)
        while_cnt += 1

    return zero_minor_cnt


def is_minor_present(mat: np.ndarray, start: int, end: int) -> int:
    with open("out.txt", "w") as fout:
        flag = True
        for i in range(start, end + 1):
            print(f" Processing minors of dim :: {i}\t mat_row :: {mat.shape[0]}", end="")
            s_time = time.time()
            count = process_all_sub_matrices_of_dimension(i, mat, fout)
            if count:
                print(f"\t count :: {count}\t time :: {time.time() - s_time} Sec. ")
                flag = True
            else:
                flag = False
        return int(flag)


def all_determinant_enumeration(mat: np.ndarray, start: int, end: int) -> int:
    with open("out.txt", "w") as fout:
        flag = True
        for i in range(start, end + 1):
            print(f"\n Processing minors of dim :: {i}\t mat_row :: {mat.shape[0]}", end="")
            s_time = time.time()
            count = process_all_sub_matrices_of_dimension_all_det(i, mat, fout)
            if count:
                print(f"\t ZM count :: {count}\t time :: {time.time() - s_time} Sec. ")
                flag = True
            else:
                flag = False
        return int(flag)


def get_first_minor(mat: np.ndarray, dimension: int) -> RowCol:
    num_rows, _ = mat.shape
    if num_rows <= dimension:
        print("\n Invalid dimension in processAllSubMatricesOfDimension() \n")
        return RowCol(0)

    row_comb = list(range(dimension))
    col_comb = list(range(dimension))

    rc = RowCol(dimension)
    for i in range(dimension):
        rc.rows[i] = row_comb[i]
        rc.cols[i] = col_comb[i]

    return get_next_minor(mat, rc)


def get_next_minor(mat: np.ndarray, rc: RowCol) -> RowCol:
    num_rows, _ = mat.shape
    if num_rows <= rc.dimension:
        print("\n Invalid dimension in processAllSubMatricesOfDimension() \n")
        return RowCol(0)

    row_comb = rc.rows[:]
    col_comb = rc.cols[:]

    if not is_last_combination(col_comb, rc.dimension, num_rows):
        _get_next_combination(col_comb, num_rows, rc.dimension)
    else:
        if is_last_combination(row_comb, rc.dimension, num_rows):
            return RowCol(0)
        init_combinations(col_comb, rc.dimension)
        _get_next_combination(row_comb, num_rows, rc.dimension)

    while True:
        minor = make_matrix_from_row_col_combination(row_comb, col_comb, mat)
        if is_zero(determinant(minor)):
            rc2 = RowCol(rc.dimension)
            rc2.rows = row_comb[:]
            rc2.cols = col_comb[:]
            return rc2

        if not is_last_combination(col_comb, rc.dimension, num_rows):
            _get_next_combination(col_comb, num_rows, rc.dimension)
        else:
            if is_last_combination(row_comb, rc.dimension, num_rows):
                return RowCol(0)
            init_combinations(col_comb, rc.dimension)
            _get_next_combination(row_comb, num_rows, rc.dimension)


def get_kth_combination_simple(n: int, r: int, k: int, combination: List[int]) -> None:
    symbols = list(range(n))
    result = get_kth_combination_vec(symbols, n, r, k)
    for i in range(r):
        combination[i] = result[i]


def get_kth_combination_vec(vec: List[int], n: int, r: int, index: int) -> List[int]:
    if r < 0 or r > n:
        raise ValueError(f"invalid r={r}")
    c = 1
    k = r if r < (n - r) else (n - r)
    for i in range(1, k + 1):
        c = c * (n - k + i)
        c = c // i
    if index < 0:
        index += c
    if index < 0 or index >= c:
        raise ValueError(f"invalid index={index}")

    result = []
    vec_len = n
    while r:
        c = (c * r) // n
        n -= 1
        r -= 1
        while index >= c:
            index -= c
            c = c * (n - r) // n
            n -= 1
        result.append(vec[vec_len - n - 1])
    return result


def get_kth_combination_vec_z(vec: List[int], n: int, r: int, index: int, result_vec: List[int]) -> bool:
    try:
        res = get_kth_combination_vec(vec, n, r, index)
    except ValueError as exc:
        print(f"\n get_kth_combination :: Index error : {exc}")
        return False
    for i in range(r):
        result_vec[i] = res[i]
    return True


def is_last_combination_part_wise(vec: List[int], seg_len_vec: List[int], r: int, dimension: int) -> bool:
    last_element_seg = [0] * r
    org_main_vec = [0] * r
    org_main_vec[0] = 0
    org_main_vec[1] = seg_len_vec[0]
    for i in range(2, r):
        org_main_vec[i] = seg_len_vec[i - 2] + seg_len_vec[i - 1]

    for i in range(r):
        last_element_seg[i] = org_main_vec[i + 1] - 1
    last_element_seg[r - 1] = dimension - 1

    for i in range(r):
        if last_element_seg[i] != vec[i]:
            return False
    return True


def get_next_combination_part_wise(dimension: int, r: int, main_vec: List[int], seg_len_vec: List[int]) -> bool:
    last_element_seg = [0] * r
    org_main_vec = [0] * r
    org_main_vec[0] = 0
    org_main_vec[1] = seg_len_vec[0]
    for i in range(2, r):
        org_main_vec[i] = seg_len_vec[i - 2] + seg_len_vec[i - 1]

    for i in range(r):
        last_element_seg[i] = org_main_vec[i + 1] - 1
    last_element_seg[r - 1] = dimension - 1

    flag = True
    i = r - 1
    while i >= 0:
        if main_vec[i] != last_element_seg[i]:
            main_vec[i] += 1
            flag = False
            break
        if main_vec[i] == last_element_seg[i]:
            for j in range(i, r):
                main_vec[j] = org_main_vec[j]
        i -= 1
    return flag


# C++-style aliases for parity
writeCombinationToFile = write_combination_to_file
_getNextCombination = _get_next_combination
isLastCombination = is_last_combination
initCombinations = init_combinations
printCombination2 = print_combination
makeMatrixFromRowColCombination = make_matrix_from_row_col_combination
processAllSubMatricesOfDimension = process_all_sub_matrices_of_dimension
initCombinations_partWise = init_combinations_part_wise
makeVectorFrom_partWiseSegCombination = make_vector_from_partwise_seg_combination
processAllSubMatricesOfDimension_partWise = process_all_sub_matrices_of_dimension_partwise
processAllSubMatricesOfDimension_parallel = process_all_sub_matrices_of_dimension_parallel
processAllSubMatricesOfDimension_ALL_Det = process_all_sub_matrices_of_dimension_all_det
_getNextCombination_continous = _get_next_combination_continous
processAllSubMatricesOfDimension_continousRow = process_all_sub_matrices_of_dimension_continous_row
processAllSubMatricesOfDimension_FirstContinousRow = process_all_sub_matrices_of_dimension_first_continous_row
processAllSubMatricesOfDimension_MaximalMinor = process_all_sub_matrices_of_dimension_maximal_minor
isMinorPresent = is_minor_present
allDeterminant_emumeration = all_determinant_enumeration
getFirstMinor = get_first_minor
getNextMinor = get_next_minor
getKth_combination = get_kth_combination_simple
get_kth_combination = get_kth_combination_vec_z
isLastCombination_partWise = is_last_combination_part_wise
getNextCombination_partWise = get_next_combination_part_wise
