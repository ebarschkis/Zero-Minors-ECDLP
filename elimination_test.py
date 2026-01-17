import numpy as np

from containment import (
    _getNextCombination,
    initCombinations,
    isLastCombination,
    makeMatrixFromRowColCombination,
)
from ntl_compat import determinant, is_zero
from search_all_two_minors import elimination_minor

TEN_MILLION = 10000000


def print_combination(vec, dimension):
    print("\t".join(str(vec[i]) for i in range(dimension)))


def get_new_indexes(row_combination, new_combination, mat_row):
    index = 0
    for j in range(mat_row):
        flag = True
        for i in range(mat_row - new_combination):
            if j == row_combination[i]:
                flag = False
                break
        if flag:
            new_combination[index] = j
            index += 1


def make_matrix_from_row_col_combination_2(row_combination, col_combination, mat, minor, row_col_dimension):
    new_combination_size = minor.shape[0]
    new_row_combination = [0] * new_combination_size
    new_col_combination = [0] * new_combination_size

    get_new_indexes(row_combination, new_row_combination, mat.shape[0])
    get_new_indexes(col_combination, new_col_combination, mat.shape[0])
    tmp = makeMatrixFromRowColCombination(new_row_combination, new_col_combination, mat)
    minor[:, :] = tmp


def is_any_column_zero(minor: np.ndarray) -> bool:
    for col in range(minor.shape[1]):
        flag = True
        for row in range(minor.shape[0]):
            if not is_zero(minor[row][col]):
                flag = False
                break
        if flag:
            return True
    return False


def is_any_row_column_zero(minor: np.ndarray) -> bool:
    return is_any_column_zero(minor) or is_any_column_zero(minor.T)


def get_row_sum(mat: np.ndarray) -> list:
    arr = [0] * mat.shape[0]
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            arr[i] += mat[i][j]
    return arr


def elimination_test(mat: np.ndarray, org_mat: np.ndarray, ord_p: int) -> np.ndarray:
    mat_row = mat.shape[0]
    mat_col = mat.shape[1]
    dimension = 1

    if mat_row < 3:
        print(f"\n In Elimination test ... matrix size TOOOOO small... {mat_row}")
        return np.zeros((mat_row - dimension, mat_row - dimension), dtype=object)

    while mat_col > (dimension + 1):
        row_combination = [0] * dimension
        col_combination = [0] * dimension

        print(f"\n dimension :: {dimension}\t sub-matrix-dim :: {mat_row - dimension}\t mat-dim :: {mat_row}")

        initCombinations(row_combination, dimension)
        initCombinations(col_combination, dimension)
        det_cnt = 0
        non_det_cnt = 0
        while_cnt = 0
        while while_cnt < TEN_MILLION:
            while_cnt += 1
            minor = np.zeros((mat_row - dimension, mat_row - dimension), dtype=object)
            make_matrix_from_row_col_combination_2(row_combination, col_combination, mat, minor, dimension)
            if not is_any_row_column_zero(minor):
                new_combination_size = minor.shape[0]
                new_row_combination = [0] * new_combination_size
                new_col_combination = [0] * new_combination_size
                get_new_indexes(row_combination, new_row_combination, mat.shape[0])
                get_new_indexes(col_combination, new_col_combination, mat.shape[0])

                if is_zero(determinant(minor)):
                    if elimination_minor(minor, org_mat, ord_p):
                        print("\n minor :: \n", minor)
                        print("\n determinant(minor) :: ", determinant(minor))
                        arr = get_row_sum(minor)
                        print(" Row-Sum :: ", "\t".join(str(x) for x in arr))
                        print_combination(new_row_combination, new_combination_size)
                        print_combination(new_col_combination, new_combination_size)
                        det_cnt += 1
                    else:
                        non_det_cnt += 1
                else:
                    non_det_cnt += 1

            if not isLastCombination(col_combination, dimension, mat_row):
                _getNextCombination(col_combination, mat_row, dimension)
            else:
                if isLastCombination(row_combination, dimension, mat_row):
                    break
                initCombinations(col_combination, dimension)
                _getNextCombination(row_combination, mat_row, dimension)

        print(f" zero-detCnt :: {det_cnt}\t non_detCnt :: {non_det_cnt}\t whileCnt :: {while_cnt}")
        print(" ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        dimension += 1

    return np.zeros((mat_row - dimension, mat_row - dimension), dtype=object)
