import numpy as np
from sympy import Matrix

from logging_utils import get_logger
from linear_algebra import gauss, determinant_mod
from minors import det_mod, brute_force_2x2
from partitioning import compute_partition_data_2x2
from mpi4py import MPI


def make_kernel_from_matrix(non_identity_kernel: np.ndarray, p: int) -> np.ndarray:
    rows, cols = non_identity_kernel.shape
    ker = np.zeros((rows, cols * 2), dtype=object)
    for i in range(rows):
        if i < cols:
            for j in range(cols):
                ker[i, j] = non_identity_kernel[i, j] % p
        ker[i, ker.shape[1] - i - 1] = 1
    return ker % p


def get_submatrix_extended(org: np.ndarray, n: int, p: int = None) -> np.ndarray:
    """
    Port of getSubMatrix_extended: first block is org[n:, n:], then append bottom-right block.
    """
    size = org.shape[0] - n
    mat = np.zeros((size, org.shape[1]), dtype=object)
    # first block
    for i in range(size):
        for j in range(size):
            mat[i, j] = org[i + n, j + n]
    mat_col = org.shape[0] - size
    row = 0
    for i in range(org.shape[0] - size, org.shape[0]):
        col = size
        for j in range(org.shape[1] - mat_col, org.shape[1]):
            mat[row, col] = org[i, j]
            col += 1
        row += 1
    if p is not None:
        mat %= p
    return mat


def is_dependence_found(arr) -> tuple:
    seen = {}
    for idx, val in enumerate(arr):
        if val in seen:
            return True, seen[val], idx
        seen[val] = idx
    return False, -1, -1


def is_2by2_DeterminantZero_2(mat: np.ndarray, pD: dict, orgMat: np.ndarray, p: int) -> tuple:
    """
    Port of is_2by2_DeterminantZero_2: search ratios across rows; returns (found, resultData dict)
    """
    n = mat.shape[0]
    start_row1 = pD["i_start"]
    start_row2 = pD["j_start"]
    for row1 in range(start_row1, n):
        for row2 in range(start_row2, n):
            ratios = []
            zero_col = None
            for col in range(mat.shape[1]):
                if mat[row1, col] % p == 0 or mat[row2, col] % p == 0:
                    return True, {"row1": row1, "row2": row2, "col1": col, "col2": col}
                ratios.append((mat[row1, col] * pow(int(mat[row2, col] % p), -1, p)) % p)
            dep, c1, c2 = is_dependence_found(ratios)
            if dep:
                info = {"row1": row1, "row2": row2, "col1": c1, "col2": c2}
                return True, info
        start_row2 = start_row1 + 2
    return False, {}


def schur_complement_serial(non_identity_kernel: np.ndarray, p: int) -> tuple:
    """
    Faithful port of schurComplement_serial: iterate column reductions, gauss, getSubMatrix_extended, and is_2by2_DeterminantZero_2.
    """
    logger = get_logger("schur_serial")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world = comm.Get_size()
    ker = make_kernel_from_matrix(non_identity_kernel, p)
    kerColCnt = ker.shape[1]
    columnReduceConstant = 1
    columnsToBeReduced = kerColCnt // columnReduceConstant
    cloumnReduceCount = 0
    total_procs = world

    while cloumnReduceCount < columnsToBeReduced:
        ker_gauss = gauss(ker, cloumnReduceCount, p)
        hPrime = get_submatrix_extended(ker_gauss, cloumnReduceCount, p=p)

        pD_all = compute_partition_data_2x2(hPrime.shape[0], total_procs)
        pD = pD_all[rank]
        found, rD = is_2by2_DeterminantZero_2(hPrime, pD, ker_gauss, p)
        any_found = comm.allreduce(int(found), op=MPI.SUM)
        if found:
            logger.info(f"SchurComplement_serial found zero minor at {rD} after reduce {cloumnReduceCount}")
            return True, rD
        if any_found:
            return False, {}
        cloumnReduceCount += 1
    return False, {}


def obliqueElimination(non_identity_kernel: np.ndarray, p: int) -> tuple:
    """
    Faithful port of obliqueElimination: reduceOblique_2 + schurComplement_OE checks.
    """
    logger = get_logger("oblique_full")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world = comm.Get_size()

    ker_full = make_kernel_from_matrix(non_identity_kernel, p)
    mat = ker_full.copy()
    matDim = non_identity_kernel.shape[0]

    def reduceOblique_2(obliqueCnt: int, M: np.ndarray) -> tuple:
        matR = M.shape[0]
        startRow = matR - obliqueCnt
        col = 0
        for i in range(startRow, matR):
            if M[i - 1, col] % p != 0:
                ele = (-M[i, col] * pow(int(M[i - 1, col] % p), -1, p)) % p
                M[i] = (M[i] + ele * M[i - 1]) % p
                col += 1
            else:
                return True, {"row1": i - 1, "row2": i, "col": col}
        return False, {}

    def schurComplement_OE(mat: np.ndarray, obliqueCnt: int, orgMat: np.ndarray) -> tuple:
        rows = mat.shape[0]
        if obliqueCnt <= 1:
            return False, {}
        row1_start = rows - obliqueCnt
        col1_start = obliqueCnt
        for row1 in range(row1_start, rows):
            for row2 in range(row1 + 1, rows):
                for col1 in range(col1_start, rows):
                    for col2 in range(col1 + 1, rows):
                        det = (mat[row1, col1] * mat[row2, col2] - mat[row1, col2] * mat[row2, col1]) % p
                        if det == 0:
                            startRow = rows - obliqueCnt - 1
                            I = list(range(startRow, row1)) + [row1, row2]
                            J = list(range(col1_start)) + [col1, col2]
                            minor = orgMat[np.ix_(I, J)]
                            logger.info(f"OESC zero det obliqueCnt={obliqueCnt} rows={row1,row2} cols={col1,col2}")
                            return True, {"rows": (row1, row2), "cols": (col1, col2)}
        return False, {}

    for i in range(1, (matDim // 2) + 1):
        mat1 = mat.copy()
        found_red, info_red = reduceOblique_2(i, mat1)
        any_red = comm.allreduce(int(found_red), op=MPI.SUM)
        if found_red:
            return True, {"stage": "reduceOblique_2", **info_red}
        if any_red:
            return False, {}

        found_schur, info_schur = schurComplement_OE(mat1, i, ker_full)
        any_schur = comm.allreduce(int(found_schur), op=MPI.SUM)
        if found_schur:
            return True, {"stage": "schurComplement_OE", **info_schur}
        if any_schur:
            return False, {}
    return False, {}
