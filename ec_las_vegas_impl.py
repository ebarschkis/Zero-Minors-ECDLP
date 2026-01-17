from pathlib import Path

import numpy as np
from mpi4py import MPI

from apm_port import (
    generate_weighted_vector,
    generate_random_numbers_for_processors,
    get_random_numbers_from_file,
    modular_nullspace,
    format_ntl_matrix,
)
from ntl_compat import is_zero

MASTER_NODE = 0
KERNEL_DIR = Path("kernel")


def genetateKernels(P, Q, ordP, p_bits, offset, EC):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    n = p_bits
    r = 3 * n
    k_random = r - 1
    t_random = r + 1
    mat_row = r + r
    mat_col = ((n + 1) * (n + 2)) // 2

    weight_vec = generate_weighted_vector(n)
    if processor_id == MASTER_NODE:
        generate_random_numbers_for_processors(mat_row, ordP, total)

    comm.Barrier()
    random_numbers = get_random_numbers_from_file(processor_id, total, ordP)

    M = np.zeros((mat_row, mat_col), dtype=object)
    result = generateMatrix(M, P, Q, k_random, t_random, random_numbers, weight_vec, EC)
    _ = result

    ker = modular_nullspace(M, EC.p)
    saveKernelToFile(ker, ordP)


def generateMatrix(M, P, Q, k_random_nums, t_random_nums, PQ_random_numbers, weighted_vector_arr, EC):
    for i in range(k_random_nums):
        P1 = type(P)()
        EC.scalarMultiplicationDA(P, int(PQ_random_numbers[i]), P1)
        if (P1.x == Q.x) and (P1.y == Q.y):
            return 1
        for j in range(M.shape[1]):
            M[i, j] = (
                pow(P1.x, weighted_vector_arr[j][0], EC.p) * pow(P1.y, weighted_vector_arr[j][1], EC.p)
            ) % EC.p

    for i in range(k_random_nums, k_random_nums + t_random_nums):
        P1 = type(P)()
        EC.pointNegation(Q, P1)
        P2 = type(P)()
        EC.scalarMultiplicationDA(P1, int(PQ_random_numbers[i]), P2)
        if (P2.x == Q.x) and (P2.y == Q.y):
            return 1
        for j in range(M.shape[1]):
            M[i, j] = (
                pow(P2.x, weighted_vector_arr[j][0], EC.p) * pow(P2.y, weighted_vector_arr[j][1], EC.p)
            ) % EC.p
    return 0


def isKernelHaving_r_Zeros(ker, r, row_index_ref=None):
    for i in range(ker.shape[0]):
        zero_cnt = 0
        for j in range(ker.shape[1]):
            if ker[i][j] == 0:
                zero_cnt += 1
        if zero_cnt == r:
            if row_index_ref is not None:
                row_index_ref[0] = i
            return True, i

        if (ker.shape[0] - 1) != zero_cnt:
            print("\n UNUSUAL NUMBER OF ZERO's DETECTED (isKernelHaving_r_Zeros) @ row ::", i)
            print("\n Expected Number of zeros ::", (ker.shape[0] - 1), " found ::", zero_cnt)
            print("\n mat_row ::", ker[i], "\n")
            MPI.COMM_WORLD.Abort(50)
    return False, -1


def getDlp(ker, row_index, k_random_nums, t_random_nums, PQ_random_numbers, ordP):
    dlp_A = 0
    dlp_B = 0
    for k in range(k_random_nums):
        if not is_zero(ker[row_index][k], ordP):
            dlp_A += PQ_random_numbers[k]
    for k in range(k_random_nums, k_random_nums + t_random_nums):
        if not is_zero(ker[row_index][k], ordP):
            dlp_B += PQ_random_numbers[k]

    A = dlp_A % ordP
    B = dlp_B % ordP
    if A != 0 and B != 0:
        return (A * pow(int(B), -1, ordP)) % ordP
    print("\n Something is wrong... A or B is zero in getDLP() ...")
    return 0


def getNonIdentityMatrixFromKernel(ker):
    cols = ker.shape[1] // 2
    return ker[:, :cols]


def saveKernelToFile(ker, ordP):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()
    node_name = MPI.Get_processor_name()

    bits = int(ordP).bit_length()
    file_name = KERNEL_DIR / f"p_{processor_id}_{total}_{bits}.txt"
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)

    non_identity_kernel = getNonIdentityMatrixFromKernel(ker)
    with file_name.open("w") as fout:
        fout.write(format_ntl_matrix(non_identity_kernel) + "\n")
        fout.write(f"{processor_id}\n")
        fout.write(f"{node_name}\n")


genetateKernels = genetateKernels
generateMatrix = generateMatrix
isKernelHaving_r_Zeros = isKernelHaving_r_Zeros
getDlp = getDlp
getNonIdentityMatrixFromKernel = getNonIdentityMatrixFromKernel
saveKernelToFile = saveKernelToFile
