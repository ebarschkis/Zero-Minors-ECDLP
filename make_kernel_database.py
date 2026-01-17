from pathlib import Path

import numpy as np
from mpi4py import MPI

from apm_port import (
    Curve,
    Point,
    generate_matrix,
    generate_weighted_vector,
    modular_nullspace,
    save_random_numbers,
    NTLRandomStream,
    generate_random_numbers,
    format_ntl_matrix,
)


def is_kernel_valid(non_identity_kernel: np.ndarray) -> bool:
    if non_identity_kernel.shape[1] < 3:
        return False
    for i in range(non_identity_kernel.shape[0]):
        if non_identity_kernel[i][0] == 0 or non_identity_kernel[i][1] == 0 or non_identity_kernel[i][2] == 0:
            return False
    for i in range(non_identity_kernel.shape[0]):
        for j in range(non_identity_kernel.shape[1]):
            if non_identity_kernel[i][j] == 0:
                return False
    return True


def make_kernel_db(P: Point, Q: Point, ord_p: int, n_param: int, curve: Curve, number_of_kernels: int) -> None:
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    if processor_id != 0:
        return

    print(f"\n Generating {number_of_kernels} kernels on MASTER_NODE => pId :: {processor_id}")
    n = n_param
    r = 3 * n
    k_random_nums = (3 * n) - 1
    t_random_nums = r + 1
    mat_row = r + r
    mat_col = ((n + 1) * (n + 2)) // 2

    weight_vec = generate_weighted_vector(n)
    iteration_cnt = 1
    accident_cnt = 0
    stream = NTLRandomStream(int(time.time()))

    print(f"\n iterationCnt :: {iteration_cnt}\t numberOfkernesTOGenerate :: {number_of_kernels}")
    while iteration_cnt <= number_of_kernels:
        print(f" Processing kernel :: {iteration_cnt}")
        random_nums = generate_random_numbers(mat_row, ord_p, stream)

        M, accident = generate_matrix(curve, P, Q, k_random_nums, t_random_nums, random_nums, weight_vec)
        if accident:
            accident_cnt += 1
            continue

        ker = modular_nullspace(M, curve.p)
        if ker.shape[0] == 0:
            continue
        if ker.shape[0] < r:
            continue

        non_identity_kernel = ker[:, : ker.shape[1] // 2]
        if not is_kernel_valid(non_identity_kernel):
            continue

        kernel_dir = Path("kernel_DB/new")
        kernel_dir.mkdir(parents=True, exist_ok=True)
        kernel_file = kernel_dir / f"kernel_{n_param}_{iteration_cnt}.txt"
        kernel_file.write_text(format_ntl_matrix(non_identity_kernel))

        rn_file = kernel_dir / f"kernel_{n_param}_{iteration_cnt}_RN.txt"
        save_random_numbers(rn_file, random_nums)
        iteration_cnt += 1
