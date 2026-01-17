import numpy as np
from mpi4py import MPI

from apm_port import load_kernel_from_file


def circular_shift_matrix_row(kernel: np.ndarray) -> np.ndarray:
    row, col = kernel.shape
    new_mat = np.zeros((row, col), dtype=object)
    for i in range(row - 1):
        new_mat[i] = kernel[i + 1]
    new_mat[row - 1] = kernel[0]
    return new_mat


def schur_complement_circular_swap(ord_p: int) -> None:
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    for i in range(total):
        ker = load_kernel_from_file(i, total, ord_p)
        for j in range(ker.shape[0]):
            if processor_id == 0:
                if ker.shape[0] != ker.shape[1]:
                    print("\n ker row != col ==> using top-left square submatrix\n")
                    dim = min(ker.shape[0], ker.shape[1])
                    ker = ker[:dim, :dim]
                ker = circular_shift_matrix_row(ker)
